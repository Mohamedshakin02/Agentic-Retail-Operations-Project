"""
inventory_risk.py
-------------------
Converts forecast + inventory numbers into business risk buckets.

UPDATE: risk thresholds are now category-aware instead of one fixed
number for every product. The logic: how many days of cover counts as
"critical" should depend on how long it normally takes to restock that
category — 2 days is genuinely scary for Groceries (short shelf life,
fast restock cycle) but not remotely dangerous for Furniture (slow
mover, long supplier lead time). See CATEGORY_RISK_THRESHOLDS below.

Two layers in this file:
1. Single-item functions (calculate_inventory_cover, detect_stockout_risk) —
   SAME names as before. detect_stockout_risk now takes an OPTIONAL
   `category` argument — existing calls without it still work exactly
   as before (falls back to DEFAULT_THRESHOLDS, which equals the old
   hardcoded numbers, so nothing breaks).
2. A batch function (add_risk_columns) that runs this across an entire
   feature table, producing data/outputs/risk_output.csv.

⚠️ KNOWN GAP, READ BEFORE RUNNING ON REAL DATA:
   add_risk_columns() needs a forecast_7_day_demand column, which only
   exists after merging with forecast_output.csv — see
   merge_features_and_forecast() below.

⚠️ THE NUMBERS IN CATEGORY_RISK_THRESHOLDS ARE STARTING ESTIMATES, NOT
   REAL DATA. They're a reasonable judgment call (fast movers get short
   critical windows, slow movers get long ones) — adjust them if you
   learn anything more specific about real lead times, and note in
   docs/model_card.md that these are an assumption, not a measured fact.
   That's a perfectly normal thing to document, not a flaw.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Category-aware thresholds
# ---------------------------------------------------------------------------

# category: (critical_days, warning_days, normal_max_days)
CATEGORY_RISK_THRESHOLDS = {
    "Groceries":   (2, 5, 14),    # short shelf life, frequent/fast restocking
    "Clothing":    (5, 12, 30),   # seasonal, moderate restock cycle
    "Electronics": (5, 12, 30),   # moderate lead time, high cost of stockout
    "Toys":        (5, 12, 30),
    "Furniture":   (10, 25, 60),  # slow-moving, long supplier lead time
}

# Falls back to this if category is missing or not in the dict above.
# These three numbers match the OLD hardcoded values exactly, so any
# code that doesn't pass a category behaves identically to before.
DEFAULT_THRESHOLDS = (2, 5, 21)


def get_thresholds_for_category(category) -> tuple:
    """Look up (critical_days, warning_days, normal_max_days) for a category."""
    if category is None or (isinstance(category, float) and pd.isna(category)):
        return DEFAULT_THRESHOLDS
    return CATEGORY_RISK_THRESHOLDS.get(category, DEFAULT_THRESHOLDS)


# ---------------------------------------------------------------------------
# 1. Single-item functions
# ---------------------------------------------------------------------------

def calculate_inventory_cover(current_inventory: float, average_daily_sales: float) -> dict:
    """
    inventory_cover_days = current_inventory / average_daily_sales
    Returns infinity (not an error) when average_daily_sales is 0 — a
    product with zero recent sales has, by definition, unlimited cover.
    """
    if average_daily_sales <= 0:
        cover_days = float("inf")
    else:
        cover_days = round(current_inventory / average_daily_sales, 2)
    return {"inventory_cover_days": cover_days}


def detect_stockout_risk(
    inventory_cover_days: float,
    forecast_7_day_demand: float,
    current_inventory: float,
    category: str = None,
) -> dict:
    """
    Critical / Warning / Normal / Overstock, using thresholds that vary
    by category (see CATEGORY_RISK_THRESHOLDS). Pass category=None (or
    omit it) to use the old fixed thresholds — fully backward compatible.
    """
    critical_days, warning_days, normal_max_days = get_thresholds_for_category(category)

    if inventory_cover_days < critical_days:
        bucket = "Critical"
    elif inventory_cover_days < warning_days:
        bucket = "Warning"
    elif inventory_cover_days <= normal_max_days:
        bucket = "Normal"
    else:
        bucket = "Overstock"

    stock_out_risk = forecast_7_day_demand > current_inventory

    return {
        "risk_bucket": bucket,
        "stock_out_risk": stock_out_risk,
        "thresholds_used": {
            "critical_days": critical_days,
            "warning_days": warning_days,
            "normal_max_days": normal_max_days,
        },
    }


# ---------------------------------------------------------------------------
# 2. Batch functions
# ---------------------------------------------------------------------------

def merge_features_and_forecast(features_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join the feature table with the forecast output on store_id + product_id.

    ⚠️ ERROR-PRONE SPOT: mismatched types (e.g. "S003" vs S003 as an int,
    or trailing whitespace) make a merge silently DROP rows instead of
    erroring. This forces both to clean strings first.
    """
    for df in (features_df, forecast_df):
        df["store_id"] = df["store_id"].astype(str).str.strip()
        df["product_id"] = df["product_id"].astype(str).str.strip()

    merged = features_df.merge(
        forecast_df[["store_id", "product_id", "forecast_7_day_demand"]],
        on=["store_id", "product_id"],
        how="left",
    )

    missing = merged["forecast_7_day_demand"].isna().sum()
    if missing > 0:
        print(f"⚠️ {missing} rows have no matching forecast row — check that "
              f"store_id/product_id values match exactly between the two files.")
        merged["forecast_7_day_demand"] = merged["forecast_7_day_demand"].fillna(0)

    return merged


def add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add inventory_cover_days, risk_bucket, and stock_out_risk to every row.
    Uses the `category` column if present, otherwise falls back to the
    default thresholds for every row.

    Expects: inventory_level, rolling_7_day_avg_sales, forecast_7_day_demand
    """
    required = ["inventory_level", "rolling_7_day_avg_sales", "forecast_7_day_demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Did you forget to run merge_features_and_forecast() first?"
        )

    has_category = "category" in df.columns

    cover = df.apply(
        lambda row: calculate_inventory_cover(
            row["inventory_level"], row["rolling_7_day_avg_sales"]
        )["inventory_cover_days"],
        axis=1,
    )
    df["inventory_cover_days"] = cover

    risk = df.apply(
        lambda row: detect_stockout_risk(
            row["inventory_cover_days"],
            row["forecast_7_day_demand"],
            row["inventory_level"],
            category=row["category"] if has_category else None,
        ),
        axis=1,
    )
    df["risk_bucket"] = risk.apply(lambda r: r["risk_bucket"])
    df["stock_out_risk"] = risk.apply(lambda r: r["stock_out_risk"])
    return df


def _build_dummy_inputs():
    """Fake feature + forecast tables, including category, for testing today."""
    features_df = pd.DataFrame({
        "store_id": ["S003", "S002", "S003"],
        "product_id": ["P001", "P004", "P009"],
        "category": ["Groceries", "Furniture", "Electronics"],
        "inventory_level": [35, 200, 500],
        "rolling_7_day_avg_sales": [17.2, 30.0, 5.0],
    })
    forecast_df = pd.DataFrame({
        "store_id": ["S003", "S002", "S003"],
        "product_id": ["P001", "P004", "P009"],
        "forecast_7_day_demand": [120, 150, 10],
    })
    return features_df, forecast_df


if __name__ == "__main__":
    features_path = "data/processed/retail_features.csv"
    forecast_path = "data/outputs/forecast_output.csv"

    if os.path.exists(features_path) and os.path.exists(forecast_path):
        print("Found real feature + forecast files — using them.")
        features_df = pd.read_csv(features_path)
        forecast_df = pd.read_csv(forecast_path)
    else:
        print("Real files not ready yet — using dummy data to test the pipeline.")
        features_df, forecast_df = _build_dummy_inputs()

    merged = merge_features_and_forecast(features_df, forecast_df)
    result = add_risk_columns(merged)

    cols = ["store_id", "product_id", "inventory_cover_days", "risk_bucket", "stock_out_risk"]
    if "category" in result.columns:
        cols.insert(2, "category")
    print(result[cols])

    os.makedirs("data/outputs", exist_ok=True)
    result.to_csv("data/outputs/risk_output.csv", index=False)
    print("\nSaved to data/outputs/risk_output.csv")