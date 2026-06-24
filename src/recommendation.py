"""
recommendation.py
-------------------
Turns risk buckets into a recommended business action.

Two layers, same pattern as inventory_risk.py:
1. Single-item functions (recommend_business_action, recommend_reorder_quantity) —
   SAME names as the versions that were inline in agent_tools.py.
   recommend_business_action now optionally accepts holiday_or_promo_flag
   and demand_trend — both default to None, so any existing call without
   them behaves exactly as before. Update agent_tools.py to import from
   here instead of keeping its own copy (same fix pattern as inventory_risk.py:
   delete the inline versions, add
   `from recommendation import recommend_business_action, recommend_reorder_quantity`
   near the top).
2. Batch functions that run across the whole risk table, producing
   data/outputs/action_recommendation.csv — Module 7's actual deliverable.

⚠️ KNOWN GAP: the batch functions expect the output of
   inventory_risk.add_risk_columns() — i.e. a table that already has
   risk_bucket, stock_out_risk, inventory_cover_days, forecast_7_day_demand.
   Run inventory_risk.py first.
"""

import os
from typing import Optional
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. Single-item functions
# ---------------------------------------------------------------------------

def recommend_business_action(
    risk_bucket: str,
    stock_out_risk: bool,
    holiday_or_promo_flag=None,
    demand_trend: Optional[str] = None,
) -> dict:
    """
    Map a risk bucket to a recommended action.
    holiday_or_promo_flag and demand_trend are OPTIONAL extra context —
    omit them and you get the same base behavior as before.
    """
    base_actions = {
        "Critical": "Reorder urgently",
        "Warning": "Replenish / monitor",
        "Normal": "No action",
        "Overstock": "Review stock / consider markdown",
    }
    action = base_actions.get(risk_bucket, "No action")

    # Refine with extra context, only if it's actually provided
    if holiday_or_promo_flag and risk_bucket in ("Warning", "Critical"):
        action = "Pre-stock before promotion/holiday"
    elif demand_trend == "decreasing" and risk_bucket == "Overstock":
        action = "Reduce replenishment"

    approval_required = risk_bucket in ("Critical", "Warning") or action.startswith("Pre-stock")

    return {"recommended_action": action, "approval_required": approval_required}


def recommend_reorder_quantity(forecast_7_day_demand: float, current_inventory: float) -> int:
    """Suggested reorder qty = forecasted demand minus current stock, floored at 0."""
    return max(0, round(forecast_7_day_demand - current_inventory))


# ---------------------------------------------------------------------------
# 2. Batch functions
# ---------------------------------------------------------------------------

REQUIRED_RISK_COLUMNS = [
    "store_id", "product_id", "inventory_level", "forecast_7_day_demand",
    "inventory_cover_days", "risk_bucket", "stock_out_risk",
]


def add_recommendation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add recommended_action, recommended_quantity, and approval_required
    to every row. Expects the output of inventory_risk.add_risk_columns().
    """
    missing = [c for c in REQUIRED_RISK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Run inventory_risk.add_risk_columns() first."
        )

    has_promo = "holiday_or_promo_flag" in df.columns

    recs = df.apply(
        lambda row: recommend_business_action(
            row["risk_bucket"],
            row["stock_out_risk"],
            holiday_or_promo_flag=row["holiday_or_promo_flag"] if has_promo else None,
        ),
        axis=1,
    )
    df["recommended_action"] = recs.apply(lambda r: r["recommended_action"])
    df["approval_required"] = recs.apply(lambda r: r["approval_required"])

    df["recommended_quantity"] = df.apply(
        lambda row: recommend_reorder_quantity(row["forecast_7_day_demand"], row["inventory_level"]),
        axis=1,
    )
    return df


def build_action_recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select/rename columns to match the project doc's output shape, and
    add an action_id + pending approval status for each row.

    ⚠️ ASSUMES one row per store_id/product_id (e.g. latest snapshot only).
    If your table has multiple historical rows per product, action_id
    needs the date in it too, or duplicates will collide:
        out["action_id"] = out["store_id"] + "_" + out["product_id"] + "_" + out["date"].astype(str)
    """
    out = df.copy()
    batch_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out["action_id"] = out["store_id"].astype(str) + "_" + out["product_id"].astype(str) + "_" + batch_timestamp
    out["approval_status"] = "pending"

    columns = [
        "action_id", "store_id", "product_id", "forecast_7_day_demand",
        "inventory_level", "inventory_cover_days", "risk_bucket",
        "recommended_action", "recommended_quantity", "approval_required",
        "approval_status",
    ]
    columns = [c for c in columns if c in out.columns]  # tolerate optional cols missing
    return out[columns].rename(columns={"inventory_level": "current_inventory"})


def get_top_risk_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Bonus helper — directly answers "find the top N products at stock-out
    risk": filters to Critical/Warning, sorts by lowest cover days first.
    Useful for your headline demo question.
    """
    at_risk = df[df["risk_bucket"].isin(["Critical", "Warning"])]
    return at_risk.sort_values("inventory_cover_days").head(n)


def _build_dummy_risk_table():
    """Fake risk table, shaped like inventory_risk.py's real output, for testing today."""
    return pd.DataFrame({
        "store_id": ["S003", "S002", "S003", "S001"],
        "product_id": ["P001", "P004", "P009", "P012"],
        "inventory_level": [35, 200, 500, 12],
        "forecast_7_day_demand": [120, 150, 10, 80],
        "inventory_cover_days": [2.04, 6.7, 100.0, 1.2],
        "risk_bucket": ["Critical", "Normal", "Overstock", "Critical"],
        "stock_out_risk": [True, False, False, True],
        "holiday_or_promo_flag": [0, 0, 0, 1],
    })


if __name__ == "__main__":
    risk_output_path = "data/outputs/risk_output.csv"

    if os.path.exists(risk_output_path):
        print("Found real risk_output.csv — using it.")
        risk_df = pd.read_csv(risk_output_path)
    else:
        print("Real risk_output.csv not ready yet — using dummy data to test the pipeline.")
        risk_df = _build_dummy_risk_table()

    with_recs = add_recommendation_columns(risk_df)
    final_table = build_action_recommendation_table(with_recs)

    print(final_table)

    print("\n--- Top 5 at risk (for the demo question) ---")
    top5 = get_top_risk_products(with_recs, n=5)
    print(top5[["store_id", "product_id", "risk_bucket", "inventory_cover_days"]])

    os.makedirs("data/outputs", exist_ok=True)
    final_table.to_csv("data/outputs/action_recommendation.csv", index=False)
    print("\nSaved to data/outputs/action_recommendation.csv")