"""
feature_engineering.py
------------------------
Turns retail_cleaned.csv into a feature table the forecasting model can train on.

STATUS: testable today with a small built-in dummy dataset. The moment A
pushes the real data/processed/retail_cleaned.csv (matching the agreed
column contract), this same code works on it unchanged — see __main__
at the bottom, which auto-detects whether real data exists yet.

NOTE: this was originally B's file. If you're picking it up to keep the
sprint moving, let B and your PM know, so this doesn't collide with B's
own version later.
"""

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "date", "store_id", "product_id", "units_sold",
    "inventory_level", "price",
]


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week, week_of_year, month, is_weekend from the `date` column."""
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek            # 0 = Monday
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag_1_day_sales and lag_7_day_sales.
    Groups by store_id + product_id FIRST, so "yesterday's sales" never
    accidentally pulls from a different product or store.
    """
    df = df.sort_values(["store_id", "product_id", "date"])
    grouped = df.groupby(["store_id", "product_id"])["units_sold"]
    df["lag_1_day_sales"] = grouped.shift(1)
    df["lag_7_day_sales"] = grouped.shift(7)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7/14/28-day rolling average sales, and 7-day rolling inventory."""
    grouped_sales = df.groupby(["store_id", "product_id"])["units_sold"]
    df["rolling_7_day_avg_sales"] = grouped_sales.transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["rolling_14_day_avg_sales"] = grouped_sales.transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["rolling_28_day_avg_sales"] = grouped_sales.transform(lambda s: s.rolling(28, min_periods=1).mean())

    grouped_inv = df.groupby(["store_id", "product_id"])["inventory_level"]
    df["rolling_7_day_inventory"] = grouped_inv.transform(lambda s: s.rolling(7, min_periods=1).mean())
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_change_pct: % change in price vs. the previous day, same product/store."""
    grouped_price = df.groupby(["store_id", "product_id"])["price"]
    df["price_change_pct"] = grouped_price.pct_change().fillna(0) * 100
    return df


def add_inventory_cover_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add inventory_cover_days = inventory_level / rolling_7_day_avg_sales.
    Guards against divide-by-zero for products with no recent sales.
    """
    safe_avg_sales = df["rolling_7_day_avg_sales"].replace(0, np.nan)
    df["inventory_cover_days"] = (df["inventory_level"] / safe_avg_sales).fillna(0).round(2)
    return df


def add_categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode text columns into numbers — fine for tree-based models
    like Random Forest, no need for one-hot encoding here.

    Skips any column whose _encoded version already exists (e.g. a
    teammate's file already has category_encoded, region_encoded, etc.)
    so we never silently overwrite someone else's encoding with a
    different mapping.
    """
    for col in ["category", "region", "weather_condition", "seasonality"]:
        encoded_col = f"{col}_encoded"
        if col in df.columns and encoded_col not in df.columns:
            df[encoded_col] = df[col].astype("category").cat.codes
        elif encoded_col in df.columns:
            print(f"Skipping {encoded_col} — already present, leaving it as is.")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline, in the correct order."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Check the column rename map in data_ingestion.py."
        )

    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_features(df)
    df = add_inventory_cover_feature(df)
    df = add_categorical_encoding(df)
    return df


def _build_dummy_dataset() -> pd.DataFrame:
    """A tiny fake dataset matching the real schema — just for testing today."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date": dates,
        "store_id": ["S003"] * 10,
        "product_id": ["P001"] * 10,
        "units_sold": [12, 15, 9, 20, 18, 14, 11, 25, 22, 19],
        "inventory_level": [100, 88, 79, 59, 41, 27, 16, 10, 31, 50],
        "price": [9.99, 9.99, 9.99, 8.99, 8.99, 8.99, 8.99, 8.99, 9.99, 9.99],
        "category": ["Electronics"] * 10,
        "region": ["East"] * 10,
        "weather_condition": ["Sunny"] * 10,
        "seasonality": ["Winter"] * 10,
    })


if __name__ == "__main__":
    import os

    real_data_path = "data/processed/retail_cleaned.csv"

    if os.path.exists(real_data_path):
        print(f"Found real data at {real_data_path} — using it.")
        raw_df = pd.read_csv(real_data_path)
    else:
        print("Real data not ready yet — using a dummy dataset to test the pipeline.")
        raw_df = _build_dummy_dataset()

    features_df = build_features(raw_df)
    print(features_df[[
        "date", "store_id", "product_id", "units_sold",
        "lag_7_day_sales", "rolling_7_day_avg_sales", "inventory_cover_days"
    ]])
    os.makedirs("data/processed", exist_ok=True)
    features_df.to_csv("data/processed/retail_features.csv", index=False)
    print("\nSaved to data/processed/retail_features.csv")
    