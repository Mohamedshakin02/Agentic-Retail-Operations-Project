"""
forecasting.py
-----------------
Trains a model to predict demand per store/product, and produces
data/outputs/forecast_output.csv in the exact shape inventory_risk.py
expects:  store_id, product_id, forecast_7_day_demand

PER YOUR TEAMMATE'S NOTE: try predicting raw units_sold first. Only use
units_sold_log if it's clearly more accurate — compare_raw_vs_log()
below does this comparison automatically and picks the winner for you.

⚠️ Whichever target wins internally, predictions are ALWAYS converted
back to raw units before saving the CSV. Uses log1p/expm1 (not plain
log/exp) because log1p safely handles days with zero sales — log(0) is
undefined, but log1p(0) = 0.

⚠️ KNOWN SIMPLIFICATION (worth a line in model_card.md): this uses a
random train/test split, not a chronological one. For a proper
production forecaster you'd train on earlier dates and test on later
ones, to avoid the model "seeing the future." For a 4-day prototype,
a random split is an acceptable, documented shortcut.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLUMNS = [
    "day_of_week", "week_of_year", "month", "is_weekend",
    "price", "discount", "inventory_level",
    "lag_1_day_sales", "lag_7_day_sales",
    "rolling_7_day_avg_sales", "rolling_14_day_avg_sales", "rolling_28_day_avg_sales",
    "rolling_7_day_inventory", "price_change_pct", "inventory_cover_days",
    "category_encoded", "region_encoded", "weather_condition_encoded", "seasonality_encoded",
    "holiday_or_promo_flag",
]


def _prepare_xy(df: pd.DataFrame, target_col: str):
    """
    Only use feature columns that actually exist (prints a warning for
    any that don't), and drop rows missing any of them — this mainly
    removes the first few days of each product's history, where
    lag_7_day_sales etc. don't exist yet.
    """
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"⚠️ Skipping features not found in the data: {missing}")

    clean_df = df.dropna(subset=available_features + [target_col])
    X = clean_df[available_features]
    y = clean_df[target_col]
    return X, y, available_features


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wmape = np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / max(np.sum(np.abs(y_true)), 1e-8)
    return {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3), "WMAPE": round(float(wmape), 4)}


def compare_raw_vs_log(df: pd.DataFrame) -> dict:
    """
    Trains two Random Forests — one on units_sold directly, one on
    units_sold_log (converted back to raw units with expm1 BEFORE
    scoring, so the comparison is fair) — and returns whichever one
    actually performs better in real units. Also prints a naive
    "rolling 7-day average" baseline for reference, so you know whether
    the model is even worth its complexity.
    """
    results = {}

    if "units_sold" in df.columns:
        X, y_raw, feature_names = _prepare_xy(df, "units_sold")
        X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)
        model_raw = train_random_forest(X_train, y_train)
        preds_raw = model_raw.predict(X_test)
        results["raw"] = {
            "model": model_raw,
            "feature_names": feature_names,
            "target_type": "raw",
            "metrics": evaluate(y_test, preds_raw),
        }
        if "rolling_7_day_avg_sales" in X_test.columns:
            baseline_metrics = evaluate(y_test, X_test["rolling_7_day_avg_sales"])
            print(f"Naive baseline (rolling 7-day avg) WMAPE for reference: {baseline_metrics['WMAPE']}")

    if "units_sold_log" in df.columns:
        X, y_log, feature_names = _prepare_xy(df, "units_sold_log")
        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
        model_log = train_random_forest(X_train, y_train)
        preds_log_raw_scale = np.expm1(model_log.predict(X_test))
        y_test_raw_scale = np.expm1(y_test)
        results["log"] = {
            "model": model_log,
            "feature_names": feature_names,
            "target_type": "log",
            "metrics": evaluate(y_test_raw_scale, preds_log_raw_scale),
        }

    if not results:
        raise ValueError("Neither 'units_sold' nor 'units_sold_log' found in the data.")

    print("\n--- Raw vs Log comparison (lower WMAPE is better) ---")
    for key, r in results.items():
        print(f"{key:>5}: {r['metrics']}")

    best_key = min(results, key=lambda k: results[k]["metrics"]["WMAPE"])
    print(f"\nUsing '{best_key}' target (lower WMAPE).")

    return results[best_key]


def generate_forecast_output(df: pd.DataFrame, best_result: dict) -> pd.DataFrame:
    """
    Build the final forecast_output.csv: one row per store_id/product_id,
    using each product's most recent row of features, with predictions
    always converted back to raw units.
    """
    model = best_result["model"]
    feature_names = best_result["feature_names"]
    target_type = best_result["target_type"]

    latest_rows = (
        df.sort_values("date")
        .groupby(["store_id", "product_id"])
        .tail(1)
        .dropna(subset=feature_names)
    )

    X_latest = latest_rows[feature_names]
    raw_predictions = model.predict(X_latest)

    if target_type == "log":
        raw_predictions = np.expm1(raw_predictions)

    output = latest_rows[["store_id", "product_id"]].copy()
    output["forecast_7_day_demand"] = np.round(raw_predictions, 1)
    return output.reset_index(drop=True)


def _build_dummy_dataset() -> pd.DataFrame:
    """Tiny fake feature table, shaped like feature_engineering.py's real output, for testing today."""
    n = 40
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n)
    units_sold = rng.integers(5, 30, size=n)
    return pd.DataFrame({
        "date": dates,
        "store_id": ["S003"] * n,
        "product_id": ["P001"] * n,
        "units_sold": units_sold,
        "units_sold_log": np.log1p(units_sold),
        "inventory_level": rng.integers(20, 150, size=n),
        "price": 9.99,
        "discount": 0.0,
        "day_of_week": dates.dayofweek,
        "week_of_year": dates.isocalendar().week.astype(int),
        "month": dates.month,
        "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
        "lag_1_day_sales": pd.Series(units_sold).shift(1).fillna(0),
        "lag_7_day_sales": pd.Series(units_sold).shift(7).fillna(0),
        "rolling_7_day_avg_sales": pd.Series(units_sold).rolling(7, min_periods=1).mean(),
        "rolling_14_day_avg_sales": pd.Series(units_sold).rolling(14, min_periods=1).mean(),
        "rolling_28_day_avg_sales": pd.Series(units_sold).rolling(28, min_periods=1).mean(),
        "rolling_7_day_inventory": rng.integers(20, 150, size=n),
        "price_change_pct": 0.0,
        "inventory_cover_days": rng.uniform(1, 20, size=n),
        "category_encoded": 0,
        "region_encoded": 0,
        "weather_condition_encoded": 0,
        "seasonality_encoded": 0,
        "holiday_or_promo_flag": 0,
    })


if __name__ == "__main__":
    features_path = "data/processed/retail_features.csv"

    if os.path.exists(features_path):
        print(f"Found real feature data at {features_path} — using it.")
        df = pd.read_csv(features_path, parse_dates=["date"])
    else:
        print("Real feature data not ready yet — using dummy data to test the pipeline.")
        df = _build_dummy_dataset()

    best_result = compare_raw_vs_log(df)
    forecast_df = generate_forecast_output(df, best_result)

    print("\nForecast output sample:")
    print(forecast_df.head())

    os.makedirs("data/outputs", exist_ok=True)
    forecast_df.to_csv("data/outputs/forecast_output.csv", index=False)
    print("\nSaved to data/outputs/forecast_output.csv")