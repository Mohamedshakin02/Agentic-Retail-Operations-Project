"""
data_quality.py
------------------
Checks whether the cleaned dataset is actually trustworthy before
anything gets built on top of it: missing values, duplicates,
impossible negative numbers, price/sales outliers, and gaps in the
daily date sequence per store/product.

Run with: python src/data_quality.py
Produces a printed report and saves data/outputs/data_quality_report.json
"""

import os
import json
import pandas as pd


NUMERIC_COLUMNS_THAT_SHOULDNT_BE_NEGATIVE = [
    "units_sold", "units_ordered", "inventory_level", "price",
    "discount", "competitor_price",
]


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, parse_dates=["date"])


def check_basic_shape(df: pd.DataFrame) -> dict:
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "unique_stores": int(df["store_id"].nunique()),
        "unique_products": int(df["product_id"].nunique()),
    }


def check_missing_values(df: pd.DataFrame) -> dict:
    """{column: number_of_missing_rows} for every column with at least one missing value."""
    missing = df.isna().sum()
    return {col: int(count) for col, count in missing.items() if count > 0}


def check_duplicates(df: pd.DataFrame) -> dict:
    exact_duplicates = int(df.duplicated().sum())
    # Should never have two rows for the same store + product + date
    key_duplicates = int(df.duplicated(subset=["store_id", "product_id", "date"]).sum())
    return {"exact_duplicate_rows": exact_duplicates, "duplicate_store_product_date_rows": key_duplicates}


def check_negative_values(df: pd.DataFrame) -> dict:
    """Counts of impossible negative values per column — these point to data entry errors."""
    results = {}
    for col in NUMERIC_COLUMNS_THAT_SHOULDNT_BE_NEGATIVE:
        if col in df.columns:
            negative_count = int((df[col] < 0).sum())
            if negative_count > 0:
                results[col] = negative_count
    return results


def check_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> dict:
    """
    Flags values far outside the normal range using the IQR method.
    This is informational, not an error — a real holiday sales spike
    SHOULD look like an outlier. It just tells you where to look.
    """
    if column not in df.columns:
        return {"column": column, "outlier_count": 0, "note": "column not found"}

    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    outlier_count = int(((df[column] < lower_bound) | (df[column] > upper_bound)).sum())

    return {
        "column": column,
        "normal_range": f"{round(lower_bound, 2)} to {round(upper_bound, 2)}",
        "outlier_count": outlier_count,
    }


def check_date_continuity(df: pd.DataFrame) -> dict:
    """
    For each store+product, checks whether every date in its min-to-max
    range actually has a row. A gap could mean a data collection issue,
    OR just that the product wasn't stocked yet — worth a human glance,
    not automatically a bug.
    """
    gap_summary = []
    grouped = df.groupby(["store_id", "product_id"])

    for (store_id, product_id), group in grouped:
        expected_dates = pd.date_range(group["date"].min(), group["date"].max())
        actual_dates = set(group["date"])
        missing_days = len(expected_dates) - len(actual_dates)
        if missing_days > 0:
            gap_summary.append({"store_id": store_id, "product_id": product_id, "missing_days": missing_days})

    return {
        "store_product_combos_with_gaps": len(gap_summary),
        "total_combos_checked": len(grouped),
        "details_sample": gap_summary[:5],  # just the first 5, so the report doesn't flood
    }


def generate_quality_report(df: pd.DataFrame) -> dict:
    """Runs every check above and returns one combined report."""
    return {
        "shape": check_basic_shape(df),
        "missing_values": check_missing_values(df),
        "duplicates": check_duplicates(df),
        "negative_values": check_negative_values(df),
        "outliers_units_sold": check_outliers_iqr(df, "units_sold"),
        "outliers_price": check_outliers_iqr(df, "price"),
        "date_continuity": check_date_continuity(df),
    }


def print_report(report: dict) -> None:
    print("\n=== DATA QUALITY REPORT ===\n")
    for section, content in report.items():
        print(f"--- {section} ---")
        print(json.dumps(content, indent=2, default=str))
        print()


def save_report(report: dict, filepath: str = "data/outputs/data_quality_report.json") -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved full report to {filepath}")


def _build_dummy_dataset() -> pd.DataFrame:
    """A small fake dataset with a few INTENTIONAL problems, so you can see the checks actually catch them."""
    dates = pd.date_range("2024-01-01", periods=10)
    return pd.DataFrame({
        "date": list(dates) + [dates[3]],          # intentional duplicate date
        "store_id": ["S003"] * 10 + ["S003"],
        "product_id": ["P001"] * 10 + ["P001"],
        "units_sold": [12, 15, -9, 20, 18, 14, 11, 25, 22, 19, 19],  # -9 is an intentional bad value
        "inventory_level": [100, 88, 79, 59, 41, 27, 16, 10, 31, 50, 50],
        "price": [9.99] * 11,
        "units_ordered": [0] * 11,
        "competitor_price": [10.5] * 11,
        "discount": [0.0] * 11,
    })


if __name__ == "__main__":
    cleaned_data_path = "data/processed/retail_cleaned.csv"

    if os.path.exists(cleaned_data_path):
        print(f"Found real data at {cleaned_data_path} — checking it.")
        df = load_data(cleaned_data_path)
    else:
        print("Real data not found yet — using a dummy dataset (with intentional problems) to test the checks.")
        df = _build_dummy_dataset()

    report = generate_quality_report(df)
    print_report(report)
    save_report(report)