"""
tests/test_data_quality.py
-----------------------------
Tests for the checks in src/data_quality.py. Each check gets two tests:
one proving it stays quiet on clean data, one proving it actually
catches the problem it's supposed to catch.

Run with: pytest tests/test_data_quality.py -v
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_quality import (
    check_missing_values,
    check_duplicates,
    check_negative_values,
    check_outliers_iqr,
    check_date_continuity,
    check_basic_shape,
)


def _make_clean_df():
    """A small, intentionally PERFECT dataset — every check should report zero problems."""
    dates = pd.date_range("2024-01-01", periods=5)
    return pd.DataFrame({
        "date": dates,
        "store_id": ["S001"] * 5,
        "product_id": ["P001"] * 5,
        "units_sold": [10, 12, 9, 11, 13],
        "inventory_level": [100, 90, 81, 70, 57],
        "price": [9.99] * 5,
        "units_ordered": [0] * 5,
        "competitor_price": [10.5] * 5,
        "discount": [0.0] * 5,
    })


# ---------------------------------------------------------------------------
# missing values
# ---------------------------------------------------------------------------

def test_clean_data_has_no_missing_values():
    df = _make_clean_df()
    assert check_missing_values(df) == {}


def test_missing_values_are_detected():
    df = _make_clean_df()
    df.loc[0, "price"] = None
    result = check_missing_values(df)
    assert result.get("price") == 1


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------

def test_clean_data_has_no_duplicates():
    df = _make_clean_df()
    result = check_duplicates(df)
    assert result["exact_duplicate_rows"] == 0
    assert result["duplicate_store_product_date_rows"] == 0


def test_duplicate_store_product_date_is_caught():
    df = _make_clean_df()
    duplicate_row = df.iloc[[0]]  # same store/product/date as row 0
    df_with_dupe = pd.concat([df, duplicate_row], ignore_index=True)
    result = check_duplicates(df_with_dupe)
    assert result["duplicate_store_product_date_rows"] == 1


# ---------------------------------------------------------------------------
# negative values
# ---------------------------------------------------------------------------

def test_clean_data_has_no_negative_values():
    df = _make_clean_df()
    assert check_negative_values(df) == {}


def test_negative_units_sold_is_caught():
    df = _make_clean_df()
    df.loc[0, "units_sold"] = -5
    result = check_negative_values(df)
    assert result.get("units_sold") == 1


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------

def test_outlier_detection_flags_an_extreme_value():
    df = _make_clean_df()
    df.loc[0, "units_sold"] = 5000  # absurdly high compared to the rest
    result = check_outliers_iqr(df, "units_sold")
    assert result["outlier_count"] >= 1


def test_outlier_detection_on_missing_column_does_not_crash():
    df = _make_clean_df()
    result = check_outliers_iqr(df, "column_that_does_not_exist")
    assert result["outlier_count"] == 0


# ---------------------------------------------------------------------------
# date continuity
# ---------------------------------------------------------------------------

def test_date_continuity_no_gaps_in_clean_data():
    df = _make_clean_df()
    result = check_date_continuity(df)
    assert result["store_product_combos_with_gaps"] == 0


def test_date_continuity_detects_a_gap():
    df = _make_clean_df()
    df_with_gap = df.drop(index=2).reset_index(drop=True)  # remove day 3, leaving a gap
    result = check_date_continuity(df_with_gap)
    assert result["store_product_combos_with_gaps"] == 1
    assert result["details_sample"][0]["missing_days"] == 1


# ---------------------------------------------------------------------------
# basic shape
# ---------------------------------------------------------------------------

def test_basic_shape_reports_correct_counts():
    df = _make_clean_df()
    result = check_basic_shape(df)
    assert result["total_rows"] == 5
    assert result["unique_stores"] == 1
    assert result["unique_products"] == 1   