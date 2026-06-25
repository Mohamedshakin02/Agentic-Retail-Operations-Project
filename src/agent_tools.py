"""
agent_tools.py
---------------
Every "tool" the Retail Supervisor Agent can call.

STATUS: real implementations now that real data exists throughout the
pipeline. The old TODO stubs are gone — each function below reads from
the actual files your pipeline produces, with a safe zero-value
fallback if a file isn't there yet (so this never crashes, it just
returns "nothing to report" numbers).

Three functions are intentionally NOT defined here — they're imported
from the files that actually own that logic, so there's only ever one
source of truth for each (the duplicate-logic fix from earlier, now
actually applied instead of just described):
    calculate_inventory_cover, detect_stockout_risk  -> inventory_risk.py
    recommend_business_action, recommend_reorder_quantity -> recommendation.py
    request_human_approval -> approval.py
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Mapping, Any, Optional

from inventory_risk import calculate_inventory_cover, detect_stockout_risk
from recommendation import recommend_business_action, recommend_reorder_quantity
from approval import request_human_approval
from config import CLEANED_DATA_PATH, FEATURES_DATA_PATH, FORECAST_OUTPUT_PATH


def get_sales_summary(store_id: Optional[str] = None, product_id: Optional[str] = None) -> dict:
    """
    Return total units sold, revenue, and average selling price.
    Optionally filter by store_id and/or product_id.
    """
    if not os.path.exists(CLEANED_DATA_PATH):
        return {"store_id": store_id or "ALL", "product_id": product_id or "ALL",
                "total_units_sold": 0, "total_revenue": 0, "avg_selling_price": 0}

    df = pd.read_csv(CLEANED_DATA_PATH)
    if store_id:
        df = df[df["store_id"] == store_id]
    if product_id:
        df = df[df["product_id"] == product_id]

    if df.empty:
        return {"store_id": store_id or "ALL", "product_id": product_id or "ALL",
                "total_units_sold": 0, "total_revenue": 0, "avg_selling_price": 0}

    total_units = df["units_sold"].sum()
    revenue = (df["units_sold"] * df["price"]).sum()
    avg_price = round(revenue / total_units, 2) if total_units > 0 else 0

    return {
        "store_id": store_id or "ALL",
        "product_id": product_id or "ALL",
        "total_units_sold": int(total_units),
        "total_revenue": round(float(revenue), 2),
        "avg_selling_price": avg_price,
    }


def get_inventory_summary(store_id: Optional [str] = None, product_id:Optional[str] = None) -> dict:
    """
    Return current inventory level and average daily sales for a
    store/product combination, using the most recent available row.
    Prefers retail_features.csv (has rolling_7_day_avg_sales already
    computed); falls back to retail_cleaned.csv otherwise.
    """
    path = FEATURES_DATA_PATH if os.path.exists(FEATURES_DATA_PATH) else CLEANED_DATA_PATH
    if not os.path.exists(path):
        return {"store_id": store_id, "product_id": product_id, "current_inventory": 0, "average_daily_sales": 0}

    df = pd.read_csv(path, parse_dates=["date"])
    subset = df[(df["store_id"] == store_id) & (df["product_id"] == product_id)].sort_values("date")

    if subset.empty:
        return {"store_id": store_id, "product_id": product_id, "current_inventory": 0, "average_daily_sales": 0}

    latest = subset.iloc[-1]
    current_inventory = float(latest["inventory_level"])

    if "rolling_7_day_avg_sales" in subset.columns:
        average_daily_sales = float(latest["rolling_7_day_avg_sales"])
    else:
        average_daily_sales = float(subset["units_sold"].tail(7).mean())

    return {
        "store_id": store_id,
        "product_id": product_id,
        "current_inventory": current_inventory,
        "average_daily_sales": round(average_daily_sales, 2),
    }


def forecast_demand(store_id: str, product_id: str) -> dict:
    """
    Predict the next 7-day demand for a given store/product combination,
    by reading forecast_output.csv (produced by forecasting.py).
    """
    if not os.path.exists(FORECAST_OUTPUT_PATH):
        return {"store_id": store_id, "product_id": product_id, "forecast_7_day_demand": 0, "model_used": "not_available"}

    df = pd.read_csv(FORECAST_OUTPUT_PATH)
    row = df[(df["store_id"] == store_id) & (df["product_id"] == product_id)]

    if row.empty:
        return {"store_id": store_id, "product_id": product_id, "forecast_7_day_demand": 0, "model_used": "not_available"}

    return {
        "store_id": store_id,
        "product_id": product_id,
        "forecast_7_day_demand": float(row.iloc[0]["forecast_7_day_demand"]),
        "model_used": "random_forest",
    }


def analyze_promotion_impact(product_id: str) -> dict:
    """
    Compare average sales during promotion/holiday periods vs normal
    periods for a product. Uses the combined holiday_or_promo_flag
    column — this dataset doesn't separate "promotion" from "holiday"
    (documented as a known limitation in model_card.md).
    """
    if not os.path.exists(CLEANED_DATA_PATH):
        return {"product_id": product_id, "promo_avg_sales": 0, "non_promo_avg_sales": 0, "promotion_lift_pct": 0}

    df = pd.read_csv(CLEANED_DATA_PATH)
    subset = df[df["product_id"] == product_id]

    if subset.empty or "holiday_or_promo_flag" not in subset.columns:
        return {"product_id": product_id, "promo_avg_sales": 0, "non_promo_avg_sales": 0, "promotion_lift_pct": 0}

    promo_avg = subset[subset["holiday_or_promo_flag"] == 1]["units_sold"].mean()
    non_promo_avg = subset[subset["holiday_or_promo_flag"] == 0]["units_sold"].mean()

    promo_avg = 0 if pd.isna(promo_avg) else round(float(promo_avg), 2)
    non_promo_avg = 0 if pd.isna(non_promo_avg) else round(float(non_promo_avg), 2)
    lift = round(((promo_avg / non_promo_avg) - 1) * 100, 1) if non_promo_avg > 0 else 0

    return {
        "product_id": product_id,
        "promo_avg_sales": promo_avg,
        "non_promo_avg_sales": non_promo_avg,
        "promotion_lift_pct": lift,
    }


def generate_business_summary(results: dict) -> str:
    """
    Turn a dict of combined results (forecast + risk + recommendation)
    into a short plain-English summary for the user.
    """
    return (
        f"Product {results.get('product_id')} at Store {results.get('store_id')} "
        f"is forecasted to need {results.get('forecast_7_day_demand')} units over "
        f"the next 7 days. Current inventory covers "
        f"{results.get('inventory_cover_days')} days "
        f"({results.get('risk_bucket')} risk). "
        f"Recommended action: {results.get('recommended_action')}."
    )


_trace_log = []  # in-memory for now; later this can also write to a file


def log_agent_trace(step_name: str, tool_input: Mapping[str, Any], tool_output: Mapping[str, Any]) -> None:
    """
    Record one step of the agent's reasoning, for the Agent Trace UI page.
    Uses Mapping instead of dict so this happily accepts AgentState (a
    TypedDict) as well as plain dicts — see the earlier note on why
    dict and TypedDict don't satisfy each other for type checkers.
    """
    _trace_log.append({
        "step": step_name,
        "input": dict(tool_input),
        "output": dict(tool_output),
        "timestamp": datetime.now().isoformat(),
    })


def get_trace_log() -> list:
    """Return every step logged so far (used by the Agent Trace UI page)."""
    return _trace_log


if __name__ == "__main__":
    # Quick manual test — run with: python src/agent_tools.py
    inv = get_inventory_summary("S003", "P001")
    fc = forecast_demand("S003", "P001")
    cover = calculate_inventory_cover(inv["current_inventory"], inv["average_daily_sales"])
    risk = detect_stockout_risk(
        cover["inventory_cover_days"], fc["forecast_7_day_demand"], inv["current_inventory"]
    )
    action = recommend_business_action(risk["risk_bucket"], risk["stock_out_risk"])

    combined = {**inv, **fc, **cover, **risk, **action}
    print(json.dumps(combined, indent=2))
    print("\nSales summary (S003/P001):", get_sales_summary("S003", "P001"))
    print("Promotion impact (P001):", analyze_promotion_impact("P001"))