"""
agent_tools.py
---------------
Every "tool" the Retail Supervisor Agent can call.

STATUS: STUB VERSION.
These return realistic fake data so we can build and test agent_graph.py
right now, without waiting for the real CSVs from teammates A and B.

Once these exist, swap the fake data for real reads:
    data/processed/retail_cleaned.csv      (from A's data_ingestion.py)
    data/outputs/forecast_output.csv       (from B's forecasting.py)
    data/outputs/risk_output.csv           (from B's inventory_risk.py)
    data/outputs/action_recommendation.csv (from B's recommendation.py)

Keep function names and return shapes IDENTICAL when you do that swap —
agent_graph.py should not need to change.

Docstrings matter here: the agent reads them to decide which tool to call,
so keep them accurate and specific.
"""

from datetime import datetime
import json


def get_sales_summary(store_id: str = None, product_id: str = None) -> dict:
    """
    Return total units sold, revenue, and average selling price.
    Optionally filter by store_id and/or product_id.
    """
    # TODO(C): replace with a pandas groupby on retail_cleaned.csv
    return {
        "store_id": store_id or "ALL",
        "product_id": product_id or "ALL",
        "total_units_sold": 128430,
        "total_revenue": 1840000,
        "avg_selling_price": 14.32,
    }


def get_inventory_summary(store_id: str = None, product_id: str = None) -> dict:
    """
    Return current inventory level and average daily sales for a
    store/product combination.
    """
    # TODO(C): replace with a lookup on retail_cleaned.csv
    return {
        "store_id": store_id or "S003",
        "product_id": product_id or "P001",
        "current_inventory": 35,
        "average_daily_sales": 17.2,
    }


def forecast_demand(store_id: str, product_id: str) -> dict:
    """
    Predict the next 7-day demand for a given store/product combination.
    """
    # TODO(C): replace with a row lookup on forecast_output.csv
    return {
        "store_id": store_id,
        "product_id": product_id,
        "forecast_7_day_demand": 120,
        "model_used": "baseline_moving_average",  # later: "random_forest"
    }


def calculate_inventory_cover(current_inventory: float, average_daily_sales: float) -> dict:
    """
    Calculate how many days of inventory remain at the current sales pace.
    inventory_cover_days = current_inventory / average_daily_sales
    """
    if average_daily_sales <= 0:
        cover_days = float("inf")
    else:
        cover_days = round(current_inventory / average_daily_sales, 2)
    return {"inventory_cover_days": cover_days}


def detect_stockout_risk(inventory_cover_days: float, forecast_7_day_demand: float, current_inventory: float) -> dict:
    """
    Classify risk into Critical / Warning / Normal / Overstock,
    and flag whether forecasted demand exceeds current inventory.
    """
    if inventory_cover_days < 2:
        bucket = "Critical"
    elif inventory_cover_days < 5:
        bucket = "Warning"
    elif inventory_cover_days <= 21:
        bucket = "Normal"
    else:
        bucket = "Overstock"

    stock_out_risk = forecast_7_day_demand > current_inventory

    return {"risk_bucket": bucket, "stock_out_risk": stock_out_risk}


def analyze_promotion_impact(product_id: str) -> dict:
    """
    Compare average sales during promotion periods vs non-promotion periods.
    """
    # TODO(C): replace with a groupby on promotion_flag in retail_cleaned.csv
    return {
        "product_id": product_id,
        "promo_avg_sales": 92,
        "non_promo_avg_sales": 61,
        "promotion_lift_pct": 50.8,
    }


def recommend_reorder_quantity(forecast_7_day_demand: float, current_inventory: float) -> int:
    """
    Suggest how many units to reorder: forecasted demand minus current stock,
    floored at zero.
    """
    return max(0, round(forecast_7_day_demand - current_inventory))


def recommend_business_action(risk_bucket: str, stock_out_risk: bool) -> dict:
    """
    Map a risk bucket to a recommended business action and whether
    human approval is required before acting.
    """
    actions = {
        "Critical": "Reorder urgently",
        "Warning": "Replenish / monitor",
        "Normal": "No action",
        "Overstock": "Review stock / consider markdown",
    }
    return {
        "recommended_action": actions.get(risk_bucket, "No action"),
        "approval_required": risk_bucket in ("Critical", "Warning"),
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


def request_human_approval(action_summary: str) -> dict:
    """
    Present the recommended action and mark it pending human approval.
    The actual Approve/Reject click is handled later by the Streamlit UI
    and approval.py — this just creates the pending record.
    """
    return {
        "action_summary": action_summary,
        "approval_status": "pending",
        "requested_at": datetime.now().isoformat(),
    }


_trace_log = []  # in-memory for now; later this can also write to a file


def log_agent_trace(step_name: str, tool_input: dict, tool_output: dict) -> None:
    """
    Record one step of the agent's reasoning, for the Agent Trace UI page.
    """
    _trace_log.append({
        "step": step_name,
        "input": tool_input,
        "output": tool_output,
        "timestamp": datetime.now().isoformat(),
    })


def get_trace_log() -> list:
    """Return every step logged so far (used by the Agent Trace UI page)."""
    return _trace_log


if __name__ == "__main__":
    # Quick manual test, no agent or LangGraph needed yet.
    # Run with: python src/agent_tools.py
    inv = get_inventory_summary("S003", "P001")
    fc = forecast_demand("S003", "P001")
    cover = calculate_inventory_cover(inv["current_inventory"], inv["average_daily_sales"])
    risk = detect_stockout_risk(
        cover["inventory_cover_days"], fc["forecast_7_day_demand"], inv["current_inventory"]
    )
    action = recommend_business_action(risk["risk_bucket"], risk["stock_out_risk"])

    combined = {**inv, **fc, **cover, **risk, **action}
    print(json.dumps(combined, indent=2))
    print("\nSummary:", generate_business_summary(combined))