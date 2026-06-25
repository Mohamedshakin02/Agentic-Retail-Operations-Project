"""
agent_graph.py
---------------
Wires agent_tools.py into an actual LangGraph agent: the Retail Supervisor Agent.

Handles TWO question shapes now:
1. ONE specific store/product, e.g. "Why is P001 at Store S003 at risk?"
   -> retail_agent.invoke({"store_id": ..., "product_id": ...})
2. "Find the top N products at risk" (the project's headline demo question)
   -> find_top_risk_products(n=5)
   Step 1 scans the whole risk table to find the N most urgent products
   (a cheap lookup). Step 2 runs the FULL single-item agent pipeline on
   each of those N — that second part is the genuinely agentic step,
   investigating each flagged product individually.
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import pandas as pd

import agent_tools as tools
from recommendation import get_top_risk_products, _build_dummy_risk_table


class AgentState(TypedDict):
    store_id: str
    product_id: str
    inventory: dict
    forecast: dict
    cover: dict
    risk: dict
    recommendation: dict
    summary: str
    approval: dict


def node_get_inventory(state: AgentState) -> dict:
    result = tools.get_inventory_summary(state["store_id"], state["product_id"])
    tools.log_agent_trace("get_inventory_summary", state, result)
    return {"inventory": result}


def node_forecast_demand(state: AgentState) -> dict:
    result = tools.forecast_demand(state["store_id"], state["product_id"])
    tools.log_agent_trace("forecast_demand", state, result)
    return {"forecast": result}


def node_calculate_cover(state: AgentState) -> dict:
    inv = state["inventory"]
    result = tools.calculate_inventory_cover(inv["current_inventory"], inv["average_daily_sales"])
    tools.log_agent_trace("calculate_inventory_cover", state, result)
    return {"cover": result}


def node_detect_risk(state: AgentState) -> dict:
    result = tools.detect_stockout_risk(
        state["cover"]["inventory_cover_days"],
        state["forecast"]["forecast_7_day_demand"],
        state["inventory"]["current_inventory"],
    )
    tools.log_agent_trace("detect_stockout_risk", state, result)
    return {"risk": result}


def node_recommend_action(state: AgentState) -> dict:
    result = tools.recommend_business_action(
        state["risk"]["risk_bucket"], state["risk"]["stock_out_risk"]
    )
    tools.log_agent_trace("recommend_business_action", state, result)
    return {"recommendation": result}


def node_generate_summary(state: AgentState) -> dict:
    combined = {
        "store_id": state["store_id"],
        "product_id": state["product_id"],
        "forecast_7_day_demand": state["forecast"]["forecast_7_day_demand"],
        "inventory_cover_days": state["cover"]["inventory_cover_days"],
        "risk_bucket": state["risk"]["risk_bucket"],
        "recommended_action": state["recommendation"]["recommended_action"],
    }
    summary = tools.generate_business_summary(combined)
    tools.log_agent_trace("generate_business_summary", state, {"summary": summary})
    return {"summary": summary}


def node_request_approval(state: AgentState) -> dict:
    result = tools.request_human_approval(state["summary"])
    tools.log_agent_trace("request_human_approval", state, result)
    return {"approval": result}


builder = StateGraph(AgentState)

builder.add_node("get_inventory", node_get_inventory)
builder.add_node("forecast_demand", node_forecast_demand)
builder.add_node("calculate_cover", node_calculate_cover)
builder.add_node("detect_risk", node_detect_risk)
builder.add_node("recommend_action", node_recommend_action)
builder.add_node("generate_summary", node_generate_summary)
builder.add_node("request_approval", node_request_approval)

builder.add_edge(START, "get_inventory")
builder.add_edge("get_inventory", "forecast_demand")
builder.add_edge("forecast_demand", "calculate_cover")
builder.add_edge("calculate_cover", "detect_risk")
builder.add_edge("detect_risk", "recommend_action")
builder.add_edge("recommend_action", "generate_summary")
builder.add_edge("generate_summary", "request_approval")
builder.add_edge("request_approval", END)

retail_agent = builder.compile()


def find_top_risk_products(n: int = 5, risk_data_path: str = "data/outputs/risk_output.csv") -> list:
    """
    Answers "find the top N products at stock-out risk" — the project's
    headline demo question.

    Step 1: scan the whole risk table for the N most urgent products
            (reuses recommendation.py's get_top_risk_products — no
            duplicate logic here, same lesson as before).
    Step 2: run the FULL single-item agent pipeline on each of those N,
            so every result has a real investigated summary and a
            pending approval, not just a filtered table row.

    Falls back to recommendation.py's dummy risk table if the real
    risk_output.csv isn't ready yet.
    """
    if os.path.exists(risk_data_path):
        risk_df = pd.read_csv(risk_data_path)
    else:
        print(f"{risk_data_path} not found yet — using dummy risk data to test this.")
        risk_df = _build_dummy_risk_table()

    top_candidates = get_top_risk_products(risk_df, n=n)
    tools.log_agent_trace(
        "find_top_risk_products_scan",
        {"n": n},
        {"found": len(top_candidates), "products": top_candidates["product_id"].tolist()},
    )

    investigated_results = []
    for _, row in top_candidates.iterrows():
        result = retail_agent.invoke({"store_id": row["store_id"], "product_id": row["product_id"]})
        investigated_results.append({
            "store_id": row["store_id"],
            "product_id": row["product_id"],
            "risk_bucket": row["risk_bucket"],
            "summary": result["summary"],
            "approval": result["approval"],
        })

    return investigated_results


if __name__ == "__main__":
    # Run with: python src/agent_graph.py

    print("=== Single product question ===")
    result = retail_agent.invoke({"store_id": "S003", "product_id": "P001"})
    print("Summary:", result["summary"])
    print("Approval:", result["approval"])

    print("\n=== Top 5 at risk question ===")
    top5 = find_top_risk_products(n=5)
    for item in top5:
        print(f"\n{item['product_id']} @ {item['store_id']} ({item['risk_bucket']}):")
        print(" ", item["summary"])

    print("\n--- Full Agent Trace ---")
    for step in tools.get_trace_log():
        print("-", step["step"])