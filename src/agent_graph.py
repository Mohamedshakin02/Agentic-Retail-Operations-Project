"""
agent_graph.py
---------------
Wires agent_tools.py into an actual LangGraph agent: the Retail Supervisor Agent.

STATUS: handles ONE store/product question end-to-end, e.g.
"Why is Product P001 at Store S003 at risk?" — this matches the doc's
Module 9 example flow exactly.

NOT YET HANDLED: "Find the top 5 products at risk" needs scanning every
store/product combination. That's a small loop we add ON TOP of this
once this core flow is proven to work — see the note at the bottom.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

import agent_tools as tools


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


# --- Build the graph: register every station, then connect the belts ---
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


if __name__ == "__main__":
    # Run with: python src/agent_graph.py
    result = retail_agent.invoke({"store_id": "S003", "product_id": "P001"})

    print("Summary:", result["summary"])
    print("Approval:", result["approval"])

    print("\n--- Agent Trace ---")
    for step in tools.get_trace_log():
        print("-", step["step"])
