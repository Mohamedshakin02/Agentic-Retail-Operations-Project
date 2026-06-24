# Agent Card: Retail Supervisor Agent

## 1. Overview

The Retail Supervisor Agent is an **Agentic AI system** designed to assist retail operations teams in:

- analyzing sales and inventory
- forecasting demand
- detecting stock-out risk
- recommending business actions
- requesting human approval before execution

The agent orchestrates multiple analytical tools and applies business rules to generate explainable decisions.

---

## 2. Agent Type

**Category:** Agentic AI (Tool-Calling Agent)  
**Framework:** LangGraph  
**LLM Role:** Reasoning and explanation (NOT computation)

---

## 3. Core Capabilities

The agent can:

- Retrieve inventory and sales data
- Forecast demand (7-day horizon)
- Calculate inventory coverage
- Detect stock-out risk
- Recommend business actions
- Generate business summaries
- Request human approval
- Log reasoning steps (trace)

---

## 4. Agent Workflow

The agent follows a deterministic multi-step pipeline:

### Step 1: Get Inventory Data
Calls:get_inventory_summary()

Returns:
- current_inventory
- average_daily_sales

---

### Step 2: Forecast Demand
Calls:

forecast_demand()

Returns:
- forecast_7_day_demand

---

### Step 3: Calculate Inventory Cover
Calls:

calculate_inventory_cover()

Formula:

inventory_cover_days = current_inventory / average_daily_sales

---

### Step 4: Detect Risk
Calls:

detect_stockout_risk()

Outputs:
- risk_bucket (Critical / Warning / Normal / Overstock)
- stock_out_risk (True/False)

---

### Step 5: Recommend Action
Calls:

recommend_business_action()

Outputs:
- recommended_action
- approval_required

---

### Step 6: Generate Summary
Calls:

generate_business_summary()

Produces natural language explanation.

---

### Step 7: Request Human Approval
Calls:

request_human_approval()

Marks action as:

approval_status = "pending"

---

## 5. Agent Architecture

### Framework

Built using LangGraph:

- Nodes represent tool calls
- Edges define execution order
- State carries intermediate results

### State Object

```python
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