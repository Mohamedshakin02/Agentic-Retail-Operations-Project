# Business Rules: Retail Operations Copilot

## Overview

This document defines the business rules used to:

- Detect inventory risk
- Classify stock conditions
- Recommend actions
- Determine human approval requirements

These rules are implemented in:

- `inventory_risk.py`
- `recommendation.py`

---

## 1. Inventory Cover Rule

### Logic

inventory_cover_days = inventory_level / average_daily_sales

### Special Handling

- If average_daily_sales <= 0 → inventory_cover_days = infinity

### Meaning

- Low cover → risk of stock-out
- High cover → potential overstock

---

## 2. Category-Based Risk Thresholds

Risk thresholds vary by product category:

| Category | Critical | Warning | Normal |
|--------|---------|--------|--------|
| Groceries | <2 days | <5 days | <=14 days |
| Clothing | <5 | <12 | <=30 |
| Electronics | <5 | <12 | <=30 |
| Furniture | <10 | <25 | <=60 |

Default (fallback):
- Critical <2
- Warning <5
- Normal <=21

---

## 3. Risk Bucket Classification

### Logic

| Condition | Risk Bucket |
|----------|------------|
| cover < critical | Critical |
| cover < warning | Warning |
| cover <= normal_max | Normal |
| cover > normal_max | Overstock |

---

## 4. Stock-Out Risk Rule

### Logic

stock_out_risk = forecast_7_day_demand > inventory_level

### Meaning

- True → demand exceeds available stock
- False → inventory is sufficient

---

## 5. Action Recommendation Rules

### Base Mapping

| Risk | Action |
|-----|--------|
| Critical | Reorder urgently |
| Warning | Replenish / monitor |
| Normal | No action |
| Overstock | Review stock / markdown |

---

### Context-Based Overrides

#### Promotion / Holiday

If:
- holiday_or_promo_flag = True
- AND risk is Critical or Warning

Then:
→ Action = "Pre-stock before promotion/holiday"

---

#### Demand Trend Adjustment

If:
- demand_trend = decreasing
- AND risk = Overstock

Then:
→ Action = "Reduce replenishment"

---

## 6. Reorder Quantity Rule

### Logic

recommended_quantity = max(0, forecast_7_day_demand - inventory_level)

### Meaning

- Prevents negative orders
- Aligns stock with demand forecast

---

## 7. Approval Requirement Rule

### Approval Required If:

- Risk = Critical
- OR Risk = Warning
- OR action = "Pre-stock before promotion/holiday"

### Meaning

No automated actions are allowed without approval.

---

## 8. Data Merge Rule (Important)

Forecast and feature tables must match on:

- store_id
- product_id

### Risk

Mismatch causes:
- missing forecasts
- incorrect risk calculations

### Protection (implemented)

- Convert both to strings
- Strip whitespace

---

## 9. Error Handling Rules

### Missing Columns

- Raises error if required inputs not present

### Missing Forecast

- Defaults to 0
- Logs warning

---

## 10. Output Rules

Final output must include:

- store_id
- product_id
- forecast_7_day_demand
- current_inventory
- inventory_cover_days
- risk_bucket
- recommended_action
- recommended_quantity
- approval_required

---

## 11. Agent Constraint

The AI agent:
- MUST NOT calculate KPIs directly
- MUST call tools/functions instead

---

## Summary

These business rules ensure:

- consistent decision-making
- explainable outputs
- safe execution via human approval

They are deterministic, transparent, and aligned with real retail operations workflows.