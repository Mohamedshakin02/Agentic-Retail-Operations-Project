"""
tests/test_agent_scenarios.py
-------------------------------
Tests for the core agent logic: risk detection, recommendations, the
full agent graph, and the approval flow.

Run with: pytest tests/test_agent_scenarios.py -v
(the -v flag shows each test's name and pass/fail individually)

If a test fails with an import error instead of an assertion error,
that means one of your modules (inventory_risk, recommendation,
approval, agent_graph) couldn't be found or has a bug in it — fix
that first, the actual test logic isn't the problem in that case.
"""

import os
import sys

# Make src/ importable regardless of where pytest is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from inventory_risk import calculate_inventory_cover, detect_stockout_risk
from recommendation import recommend_business_action, recommend_reorder_quantity
from approval import submit_approval_decision, save_approved_action, get_approval_history
from agent_graph import retail_agent


# ---------------------------------------------------------------------------
# inventory_risk.py
# ---------------------------------------------------------------------------

def test_calculate_inventory_cover_normal_case():
    result = calculate_inventory_cover(current_inventory=35, average_daily_sales=17.2)
    assert result["inventory_cover_days"] == pytest.approx(2.03, abs=0.01)


def test_calculate_inventory_cover_zero_sales_is_infinite_not_a_crash():
    """A product with zero recent sales should be 'infinite cover', not a ZeroDivisionError."""
    result = calculate_inventory_cover(current_inventory=50, average_daily_sales=0)
    assert result["inventory_cover_days"] == float("inf")


def test_detect_stockout_risk_critical_with_default_thresholds():
    result = detect_stockout_risk(inventory_cover_days=1.5, forecast_7_day_demand=100, current_inventory=20)
    assert result["risk_bucket"] == "Critical"
    assert result["stock_out_risk"] is True


def test_category_thresholds_change_the_outcome():
    """
    The same 8-day cover should NOT mean the same risk for every category —
    that was the entire point of making thresholds category-aware.
    """
    groceries = detect_stockout_risk(
        inventory_cover_days=8, forecast_7_day_demand=10, current_inventory=50, category="Groceries"
    )
    furniture = detect_stockout_risk(
        inventory_cover_days=8, forecast_7_day_demand=10, current_inventory=50, category="Furniture"
    )

    assert groceries["risk_bucket"] == "Normal"     # 8 days is plenty for fast-moving groceries
    assert furniture["risk_bucket"] == "Critical"   # 8 days is dangerously low for slow-restock furniture


def test_unknown_category_falls_back_to_default_thresholds():
    with_unknown_category = detect_stockout_risk(
        inventory_cover_days=3, forecast_7_day_demand=10, current_inventory=50, category="Pet Supplies"
    )
    with_no_category = detect_stockout_risk(
        inventory_cover_days=3, forecast_7_day_demand=10, current_inventory=50
    )
    assert with_unknown_category["risk_bucket"] == with_no_category["risk_bucket"]


# ---------------------------------------------------------------------------
# recommendation.py
# ---------------------------------------------------------------------------

def test_recommend_business_action_critical_requires_approval():
    result = recommend_business_action(risk_bucket="Critical", stock_out_risk=True)
    assert result["recommended_action"] == "Reorder urgently"
    assert result["approval_required"] is True


def test_recommend_business_action_normal_needs_no_approval():
    result = recommend_business_action(risk_bucket="Normal", stock_out_risk=False)
    assert result["recommended_action"] == "No action"
    assert result["approval_required"] is False


def test_recommend_reorder_quantity_never_negative():
    """If current inventory already exceeds forecasted demand, don't recommend a negative reorder."""
    quantity = recommend_reorder_quantity(forecast_7_day_demand=50, current_inventory=80)
    assert quantity == 0


def test_recommend_reorder_quantity_normal_case():
    quantity = recommend_reorder_quantity(forecast_7_day_demand=120, current_inventory=35)
    assert quantity == 85


# ---------------------------------------------------------------------------
# agent_graph.py — the full pipeline, end to end
# ---------------------------------------------------------------------------

def test_full_agent_graph_runs_without_crashing():
    result = retail_agent.invoke({"store_id": "S003", "product_id": "P001"})
    # Not checking exact numbers here — those are stub values that will
    # change once real data lands. Just confirming every step ran and
    # produced the shape of output the UI will expect.
    assert "summary" in result
    assert "approval" in result
    assert result["approval"]["approval_status"] == "pending"


# ---------------------------------------------------------------------------
# approval.py — the human-in-the-loop flow
# ---------------------------------------------------------------------------

def test_approval_decision_and_save(tmp_path):
    """
    tmp_path is a built-in pytest fixture: a temporary folder that gets
    deleted after the test runs. We use it so this test never writes
    into your real data/outputs/approved_actions.csv.
    """
    fake_action = {
        "action_id": "TEST_001",
        "store_id": "S003",
        "product_id": "P001",
        "recommended_action": "Reorder urgently",
        "recommended_quantity": 100,
    }

    decision = submit_approval_decision(fake_action, approved=True, approved_by="test_user")
    assert decision["approval_status"] == "approved"

    test_csv_path = str(tmp_path / "approved_actions.csv")
    save_approved_action(decision, filepath=test_csv_path)

    history = get_approval_history(filepath=test_csv_path)
    assert len(history) == 1
    assert history.iloc[0]["action_id"] == "TEST_001"