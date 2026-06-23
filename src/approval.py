"""
approval.py
-------------
The human-approval checkpoint. The agent never executes an action on its
own — it requests approval, and a person (in the Streamlit UI) clicks
Approve or Reject. This file handles both halves of that handoff.

Three functions:
1. request_human_approval(action_summary) — SAME name/shape as the
   version that was inline in agent_tools.py. Called by agent_graph.py
   to mark a recommendation as "pending."
2. submit_approval_decision(...) — called when the UI's Approve/Reject
   button is clicked. Takes one row's worth of recommendation data
   (e.g. from action_recommendation.csv) plus the decision.
3. save_approved_action(...) — appends the finalized decision to
   data/outputs/approved_actions.csv, matching the fields from the
   project doc (Module 11).

⚠️ NOTE: agent_tools.py currently has its own inline copy of
   request_human_approval — same duplicate-logic issue as
   calculate_inventory_cover/detect_stockout_risk/recommend_business_action.
   Delete it there and import from here instead (see the message above
   this code for the exact fix). - [done]
"""

import os
from datetime import datetime
import pandas as pd


APPROVED_ACTIONS_PATH = "data/outputs/approved_actions.csv"

APPROVAL_COLUMNS = [
    "action_id", "date", "store_id", "product_id",
    "recommended_action", "recommended_quantity",
    "approval_status", "approved_by", "approval_timestamp",
]


def request_human_approval(action_summary: str) -> dict:
    """
    Mark a recommendation as pending approval. Called by agent_graph.py
    right after generate_business_summary().
    """
    return {
        "action_summary": action_summary,
        "approval_status": "pending",
        "requested_at": datetime.now().isoformat(),
    }


def submit_approval_decision(action_row: dict, approved: bool, approved_by: str = "user") -> dict:
    """
    Called when someone clicks Approve/Reject in the Streamlit UI.

    action_row: one row's worth of fields, e.g. from action_recommendation.csv
        {"action_id": ..., "store_id": ..., "product_id": ...,
         "recommended_action": ..., "recommended_quantity": ...}
    approved: True if the user clicked Approve, False if Reject.
    approved_by: who made the decision — pass a real username later if you add login.
    """
    record = dict(action_row)  # copy so we don't mutate the caller's original dict
    record["approval_status"] = "approved" if approved else "rejected"
    record["approved_by"] = approved_by
    record["approval_timestamp"] = datetime.now().isoformat()
    return record


def save_approved_action(record: dict, filepath: str = APPROVED_ACTIONS_PATH) -> None:
    """
    Append one approval decision as a new row in approved_actions.csv.
    Creates the file with headers if it doesn't exist yet, otherwise
    appends below the existing rows.
    """
    row = {col: record.get(col, "") for col in APPROVAL_COLUMNS}
    if not row["date"]:
        row["date"] = datetime.now().strftime("%Y-%m-%d")

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    file_exists = os.path.exists(filepath)
    pd.DataFrame([row]).to_csv(filepath, mode="a", header=not file_exists, index=False)


def get_approval_history(filepath: str = APPROVED_ACTIONS_PATH) -> pd.DataFrame:
    """Read back every past approval decision — used by the Action Plan / Governance UI pages."""
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=APPROVAL_COLUMNS)
    return pd.read_csv(filepath)


if __name__ == "__main__":
    # Simulate the full loop: agent flags something -> user clicks Approve -> it gets saved.
    fake_action_row = {
        "action_id": "S003_P001_TEST",
        "store_id": "S003",
        "product_id": "P001",
        "recommended_action": "Reorder urgently",
        "recommended_quantity": 100,
    }

    pending = request_human_approval("Reorder 100 units of P001 for Store S003 — Critical risk.")
    print("Pending:", pending)

    decision = submit_approval_decision(fake_action_row, approved=True, approved_by="intern_demo")
    print("Decision:", decision)

    save_approved_action(decision)

    print("\nApproval history so far:")
    print(get_approval_history())