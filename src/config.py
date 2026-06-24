"""
config.py
-----------
Single source of truth for file paths and shared settings used across
the project — import from here instead of re-typing paths in every file.

NOTE: your existing files (feature_engineering.py, inventory_risk.py,
etc.) currently have these paths hardcoded inline — that still works
fine and doesn't need to change today under time pressure. Use this
file for anything NEW you write. Swapping old files over to import
from here is a nice cleanup if you get spare time, not a fix for
something broken.

⚠️ DELIBERATELY NOT INCLUDED HERE: the CATEGORY_RISK_THRESHOLDS dict.
It already lives in inventory_risk.py — copying it here too would
recreate the exact "two sources of truth" bug we fixed earlier in the
project (the same one with calculate_inventory_cover/detect_stockout_risk).
If you ever want to formally centralize it, CUT it out of
inventory_risk.py and import from here instead — never keep both.
"""

import os

# --- Folders ---
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_OUTPUTS_DIR = "data/outputs"
DOCS_DIR = "docs"

# --- Specific files, by pipeline stage ---
RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, "retail_inventory.csv")
CLEANED_DATA_PATH = os.path.join(DATA_PROCESSED_DIR, "retail_cleaned.csv")
FEATURES_DATA_PATH = os.path.join(DATA_PROCESSED_DIR, "retail_features.csv")

FORECAST_OUTPUT_PATH = os.path.join(DATA_OUTPUTS_DIR, "forecast_output.csv")
RISK_OUTPUT_PATH = os.path.join(DATA_OUTPUTS_DIR, "risk_output.csv")
ACTION_RECOMMENDATION_PATH = os.path.join(DATA_OUTPUTS_DIR, "action_recommendation.csv")
APPROVED_ACTIONS_PATH = os.path.join(DATA_OUTPUTS_DIR, "approved_actions.csv")
DATA_QUALITY_REPORT_PATH = os.path.join(DATA_OUTPUTS_DIR, "data_quality_report.json")

# --- Shared settings ---
RANDOM_SEED = 42                               # for reproducible model training/splits
TEST_SIZE = 0.2                                # train/test split fraction (forecasting.py)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"      # used in rag.py


def ensure_output_dirs_exist():
    """Call this once at the start of any script to guarantee the output folders exist."""
    for folder in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_OUTPUTS_DIR, DOCS_DIR):
        os.makedirs(folder, exist_ok=True)


if __name__ == "__main__":
    # Run with: python src/config.py — prints every path/setting as a sanity check
    print("Project configuration:")
    for name, value in list(globals().items()):
        if name.isupper():
            print(f"  {name} = {value}")