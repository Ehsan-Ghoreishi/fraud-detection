"""Project-wide paths, seeds and business assumptions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "transactions.csv"
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# Business cost assumptions (euros). These are the lever a risk team tunes.
COST_FALSE_NEGATIVE = 150.0   # average loss from letting one fraud through
COST_FALSE_POSITIVE = 5.0     # cost of reviewing/blocking one legit txn
