"""Project-wide paths, seeds and business assumptions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "transactions.csv"
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
# out-of-time evaluation: train on Jan-Apr, test on May-Jun
TEST_START = "2026-05-01"
CV_FOLDS = 4
# every CV fold model trains on at least this share of the training history
CV_MIN_TRAIN_FRACTION = 0.5
# fraud labels (chargebacks, customer reports) arrive with a delay: at
# deployment the most recent weeks cannot be used for training yet
LABEL_DELAY_DAYS = 14
# the threshold is tuned on the most recent CV folds only: their models saw
# the most history, so their scores are closest to the final model's
THRESHOLD_FOLDS = 2

# Business cost assumptions (euros). These are the lever a risk team tunes.
# A missed fraud costs the transaction amount plus a chargeback handling fee;
# a false alarm costs an analyst review and some customer friction.
CHARGEBACK_FEE = 20.0
COST_FALSE_POSITIVE = 5.0
