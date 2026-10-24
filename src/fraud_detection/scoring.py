"""Persist a trained model with its threshold and score new transactions."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fraud_detection import __version__
from fraud_detection.config import MODELS_DIR
from fraud_detection.features import RAW_COLUMNS

MODEL_PATH = MODELS_DIR / "fraud_model.joblib"


def save_model(model, threshold: float, metadata: dict | None = None,
               path: Path = MODEL_PATH) -> Path:
    """Save the fitted pipeline together with the threshold it ships with.

    The threshold is part of the decision rule, so it travels with the
    model rather than living in a separate config that can drift.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": model,
        "threshold": float(threshold),
        "package_version": __version__,
        "metadata": metadata or {},
    }, path)
    return path


def load_model(path: Path = MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    if not {"pipeline", "threshold"} <= bundle.keys():
        raise ValueError(f"{path} is not a fraud_detection model bundle")
    return bundle


def score(transactions: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return the input with ``fraud_probability`` and ``flag`` columns."""
    missing = [c for c in RAW_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    proba = bundle["pipeline"].predict_proba(transactions[RAW_COLUMNS])[:, 1]
    return transactions.assign(
        fraud_probability=proba.round(4),
        flag=(proba >= bundle["threshold"]).astype(int),
    )
