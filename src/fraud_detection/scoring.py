"""Persist a trained model with its threshold and score new transactions."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fraud_detection import __version__
from fraud_detection.config import MODELS_DIR
from fraud_detection.data import LOG_COLUMNS
from fraud_detection.explain import describe, reason_codes
from fraud_detection.features import RAW_COLUMNS, add_behavioural_features

MODEL_PATH = MODELS_DIR / "fraud_model.joblib"


def save_model(model, threshold: float, metadata: dict | None = None,
               path: Path = MODEL_PATH, reference=None) -> Path:
    """Save the fitted pipeline together with the threshold it ships with.

    The threshold is part of the decision rule, so it travels with the
    model rather than living in a separate config that can drift.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": model,
        "threshold": float(threshold),
        "reference": reference,
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
    """Score a transaction log; adds ``fraud_probability``, ``flag`` and
    ``reasons`` (reason codes for flagged rows, when the bundle has a
    reference).

    Behavioural features need each customer's history, so pass the log
    including earlier transactions — every row is scored using only the
    rows before it, exactly as at authorisation time.
    """
    missing = [c for c in LOG_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    enriched = add_behavioural_features(transactions)
    X = enriched[RAW_COLUMNS]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    flag = proba >= bundle["threshold"]
    reasons = pd.Series("", index=transactions.index)
    if bundle.get("reference") is not None and flag.any():
        codes = reason_codes(bundle["pipeline"], X[flag], bundle["reference"])
        reasons[flag] = [describe(c) for c in codes]
    return transactions.assign(
        fraud_probability=proba.round(4),
        flag=flag.astype(int),
        reasons=reasons,
    )
