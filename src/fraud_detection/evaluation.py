"""Cost-based decision threshold selection and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

DEFAULT_THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)


def cost_curve(y_true, proba, cost_fn: float = COST_FALSE_NEGATIVE,
               cost_fp: float = COST_FALSE_POSITIVE,
               thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Expected business cost, FP and FN counts for every threshold."""
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba)
    thresholds = np.asarray(thresholds)

    # (n_thresholds, n_samples) boolean matrix of "flagged as fraud"
    flagged = proba[None, :] >= thresholds[:, None]
    fp = (flagged & ~y_true).sum(axis=1)
    fn = (~flagged & y_true).sum(axis=1)
    return pd.DataFrame({
        "threshold": thresholds,
        "cost": fp * cost_fp + fn * cost_fn,
        "fp": fp,
        "fn": fn,
    })


def choose_threshold(y_true, proba, cost_fn: float = COST_FALSE_NEGATIVE,
                     cost_fp: float = COST_FALSE_POSITIVE,
                     thresholds=DEFAULT_THRESHOLDS):
    """Pick the probability threshold that minimises expected business cost.

    Returns ``(threshold, cost_table)``. Ties resolve to the lowest
    threshold, i.e. the one that catches the most fraud.
    """
    table = cost_curve(y_true, proba, cost_fn, cost_fp, thresholds)
    best = table.loc[table["cost"].idxmin(), "threshold"]
    return float(best), table
