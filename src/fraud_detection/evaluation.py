"""Cost-based decision threshold selection and evaluation helpers.

``cost_fn`` may be a scalar (every missed fraud costs the same) or one value
per transaction, e.g. ``amount + chargeback fee``, so that missing a €900
cash-out weighs more than missing a €2 test charge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detection.config import COST_FALSE_POSITIVE

DEFAULT_THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)


def _per_txn(cost, n: int) -> np.ndarray:
    cost = np.asarray(cost, dtype=float)
    return np.broadcast_to(cost, n) if cost.ndim == 0 else cost


def cost_curve(y_true, proba, cost_fn,
               cost_fp: float = COST_FALSE_POSITIVE,
               thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Expected business cost, FP and FN counts for every threshold."""
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba)
    thresholds = np.asarray(thresholds)
    fn_cost = _per_txn(cost_fn, len(y_true))

    # (n_thresholds, n_samples) boolean matrix of "flagged as fraud"
    flagged = proba[None, :] >= thresholds[:, None]
    missed = ~flagged & y_true
    fp = (flagged & ~y_true).sum(axis=1)
    return pd.DataFrame({
        "threshold": thresholds,
        "cost": fp * cost_fp + missed @ fn_cost,
        "fp": fp,
        "fn": missed.sum(axis=1),
    })


def choose_threshold(y_true, proba, cost_fn,
                     cost_fp: float = COST_FALSE_POSITIVE,
                     thresholds=DEFAULT_THRESHOLDS):
    """Pick the probability threshold that minimises expected business cost.

    Returns ``(threshold, cost_table)``. Ties resolve to the lowest
    threshold, i.e. the one that catches the most fraud.
    """
    table = cost_curve(y_true, proba, cost_fn, cost_fp, thresholds)
    best = table.loc[table["cost"].idxmin(), "threshold"]
    return float(best), table


def bootstrap_ci(y_true, proba, threshold: float, cost_fn,
                 cost_fp: float = COST_FALSE_POSITIVE,
                 baseline_threshold: float = 0.5, n_boot: int = 1_000,
                 level: float = 0.95, seed: int = 0, groups=None) -> dict:
    """Bootstrap confidence intervals for the headline test metrics.

    With ``groups`` (e.g. customer IDs) whole groups are resampled — a
    cluster bootstrap. Fraud arrives in bursts on the same card, so its
    transactions are not independent; resampling them one by one would
    understate the uncertainty. Without ``groups`` transactions are
    resampled, stratified by class.

    ``cost_vs_baseline`` is the cost at ``threshold`` minus the cost at
    ``baseline_threshold``: negative means the tuned threshold is cheaper.
    """
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba)
    fn_cost = _per_txn(cost_fn, len(y_true))
    rng = np.random.default_rng(seed)
    draw = (_cluster_sampler(np.asarray(groups), rng) if groups is not None
            else _stratified_sampler(y_true, rng))

    stats = {k: [] for k in ("pr_auc", "recall", "precision",
                             "cost_vs_baseline")}
    for _ in range(n_boot):
        idx = draw()
        y, p = y_true[idx], proba[idx]
        if not y.any():
            continue
        flagged = p >= threshold
        tp = (flagged & y).sum()
        stats["pr_auc"].append(average_precision_score(y, p))
        stats["recall"].append(tp / y.sum())
        stats["precision"].append(tp / max(flagged.sum(), 1))
        costs = cost_curve(y, p, fn_cost[idx], cost_fp,
                           thresholds=[threshold, baseline_threshold])["cost"]
        stats["cost_vs_baseline"].append(costs.iloc[0] - costs.iloc[1])

    tail = (1 - level) / 2 * 100
    return {k: [round(float(v), 4) for v in np.percentile(s, [tail, 100 - tail])]
            for k, s in stats.items()}


def _stratified_sampler(y_true, rng):
    pos, neg = np.flatnonzero(y_true), np.flatnonzero(~y_true)
    return lambda: np.concatenate([rng.choice(pos, pos.size),
                                   rng.choice(neg, neg.size)])


def _cluster_sampler(groups, rng):
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    members = np.split(order, starts[1:])
    return lambda: np.concatenate(
        [members[i] for i in rng.integers(0, len(members), len(members))])


def value_recall(y_true, flagged, amount) -> float:
    """Share of fraudulent *value* (€) that the flag would have stopped."""
    y_true = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype=float)
    total = amount[y_true].sum()
    return float(amount[y_true & np.asarray(flagged, bool)].sum() / total) \
        if total else float("nan")
