"""Per-alert reason codes: why was *this* transaction flagged?

Model-agnostic and cheap: for each alert, every feature group is reset to a
typical (reference) value one at a time, and the groups whose reset lowers
the fraud score the most become the alert's reasons. Analysts get "device
first seen 3 minutes ago; 6x usual spend" instead of a bare score, which is
what a regulated review process needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.features import FEATURES

# feature group -> columns; one-hot categories move together
GROUPS = {f: [f] for f in FEATURES if not f.startswith("cat_")}
GROUPS["merchant_category"] = [f for f in FEATURES if f.startswith("cat_")]

DESCRIPTIONS = {
    "log_amount": "large amount",
    "log_amount_vs_customer": "unusual amount for this customer",
    "log_amount_vs_category": "unusual amount for this merchant category",
    "hour": "unusual time of day",
    "is_night": "night-time transaction",
    "is_online": "card-not-present channel",
    "foreign_country": "foreign country",
    "log_distance": "far from home",
    "merchant_risk": "high-risk merchant",
    "txn_count_1h": "burst of transactions in the last hour",
    "txn_count_24h": "many transactions in the last day",
    "log_amount_sum_24h": "high spend in the last day",
    "small_txn_count_1h": "small test charges in the last hour",
    "log_secs_since_last_txn": "unusual gap since the previous transaction",
    "log_prior_txn_count": "little customer history",
    "log_device_age_days": "recently first-seen device",
    "new_device": "device first seen in the last 24h",
    "new_category": "first purchase in this merchant category",
    "merchant_category": "risky merchant category",
}


def reference_row(features: pd.DataFrame,
                  one_hot_groups=(GROUPS["merchant_category"],)) -> pd.Series:
    """Typical feature values: medians, and the most common category of
    each one-hot group (a median of indicator columns is not a valid row)."""
    ref = features.median()
    for cats in one_hot_groups:
        ref[cats] = 0
        ref[features[cats].sum().idxmax()] = 1
    return ref


def reason_codes(pipeline, X: pd.DataFrame, reference: pd.Series,
                 top: int = 3, groups: dict | None = None) -> list[list[str]]:
    """Top ``top`` reason groups for each row of raw model inputs ``X``.

    ``groups`` maps a reason name to its feature columns (default: the
    simulated-data feature groups)."""
    features = pipeline.named_steps["features"].transform(X)
    head = pipeline[1:]
    base = head.predict_proba(features)[:, 1]
    drops = {}
    for group, cols in (groups or GROUPS).items():
        what_if = features.copy()
        what_if[cols] = reference[cols].to_numpy()
        drops[group] = base - head.predict_proba(what_if)[:, 1]
    drops = pd.DataFrame(drops, index=features.index)
    order = np.argsort(-drops.to_numpy(), axis=1)[:, :top]
    names = drops.columns.to_numpy()
    return [[g for g, d in zip(names[idx], drops.to_numpy()[i, idx], strict=True)
             if d > 0] for i, idx in enumerate(order)]


def describe(reasons: list[str], descriptions: dict | None = None) -> str:
    descriptions = descriptions or DESCRIPTIONS
    return "; ".join(descriptions[r] for r in reasons)


def grouped_permutation_importance(head, features: pd.DataFrame, y,
                                   groups: dict, n_repeats: int = 3,
                                   seed: int = 0) -> dict:
    """Drop in PR-AUC when all columns of a group are shuffled together.

    With hundreds of correlated columns (e.g. Vesta's V block), shuffling
    one column at a time understates a family's value because its
    neighbours carry the same signal; shuffling the whole family does not.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    base = average_precision_score(y, head.predict_proba(features)[:, 1])
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = features.copy()
            perm = rng.permutation(len(features))
            shuffled[cols] = features[cols].to_numpy()[perm]
            drops.append(base - average_precision_score(
                y, head.predict_proba(shuffled)[:, 1]))
        out[name] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
