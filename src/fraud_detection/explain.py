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


def reference_row(features: pd.DataFrame) -> pd.Series:
    """Typical feature values: medians, and the most common category."""
    ref = features.median()
    cats = GROUPS["merchant_category"]
    ref[cats] = 0
    ref[features[cats].sum().idxmax()] = 1
    return ref


def reason_codes(pipeline, X: pd.DataFrame, reference: pd.Series,
                 top: int = 3) -> list[list[str]]:
    """Top ``top`` reason groups for each row of raw model inputs ``X``."""
    features = pipeline.named_steps["features"].transform(X)
    head = pipeline[1:]
    base = head.predict_proba(features)[:, 1]
    drops = {}
    for group, cols in GROUPS.items():
        what_if = features.copy()
        what_if[cols] = reference[cols].to_numpy()
        drops[group] = base - head.predict_proba(what_if)[:, 1]
    drops = pd.DataFrame(drops, index=features.index)
    order = np.argsort(-drops.to_numpy(), axis=1)[:, :top]
    names = drops.columns.to_numpy()
    return [[g for g, d in zip(names[idx], drops.to_numpy()[i, idx], strict=True)
             if d > 0] for i, idx in enumerate(order)]


def describe(reasons: list[str]) -> str:
    return "; ".join(DESCRIPTIONS[r] for r in reasons)
