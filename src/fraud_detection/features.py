"""Feature engineering for card transactions.

Two layers, mirroring how fraud features are served in production:

1. :func:`add_behavioural_features` — point-in-time customer history
   (velocity, spend vs. the customer's own norm, device age). Computed once
   over the time-ordered log; each row only sees earlier transactions.
2. :class:`FraudFeatures` — a stateless-per-row sklearn transformer that
   turns one enriched transaction into the model matrix. Anything it
   *learns* (category spend medians) is learned in ``fit``, i.e. on
   training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraud_detection.simulation import CATEGORIES

# hours (inclusive) treated as "night", when legitimate activity is rare
NIGHT_HOURS = (0, 5)

# --------------------------------------------------------------------------- #
# Behavioural features for the simulated transaction log
# --------------------------------------------------------------------------- #
# Columns added by add_behavioural_features. Every one of them is computed
# from the customer's *earlier* transactions only (point-in-time correct),
# exactly as a production feature store would serve them at authorisation.
BEHAVIOUR_COLUMNS = [
    "prior_txn_count", "secs_since_last_txn", "txn_count_1h", "txn_count_24h",
    "amount_sum_24h", "small_txn_count_1h", "amount_vs_customer_median",
    "device_age_days", "new_device", "new_category",
]
SMALL_AMOUNT = 10.0          # € — "test charge" territory for card testers
_NO_HISTORY_SECS = 30 * 86_400


def add_behavioural_features(log: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time customer behaviour features to a transaction log.

    The log must contain ``transaction_id``, ``timestamp``, ``customer_id``,
    ``amount``, ``channel``, ``device_id`` and ``merchant_category``.
    Returns a copy in the original row order.
    """
    df = log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    order = df.sort_values(["customer_id", "timestamp", "transaction_id"],
                           kind="stable").index
    df = df.loc[order]
    g = df.groupby("customer_id", sort=False)

    df["prior_txn_count"] = g.cumcount()
    gap = g["timestamp"].diff().dt.total_seconds()
    df["secs_since_last_txn"] = gap.fillna(_NO_HISTORY_SECS)

    # rolling windows over the *preceding* hour / day (closed="left"
    # excludes the current transaction)
    timed = df.set_index("timestamp")
    timed["is_small"] = (timed["amount"] < SMALL_AMOUNT).astype(float)
    tg = timed.groupby("customer_id", sort=False)

    def window(col, span, how):
        r = getattr(tg[col].rolling(span, closed="left"), how)()
        return r.fillna(0).to_numpy()

    df["txn_count_1h"] = window("amount", "1h", "count")
    df["txn_count_24h"] = window("amount", "24h", "count")
    df["amount_sum_24h"] = window("amount", "24h", "sum")
    df["small_txn_count_1h"] = window("is_small", "1h", "sum")

    past_median = g["amount"].transform(
        lambda s: s.expanding().median().shift())
    df["amount_vs_customer_median"] = (df["amount"] / past_median).fillna(1.0)

    # a device stays "new" for its first day: the whole burst of an account
    # takeover comes from a device the customer has never used before.
    # min() over the group is always at or before the current row.
    online = df["channel"] == "online"
    first_seen = df.groupby(["customer_id", "device_id"])["timestamp"]\
        .transform("min")
    age = (df["timestamp"] - first_seen).dt.total_seconds() / 86_400
    df["device_age_days"] = age.where(online, _NO_HISTORY_SECS / 86_400)
    df["new_device"] = (online & (age < 1)).astype(int)
    df["new_category"] = (
        df.groupby(["customer_id", "merchant_category"]).cumcount() == 0
    ).astype(int)
    return df.sort_index()


# columns of the enriched log that FraudFeatures consumes
RAW_COLUMNS = [
    "timestamp", "amount", "merchant_category", "merchant_risk", "channel",
    "foreign_country", "distance_from_home", *BEHAVIOUR_COLUMNS,
]

FEATURES = [
    "log_amount", "log_amount_vs_customer", "log_amount_vs_category",
    "hour", "is_night", "is_online", "foreign_country", "log_distance",
    "merchant_risk", "txn_count_1h", "txn_count_24h", "log_amount_sum_24h",
    "small_txn_count_1h", "log_secs_since_last_txn", "log_prior_txn_count",
    "log_device_age_days", "new_device", "new_category",
    *[f"cat_{c}" for c in CATEGORIES],
]


class FraudFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: enriched transactions -> model feature matrix.

    Learns the median amount per merchant category during ``fit`` so that,
    inside a Pipeline, no statistic is ever computed from evaluation data.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.category_median_ = X.groupby("merchant_category")["amount"]\
            .median().to_dict()
        self.global_median_ = float(X["amount"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        hour = pd.to_datetime(X["timestamp"]).dt.hour
        cat_median = X["merchant_category"].map(self.category_median_)\
            .fillna(self.global_median_)
        out = pd.DataFrame({
            "log_amount": np.log1p(X["amount"]),
            "log_amount_vs_customer": np.log(X["amount_vs_customer_median"]),
            "log_amount_vs_category": np.log(X["amount"] / cat_median),
            "hour": hour,
            "is_night": hour.between(*NIGHT_HOURS).astype(int),
            "is_online": (X["channel"] == "online").astype(int),
            "foreign_country": X["foreign_country"],
            "log_distance": np.log1p(X["distance_from_home"]),
            "merchant_risk": X["merchant_risk"],
            "txn_count_1h": X["txn_count_1h"],
            "txn_count_24h": X["txn_count_24h"],
            "log_amount_sum_24h": np.log1p(X["amount_sum_24h"]),
            "small_txn_count_1h": X["small_txn_count_1h"],
            "log_secs_since_last_txn": np.log1p(X["secs_since_last_txn"]),
            "log_prior_txn_count": np.log1p(X["prior_txn_count"]),
            "log_device_age_days": np.log1p(X["device_age_days"]),
            "new_device": X["new_device"],
            "new_category": X["new_category"],
        }, index=X.index)
        for c in CATEGORIES:
            out[f"cat_{c}"] = (X["merchant_category"] == c).astype(int)
        return out[FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)
