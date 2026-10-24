"""Domain-driven feature engineering for card transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# raw transaction attributes the pipeline expects as input
RAW_COLUMNS = [
    "amount", "hour", "days_since_last_txn", "txn_count_1h", "new_device",
    "foreign_country", "distance_from_home", "merchant_risk",
]

# hours (inclusive) treated as "night", when legitimate activity is rare
NIGHT_HOURS = (0, 5)

FEATURES = [
    "log_amount", "hour", "is_night", "days_since_last_txn",
    "txn_count_1h", "high_velocity", "new_device", "foreign_country",
    "distance_from_home", "geo_risk", "merchant_risk",
    "device_geo_amount_risk",
]


def engineer_features(df: pd.DataFrame,
                      high_amount_log: float | None = None) -> pd.DataFrame:
    """Add domain-driven features a fraud analyst would reach for.

    ``high_amount_log`` is the log-amount above which a transaction counts
    as "high value". It must be learned on training data only; when omitted
    it falls back to the median of ``df`` (fine for EDA, leaky for models).
    """
    df = df.copy()

    # log amount — fraud amounts are heavy-tailed; log stabilises them
    df["log_amount"] = np.log1p(df["amount"])
    if high_amount_log is None:
        high_amount_log = float(df["log_amount"].median())

    # "odd hour" flag — activity between midnight and 5am is higher risk
    df["is_night"] = df["hour"].between(*NIGHT_HOURS).astype(int)

    # velocity risk — many transactions in a short window is suspicious
    df["high_velocity"] = (df["txn_count_1h"] >= 3).astype(int)

    # geo risk — far from home OR foreign is a classic fraud signal
    df["geo_risk"] = ((df["distance_from_home"] > 100) |
                      (df["foreign_country"] == 1)).astype(int)

    # combined behavioural risk: new device on a foreign, high-value txn
    df["device_geo_amount_risk"] = (
        df["new_device"] * df["foreign_country"] *
        (df["log_amount"] > high_amount_log).astype(int)
    )
    return df


class FraudFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: raw transactions -> model feature matrix.

    Learns the high-value amount cut-off during ``fit`` so that, inside a
    Pipeline, no statistic is ever computed from evaluation data.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.high_amount_log_ = float(np.log1p(X["amount"]).median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X, self.high_amount_log_)[FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


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
