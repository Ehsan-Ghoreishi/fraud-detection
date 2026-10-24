"""Domain-driven feature engineering for card transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
