"""Domain-driven feature engineering for card transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven features a fraud analyst would reach for."""
    df = df.copy()

    # log amount — fraud amounts are heavy-tailed; log stabilises them
    df["log_amount"] = np.log1p(df["amount"])

    # "odd hour" flag — activity between midnight and 5am is higher risk
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    # velocity risk — many transactions in a short window is suspicious
    df["high_velocity"] = (df["txn_count_1h"] >= 3).astype(int)

    # geo risk — far from home OR foreign is a classic fraud signal
    df["geo_risk"] = ((df["distance_from_home"] > 100) |
                      (df["foreign_country"] == 1)).astype(int)

    # combined behavioural risk: new device on a foreign, high-value txn
    df["device_geo_amount_risk"] = (
        df["new_device"] * df["foreign_country"] *
        (df["log_amount"] > df["log_amount"].median()).astype(int)
    )
    return df


FEATURES = [
    "log_amount", "hour", "is_night", "days_since_last_txn",
    "txn_count_1h", "high_velocity", "new_device", "foreign_country",
    "distance_from_home", "geo_risk", "merchant_risk",
    "device_geo_amount_risk",
]
