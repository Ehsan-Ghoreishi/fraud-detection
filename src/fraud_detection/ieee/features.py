"""Model features for IEEE-CIS: Vesta's columns plus our behaviour features.

Vesta ships ~430 anonymised columns (C = counts, D = time deltas, M = match
flags, V = engineered features). On top of those, the point-in-time
customer behaviour features from :mod:`fraud_detection.features` are
computed on the reconstructed client ID, so the same feature store logic
runs on simulated and real data. :class:`IEEEFeatures` learns every
category encoding on training data only.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraud_detection.features import BEHAVIOUR_COLUMNS

# raw columns that are identifiers, labels or analysis-only, never inputs
EXCLUDED = {
    "TransactionID", "TransactionDT", "isFraud", "transaction_id", "timestamp",
    "customer_id", "is_fraud", "fraud_episode", "fraud_type", "amount",
    "merchant_category", "device_id", "channel", "foreign_country",
    "distance_from_home", *BEHAVIOUR_COLUMNS,
}
# behaviour features are heavy-tailed counts / durations / ratios
_LOG_BEHAVIOUR = ["secs_since_last_txn", "prior_txn_count", "amount_sum_24h",
                  "device_age_days"]


class IEEEFeatures(BaseEstimator, TransformerMixin):
    """Raw IEEE-CIS rows (+ behaviour columns) -> numeric model matrix.

    * numeric Vesta columns pass through (NaN kept: gradient boosting
      handles missing values natively, the linear baseline imputes);
    * string columns become an integer code and a frequency, both learned
      in ``fit`` (unseen categories get code -1 and frequency 0);
    * amount, time-of-day and cents features are derived;
    * with ``behaviour=True`` the point-in-time customer features are added.
    """

    def __init__(self, behaviour: bool = True):
        self.behaviour = behaviour

    def fit(self, X: pd.DataFrame, y=None):
        cols = [c for c in X.columns if c not in EXCLUDED]
        # strings are object dtype in pandas 2 and "str" in pandas 3
        self.categorical_ = [c for c in cols
                             if not pd.api.types.is_numeric_dtype(X[c])]
        self.numeric_ = [c for c in cols if c not in self.categorical_]
        self.codes_, self.freqs_ = {}, {}
        for c in self.categorical_:
            counts = X[c].value_counts(normalize=True)
            self.codes_[c] = {v: i for i, v in enumerate(counts.index)}
            self.freqs_[c] = counts.to_dict()
        self.feature_names_ = self._names()
        return self

    def _names(self):
        names = ["log_amount", "amount_cents", "hour", "weekday", *self.numeric_]
        for c in self.categorical_:
            names += [f"{c}__code", f"{c}__freq"]
        if self.behaviour:
            names += [f"bhv_{c}" for c in BEHAVIOUR_COLUMNS]
        return names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(X["timestamp"])
        amt = X["TransactionAmt"].astype("float64")
        parts = {
            "log_amount": np.log1p(amt),
            # non-integer cents often mean a currency conversion
            "amount_cents": (amt * 100).round() % 100,
            "hour": ts.dt.hour,
            "weekday": ts.dt.day_of_week,
        }
        for c in self.numeric_:
            parts[c] = X[c]
        for c in self.categorical_:
            parts[f"{c}__code"] = X[c].map(self.codes_[c]).fillna(-1)
            parts[f"{c}__freq"] = X[c].map(self.freqs_[c]).fillna(0.0)
        if self.behaviour:
            for c in BEHAVIOUR_COLUMNS:
                v = X[c].astype("float64")
                if c in _LOG_BEHAVIOUR:
                    v = np.log1p(v)
                elif c == "amount_vs_customer_median":
                    v = np.log(v)
                parts[f"bhv_{c}"] = v
        out = pd.DataFrame(parts, index=X.index)
        return out[self.feature_names_].astype("float32")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


# --------------------------------------------------------------------------- #
# Feature families: for grouped importance and reason codes
# --------------------------------------------------------------------------- #
_FAMILY_RULES = [
    ("customer behaviour (ours)", r"^bhv_"),
    ("amount", r"^(log_amount|amount_cents|TransactionAmt)$"),
    ("time of day / week", r"^(hour|weekday)$"),
    ("product", r"^ProductCD"),
    ("card", r"^card\d"),
    ("address & distance", r"^(addr|dist)\d"),
    ("email domain", r"^[PR]_emaildomain"),
    ("Vesta counts (C)", r"^C\d+$"),
    ("Vesta time deltas (D)", r"^D\d+$"),
    ("Vesta match flags (M)", r"^M\d"),
    ("Vesta engineered (V)", r"^V\d+$"),
    ("device", r"^Device"),
    ("identity (id_)", r"^id_"),
]

FAMILY_DESCRIPTIONS = {
    "customer behaviour (ours)": "unusual behaviour for this client "
                                 "(velocity, spend, device age)",
    "amount": "unusual amount",
    "time of day / week": "unusual time",
    "product": "risky product type",
    "card": "card attributes (issuer, type, BIN)",
    "address & distance": "billing address / distance",
    "email domain": "purchaser or recipient email domain",
    "Vesta counts (C)": "unusual counts linked to the card (Vesta C)",
    "Vesta time deltas (D)": "unusual time since previous activity (Vesta D)",
    "Vesta match flags (M)": "name / address mismatch flags (Vesta M)",
    "Vesta engineered (V)": "Vesta engineered risk features (V)",
    "device": "device type / model",
    "identity (id_)": "identity and browser signals (id_)",
}


def feature_families(names) -> dict[str, list[str]]:
    """Map each feature name to its family; returns family -> columns."""
    families: dict[str, list[str]] = {}
    for name in names:
        base = name.split("__")[0]
        family = next((f for f, rule in _FAMILY_RULES
                       if re.match(rule, base)), "other")
        families.setdefault(family, []).append(name)
    return families
