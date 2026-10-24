import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_detection.explain import (
    DESCRIPTIONS,
    GROUPS,
    describe,
    reason_codes,
    reference_row,
)
from fraud_detection.features import FEATURES


class _Identity:
    """Stands in for FraudFeatures: inputs already are the feature matrix."""

    def transform(self, X):
        return X


def _pipeline_driven_by(feature):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, len(FEATURES))), columns=FEATURES)
    X[GROUPS["merchant_category"]] = 0
    X["cat_grocery"] = 1
    y = (X[feature] > 1).astype(int)
    clf = LogisticRegression().fit(X, y)
    return Pipeline([("features", _Identity()), ("clf", clf)]), X


def test_reason_is_the_feature_that_drives_the_score():
    pipe, X = _pipeline_driven_by("merchant_risk")
    ref = reference_row(X)
    risky = X.nlargest(5, "merchant_risk")
    reasons = reason_codes(pipe, risky, ref, top=1)
    assert all(r == ["merchant_risk"] for r in reasons)


def test_reference_row_has_exactly_one_category():
    _, X = _pipeline_driven_by("hour")
    ref = reference_row(X)
    assert ref[GROUPS["merchant_category"]].sum() == 1
    assert ref["cat_grocery"] == 1


def test_every_group_has_a_description():
    assert set(GROUPS) == set(DESCRIPTIONS)
    assert describe(["new_device", "merchant_risk"]) == \
        "device first seen in the last 24h; high-risk merchant"
