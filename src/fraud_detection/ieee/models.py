"""Candidate pipelines for the IEEE-CIS track."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import RANDOM_STATE
from fraud_detection.ieee.features import IEEEFeatures


def _boosting(random_state):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=1.0, max_features=0.5,
        class_weight="balanced", early_stopping=False,
        random_state=random_state,
    )


def build_ieee_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Fresh candidates. ``gradient_boosting_vesta_only`` is the ablation:
    the same model without our point-in-time behaviour features, so the
    comparison measures what that feature engineering adds on real data."""
    return {
        "logistic_regression": Pipeline([
            ("features", IEEEFeatures()),
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.05, max_iter=300,
                                       class_weight="balanced")),
        ]),
        "gradient_boosting_vesta_only": Pipeline([
            ("features", IEEEFeatures(behaviour=False)),
            ("clf", _boosting(random_state)),
        ]),
        "gradient_boosting": Pipeline([
            ("features", IEEEFeatures()),
            ("clf", _boosting(random_state)),
        ]),
    }
