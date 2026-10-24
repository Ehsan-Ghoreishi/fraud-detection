"""Candidate model pipelines compared during training."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import RANDOM_STATE
from fraud_detection.features import FraudFeatures


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Return fresh, unfitted candidate pipelines keyed by name.

    Every pipeline starts with FraudFeatures so feature statistics are
    learned on whatever data the pipeline is fitted on, and nothing else.
    """
    return {
        "logistic_regression": Pipeline([
            ("features", FraudFeatures()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000,
                                       class_weight="balanced",
                                       random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("features", FraudFeatures()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1, random_state=random_state
            )),
        ]),
    }
