"""Candidate model pipelines compared during training."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import RANDOM_STATE
from fraud_detection.features import FraudFeatures


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Return fresh, unfitted candidate pipelines keyed by name.

    Every pipeline starts with FraudFeatures so feature statistics are
    learned on whatever data the pipeline is fitted on, and nothing else.
    The linear model is the interpretable baseline; the tree ensembles can
    pick up interactions such as "new device AND far above usual spend".
    """
    return {
        "logistic_regression": Pipeline([
            ("features", FraudFeatures()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5,
                                       class_weight="balanced",
                                       random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("features", FraudFeatures()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=14, min_samples_leaf=10,
                max_features=0.4, class_weight="balanced_subsample",
                n_jobs=-1, random_state=random_state,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("features", FraudFeatures()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=1.0,
                class_weight="balanced", early_stopping=False,
                random_state=random_state,
            )),
        ]),
    }
