"""Real-data track 2: the ULB credit card fraud dataset.

284,807 card transactions by European cardholders over two days in
September 2013, 492 of them fraud (Dal Pozzolo et al., ULB Machine Learning
Group). Fetched from OpenML (dataset 42175, which keeps the ``Time``
column). For confidentiality the features are PCA components (V1–V28) plus
``Time`` and ``Amount``, and there is no card or customer ID, so customer
behaviour features, clustered intervals and attack-level metrics cannot be
computed. What can be done — and is, with the shared protocol — is an
honest out-of-time evaluation with a cost-based threshold.

The data covers only 48 hours, so a realistic multi-week label delay cannot
be reproduced; a 6-hour embargo is kept solely so the threshold is placed
on scores the final model has not been trained on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import (
    CHARGEBACK_FEE,
    COST_FALSE_POSITIVE,
    CV_FOLDS,
    DATA_DIR,
    RANDOM_STATE,
    REPORTS,
    THRESHOLD_FOLDS,
)
from fraud_detection.protocol import (
    cv_gap_rows,
    deploy_threshold,
    evaluate_test,
    select_model,
    selection_diagnostics,
    split_with_embargo,
    tune_alert_rate,
)

log = logging.getLogger(__name__)

OPENML_ID = 42175
ULB_REPORTS = REPORTS / "ulb"
# "September 2013"; the exact days are not published
REFERENCE_DATE = pd.Timestamp("2013-09-01")
ULB_TEST_START = "2013-09-02 12:00"       # last 12 hours are the test period
ULB_LABEL_DELAY_DAYS = 0.25
PCA = [f"V{i}" for i in range(1, 29)]


def load_ulb(data_home: Path = DATA_DIR / "openml") -> pd.DataFrame:
    """Fetch (and cache) the dataset from OpenML as a transaction log."""
    from sklearn.datasets import fetch_openml

    raw = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home=data_home,
                       parser="auto").frame
    return to_log(raw)


def to_log(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["transaction_id"] = np.arange(len(df))
    df["timestamp"] = REFERENCE_DATE + pd.to_timedelta(df["Time"], unit="s")
    df["amount"] = df["Amount"].astype("float64")
    df["is_fraud"] = df["Class"].astype(float).astype(int)
    return df.sort_values(["timestamp", "transaction_id"],
                          kind="stable").reset_index(drop=True)


class ULBFeatures(BaseEstimator, TransformerMixin):
    """PCA components, log amount and hour of day (stateless)."""

    names = [*PCA, "log_amount", "hour"]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[PCA].astype("float64").copy()
        out["log_amount"] = np.log1p(X["amount"])
        out["hour"] = pd.to_datetime(X["timestamp"]).dt.hour
        return out[self.names]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.names, dtype=object)


def build_ulb_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("features", ULBFeatures()), ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.1,
                                       class_weight="balanced")),
        ]),
        "gradient_boosting": Pipeline([
            ("features", ULBFeatures()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                min_samples_leaf=50, l2_regularization=1.0,
                class_weight="balanced", early_stopping=False,
                random_state=random_state)),
        ]),
    }


def train_and_evaluate_ulb(data: pd.DataFrame | None = None,
                           reports_dir: Path | None = ULB_REPORTS,
                           chargeback_fee: float = CHARGEBACK_FEE,
                           cost_fp: float = COST_FALSE_POSITIVE,
                           cv_folds: int = CV_FOLDS,
                           test_start: str | None = ULB_TEST_START,
                           label_delay_days: float = ULB_LABEL_DELAY_DAYS,
                           threshold_folds: int = THRESHOLD_FOLDS,
                           random_state: int = RANDOM_STATE) -> dict:
    """The shared protocol on ULB. ``data`` takes a log from :func:`to_log`."""
    df = load_ulb() if data is None else data
    log.info("ULB: %s transactions, %.3f%% fraud", f"{len(df):,}",
             df["is_fraud"].mean() * 100)
    train, embargo, test, cut = split_with_embargo(df, test_start,
                                                   label_delay_days)
    y_train = train["is_fraud"]
    candidates = build_ulb_models(random_state)
    best_name, results, oof, folds = select_model(
        candidates, train, y_train, cv_folds,
        gap=cv_gap_rows(train, label_delay_days, cv_folds))
    threshold_cv, target_rate, recent = tune_alert_rate(
        y_train, oof[best_name], folds[best_name],
        train["amount"].to_numpy() + chargeback_fee, cost_fp, cv_folds,
        threshold_folds)
    best_model = candidates[best_name].fit(train, y_train)
    threshold, method = deploy_threshold(best_model, embargo, target_rate,
                                         threshold_cv, threshold_folds)
    proba = best_model.predict_proba(test)[:, 1]
    # no customer IDs: transactions are resampled individually
    ev = evaluate_test(test, proba, threshold, threshold_cv, chargeback_fee,
                       cost_fp, groups=None, seed=random_state)
    results[best_name].update(ev["ranking"])
    selection_diagnostics(candidates, best_name, train, y_train, test,
                          test["is_fraud"], results)

    perm = permutation_importance(
        best_model[1:], best_model.named_steps["features"].transform(test),
        test["is_fraud"], n_repeats=5, random_state=random_state,
        scoring="average_precision", n_jobs=-1)
    importance = sorted(zip(ULBFeatures.names, perm.importances_mean,
                            strict=True), key=lambda t: t[1], reverse=True)

    summary = {
        "data": {
            "source": "ULB credit card fraud (Dal Pozzolo et al.), OpenML 42175",
            "transactions": len(df),
            "train_transactions": len(train), "test_transactions": len(test),
            "train_frauds": int(y_train.sum()),
            "test_frauds": int(test["is_fraud"].sum()),
            "train_fraud_rate": round(float(y_train.mean()), 5),
            "test_fraud_rate": round(float(test["is_fraud"].mean()), 5),
            "label_delay_days": label_delay_days,
            "test_start": str(cut),
        },
        "models": results,
        "best_model": best_name,
        **ev["operating_point"],
        "business_cost": {**ev["business_cost"], "threshold_method": method,
                          "target_alert_rate": round(target_rate, 5)},
        "confidence_intervals_95": ev["confidence_intervals_95"],
        "alert_budget": ev["alert_budget"],
        "feature_importance": [{"feature": f, "importance": round(float(v), 4)}
                               for f, v in importance],
    }
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n")
    return {
        "summary": summary, "model": best_model, "threshold": threshold,
        "threshold_cv": threshold_cv,
        "train": train.assign(oof_proba=oof[best_name], threshold_fold=recent),
        "test": test.assign(proba=proba, flagged=ev["flagged"].astype(int)),
        "y_test": test["is_fraud"], "proba": proba,
    }
