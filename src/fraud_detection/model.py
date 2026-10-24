"""
Fraud detection pipeline: feature engineering, model training, and
threshold-aware evaluation.

Design choices that matter for a fraud/risk context (and that this file
demonstrates deliberately):

1. Never evaluate fraud models on accuracy. With ~1.7% fraud, a model that
   predicts "never fraud" is 98.3% accurate and 100% useless. We use
   precision, recall, PR-AUC and a cost-based threshold instead.

2. Train/test split is stratified so both sets keep the same fraud ratio,
   and all preprocessing is fit on the training data only (no leakage).

3. Class imbalance is handled with class_weight rather than naive
   oversampling, so we don't leak synthetic rows into evaluation.

4. The final threshold is chosen from a business cost model (a missed fraud
   costs far more than a false alarm), not left at the default 0.5 — this is
   the decision a risk analyst actually owns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score,
)

from fraud_detection.features import FEATURES
from fraud_detection.evaluation import choose_threshold, cost_curve
from fraud_detection.models import build_models
from fraud_detection.config import (
    COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, CV_FOLDS, DATA_PATH, RANDOM_STATE,
    REPORTS, TEST_SIZE,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #
def train_and_evaluate(data: pd.DataFrame | None = None,
                       data_path: Path = DATA_PATH,
                       reports_dir: Path | None = REPORTS,
                       cost_fn: float = COST_FALSE_NEGATIVE,
                       cost_fp: float = COST_FALSE_POSITIVE,
                       cv_folds: int = CV_FOLDS,
                       random_state: int = RANDOM_STATE) -> dict:
    """Select a model, tune its threshold and evaluate it on held-out data.

    Pass ``data`` to train on an in-memory frame instead of ``data_path``.
    Set ``reports_dir=None`` to skip writing metrics.json.
    """
    df = pd.read_csv(data_path) if data is None else data
    X, y = df.drop(columns="is_fraud"), df["is_fraud"]

    # stratified split preserves the fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )

    # Model selection uses out-of-fold predictions on the training set only;
    # the test set is touched exactly once, by the final chosen model.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_state)
    results = {}
    models = build_models(random_state)

    best_name, best_pr_auc, oof = None, -1.0, {}
    for name, model in models.items():
        oof_proba = cross_val_predict(model, X_train, y_train, cv=cv,
                                      method="predict_proba")[:, 1]
        oof[name] = oof_proba

        roc_auc = roc_auc_score(y_train, oof_proba)
        pr_auc = average_precision_score(y_train, oof_proba)   # key metric for imbalance

        results[name] = {
            "cv_roc_auc": round(roc_auc, 4),
            "cv_pr_auc": round(pr_auc, 4),
        }
        log.info(f"{name:>22} | CV ROC-AUC {roc_auc:.4f} | CV PR-AUC {pr_auc:.4f}")

        if pr_auc > best_pr_auc:
            best_name, best_pr_auc = name, pr_auc

    best_model = models[best_name].fit(X_train, y_train)
    best_proba = best_model.predict_proba(X_test)[:, 1]
    results[best_name]["test_roc_auc"] = round(roc_auc_score(y_test, best_proba), 4)
    results[best_name]["test_pr_auc"] = round(
        average_precision_score(y_test, best_proba), 4)

    log.info(f"\nBest model by PR-AUC: {best_name}\n")

    # ---- cost-based threshold selection ------------------------------------
    # tuned on out-of-fold training predictions, then applied unchanged to
    # the test set — exactly how it would be frozen before going live
    threshold, _ = choose_threshold(y_train, oof[best_name],
                                    cost_fn=cost_fn, cost_fp=cost_fp)
    preds = (best_proba >= threshold).astype(int)

    log.info(f"Cost-optimal threshold: {threshold:.3f}")
    log.info("\nClassification report at cost-optimal threshold:\n")
    log.info(classification_report(y_test, preds, digits=3,
                                target_names=["legit", "fraud"]))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    caught = tp / (tp + fn)
    review_rate = (tp + fp) / len(y_test)
    log.info(f"Fraud caught (recall):     {caught:.1%}")
    log.info(f"Transactions flagged:      {review_rate:.1%} of all traffic")
    log.info(f"False alarms per real fraud: {fp / max(tp, 1):.1f}")

    # what the threshold choice is worth, measured on unseen data
    test_costs = cost_curve(y_test, best_proba, cost_fn, cost_fp,
                            thresholds=[threshold, 0.5])["cost"]
    cost_at_threshold, cost_at_default = map(float, test_costs)
    log.info(f"Test-set cost at {threshold:.2f}: €{cost_at_threshold:,.0f} "
          f"(vs €{cost_at_default:,.0f} at 0.50)")

    # persist metrics for the README / reports
    summary = {
        "models": results,
        "best_model": best_name,
        "cost_optimal_threshold": round(threshold, 3),
        "recall_fraud": round(caught, 3),
        "precision_fraud": round(tp / max(tp + fp, 1), 3),
        "flagged_rate": round(review_rate, 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "business_cost": {
            "cost_false_negative": cost_fn,
            "cost_false_positive": cost_fp,
            "test_cost_at_threshold": cost_at_threshold,
            "test_cost_at_default_0_5": cost_at_default,
            "test_savings_vs_default": cost_at_default - cost_at_threshold,
        },
    }

    # feature importance via permutation — works for ANY model and measures
    # real predictive contribution. Explainability is non-negotiable in risk:
    # a fraud team must be able to justify why a customer was flagged.
    # permute the engineered features (not the raw columns) so importances
    # line up with what the classifier actually sees
    X_test_features = best_model.named_steps["features"].transform(X_test)
    perm = permutation_importance(
        best_model[1:], X_test_features, y_test, n_repeats=10,
        random_state=random_state, scoring="average_precision", n_jobs=-1,
    )
    importances = sorted(
        zip(FEATURES, perm.importances_mean),
        key=lambda t: t[1], reverse=True,
    )
    summary["feature_importance"] = [
        {"feature": f, "importance": round(float(v), 4)} for f, v in importances
    ]
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    log.info("\nTop fraud signals the model relies on (permutation importance):")
    for f, v in importances[:6]:
        log.info(f"  {f:>24}: {v:.4f}")

    return {"summary": summary, "model": best_model,
            "y_test": y_test, "proba": best_proba,
            "y_train": y_train, "oof_proba": oof[best_name],
            "threshold": threshold}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_and_evaluate()
