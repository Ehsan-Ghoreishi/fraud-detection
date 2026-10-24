"""
Fraud detection pipeline: model selection, threshold tuning and out-of-time
evaluation.

Design choices that matter for a fraud/risk context:

1. Never evaluate fraud models on accuracy. With ~0.5% fraud, a model that
   predicts "never fraud" is 99.5% accurate and 100% useless. We use
   PR-AUC, precision, recall, value caught and a cost-based threshold.

2. Time is respected everywhere. Behavioural features only look at earlier
   transactions, the test set is the last two months (out-of-time), and
   cross-validation is forward-chaining, so drift shows up in the numbers
   instead of being averaged away.

3. Class imbalance is handled with class weights rather than oversampling,
   so no synthetic rows leak into evaluation.

4. The decision threshold minimises expected business cost — a missed fraud
   costs its amount plus a chargeback fee, a false alarm costs a review —
   and is tuned on out-of-fold training predictions, then frozen.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from fraud_detection.config import (
    CHARGEBACK_FEE,
    COST_FALSE_POSITIVE,
    CV_FOLDS,
    DATA_PATH,
    LABEL_DELAY_DAYS,
    RANDOM_STATE,
    REPORTS,
    TEST_START,
    THRESHOLD_FOLDS,
)
from fraud_detection.data import load_transactions
from fraud_detection.evaluation import episode_metrics, value_recall
from fraud_detection.explain import describe, reason_codes, reference_row
from fraud_detection.features import FEATURES, RAW_COLUMNS, add_behavioural_features
from fraud_detection.models import build_models
from fraud_detection.protocol import (
    cv_gap_rows,
    deploy_threshold,
    evaluate_test,
    select_model,
    selection_diagnostics,
    split_with_embargo,
    tune_alert_rate,
)
from fraud_detection.rules import compare_with_rules
from fraud_detection.simulation import FRAUD_TYPES

log = logging.getLogger(__name__)


def train_and_evaluate(data: pd.DataFrame | None = None,
                       data_path: Path = DATA_PATH,
                       reports_dir: Path | None = REPORTS,
                       chargeback_fee: float = CHARGEBACK_FEE,
                       cost_fp: float = COST_FALSE_POSITIVE,
                       cv_folds: int = CV_FOLDS,
                       test_start: str | None = TEST_START,
                       label_delay_days: float = LABEL_DELAY_DAYS,
                       threshold_folds: int = THRESHOLD_FOLDS,
                       models: list[str] | None = None,
                       random_state: int = RANDOM_STATE) -> dict:
    """Select a model, tune its threshold and evaluate it out of time.

    Pass ``data`` (a raw transaction log) instead of ``data_path`` to train
    in memory, ``test_start=None`` to cut at two thirds of the period, and
    ``models`` to restrict the candidates. ``reports_dir=None`` skips
    writing metrics.json.
    """
    raw = load_transactions(data_path) if data is None else data
    enriched = add_behavioural_features(raw).sort_values(
        ["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    train, embargo, test, cut = split_with_embargo(enriched, test_start,
                                                   label_delay_days)
    X_train, y_train = train[RAW_COLUMNS], train["is_fraud"]
    X_test, y_test = test[RAW_COLUMNS], test["is_fraud"]

    candidates = build_models(random_state)
    if models:
        candidates = {k: candidates[k] for k in models}
    best_name, results, oof, folds = select_model(
        candidates, X_train, y_train, cv_folds,
        gap=cv_gap_rows(train, label_delay_days, cv_folds))

    threshold_cv, target_rate, recent = tune_alert_rate(
        y_train, oof[best_name], folds[best_name],
        train["amount"].to_numpy() + chargeback_fee, cost_fp, cv_folds,
        threshold_folds)
    best_model = candidates[best_name].fit(X_train, y_train)
    threshold, threshold_method = deploy_threshold(
        best_model, embargo[RAW_COLUMNS], target_rate, threshold_cv,
        threshold_folds)

    proba = best_model.predict_proba(X_test)[:, 1]
    ev = evaluate_test(test, proba, threshold, threshold_cv, chargeback_fee,
                       cost_fp, groups=test["customer_id"], seed=random_state)
    flagged = ev["flagged"]
    results[best_name].update(ev["ranking"])
    selection_diagnostics(candidates, best_name, X_train, y_train, X_test,
                          y_test, results)
    cost = ev["business_cost"]
    summary = {
        "data": {
            "train_transactions": len(train), "test_transactions": len(test),
            "train_fraud_rate": round(float(y_train.mean()), 5),
            "test_fraud_rate": round(float(y_test.mean()), 5),
            "train_end": str(train["timestamp"].max().date()),
            "label_delay_days": label_delay_days,
            "test_start": str(cut.date()),
        },
        "models": results,
        "best_model": best_name,
        **ev["operating_point"],
        "business_cost": {
            **{k: cost[k] for k in ("chargeback_fee", "cost_false_positive",
                                    "test_cost_at_threshold",
                                    "test_cost_at_default_0_5",
                                    "test_cost_without_model",
                                    "test_savings_vs_no_model")},
            "threshold_method": threshold_method,
            "threshold_tuned_on_folds": threshold_folds,
            "target_alert_rate": round(target_rate, 5),
            **{k: cost[k] for k in ("threshold_cv_scale",
                                    "test_cost_at_cv_scale_threshold",
                                    "test_oracle_threshold",
                                    "test_cost_at_oracle", "threshold_regret")},
        },
        "confidence_intervals_95": ev["confidence_intervals_95"],
    }

    # Explainability is non-negotiable in risk: permutation importance
    # measures each engineered feature's real contribution on unseen data.
    X_test_features = best_model.named_steps["features"].transform(X_test)
    perm = permutation_importance(
        best_model[1:], X_test_features, y_test, n_repeats=5,
        random_state=random_state, scoring="average_precision", n_jobs=-1,
    )
    importances = sorted(zip(FEATURES, perm.importances_mean, strict=True),
                         key=lambda t: t[1], reverse=True)
    summary.update(breakdowns(test, proba, flagged))
    summary["episodes"] = episode_metrics(test, flagged)
    ep = summary["episodes"]
    log.info("Attacks detected: %.1f%% of %d | median %s fraud txns before "
             "first alert | %.1f%% of fraud value lost before detection",
             ep["detection_rate"] * 100, ep["episodes"],
             ep["median_txns_before_alert"],
             ep["value_lost_before_detection"] * 100)
    summary["rules_baseline"] = compare_with_rules(test, proba)
    rb = summary["rules_baseline"]
    log.info("Rule engine: %s alerts, recall %.1f%% | model at same volume: "
             "recall %.1f%%", f"{rb['alerts']:,}", rb["rules"]["recall"] * 100,
             rb["model_same_alerts"]["recall"] * 100)
    log.info("Recall by typology: %s", {k: v["recall"] for k, v in
                                        summary["recall_by_typology"].items()})
    summary["feature_importance"] = [
        {"feature": f, "importance": round(float(v), 4)} for f, v in importances
    ]
    log.info("\nTop fraud signals (permutation importance, drop in PR-AUC):")
    for f, v in importances[:8]:
        log.info(f"  {f:>26}: {v:.4f}")

    # reason codes need a "typical transaction" to compare each alert with
    reference = reference_row(best_model.named_steps["features"]
                              .transform(X_train))
    alerts = sample_alerts(test, proba, best_model, reference)

    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        alerts.to_csv(reports_dir / "sample_alerts.csv", index=False)

    return {
        "summary": summary, "model": best_model, "threshold": threshold,
        "reference": reference, "sample_alerts": alerts,
        "threshold_cv": threshold_cv,
        "train": train.assign(oof_proba=oof[best_name],
                              threshold_fold=recent),
        "test": test.assign(proba=proba, flagged=flagged.astype(int)),
        "y_test": y_test, "proba": proba,
    }


def sample_alerts(test: pd.DataFrame, proba, model, reference,
                  n: int = 15) -> pd.DataFrame:
    """The ``n`` highest-scored test transactions with their reason codes."""
    top = test.assign(score=proba).nlargest(n, "score")
    reasons = reason_codes(model, top[RAW_COLUMNS], reference)
    return pd.DataFrame({
        "transaction_id": top["transaction_id"],
        "timestamp": top["timestamp"],
        "customer_id": top["customer_id"],
        "amount": top["amount"],
        "merchant_category": top["merchant_category"],
        "channel": top["channel"],
        "score": top["score"].round(3),
        "reasons": [describe(r) for r in reasons],
        "fraud_type": top["fraud_type"],
    })


def breakdowns(test: pd.DataFrame, proba, flagged) -> dict:
    """Where the model works and where it doesn't, on the test period.

    Uses the ground-truth ``fraud_type`` (reported or not), which a real
    bank would approximate with investigator case outcomes.
    """
    flagged = np.asarray(flagged, bool)
    by_type = {}
    for ftype in FRAUD_TYPES:
        rows = (test["fraud_type"] == ftype).to_numpy()
        if rows.any():
            by_type[ftype] = {
                "transactions": int(rows.sum()),
                "recall": round(float(flagged[rows].mean()), 3),
                "value_recall": round(value_recall(
                    rows, flagged, test["amount"]), 3),
            }

    hidden = ((test["fraud_type"] != "none") & (test["is_fraud"] == 0))\
        .to_numpy()
    month = test["timestamp"].dt.to_period("M").astype(str).to_numpy()
    by_month = {}
    for m in np.unique(month):
        rows = month == m
        y = test["is_fraud"].to_numpy()[rows]
        has_fraud = bool(y.any())   # PR-AUC and recall are undefined otherwise
        by_month[m] = {
            "pr_auc": round(float(average_precision_score(y, proba[rows])), 4)
            if has_fraud else None,
            "recall": round(float(flagged[rows][y == 1].mean()), 3)
            if has_fraud else None,
            "fraud_rate": round(float(y.mean()), 5),
        }
    return {
        "recall_by_typology": by_type,
        "unreported_fraud": {
            "transactions": int(hidden.sum()),
            "flagged": int(flagged[hidden].sum()),
        },
        "performance_by_month": by_month,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_and_evaluate()
