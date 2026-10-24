"""Train and evaluate on real IEEE-CIS transactions with the shared protocol.

Same protocol as the simulated track (:mod:`fraud_detection.protocol`):
out-of-time test period, label-delay embargo, forward-chaining CV,
cost-optimal alert rate transferred to the final model, customer-clustered
confidence intervals. On top of that, the questions real data raises:

* **Does our feature engineering help?** The ablation candidate is the same
  model without the point-in-time behaviour features.
* **Is it just recognising known accounts?** Labels in this data propagate
  across a compromised account, so a client defrauded in the training
  period keeps producing fraud labels in the test period. Performance is
  therefore also reported on clients never seen before the test period.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detection.config import (
    CHARGEBACK_FEE,
    COST_FALSE_POSITIVE,
    CV_FOLDS,
    LABEL_DELAY_DAYS,
    RANDOM_STATE,
    REPORTS,
    THRESHOLD_FOLDS,
)
from fraud_detection.evaluation import episode_metrics, value_recall
from fraud_detection.explain import (
    describe,
    grouped_permutation_importance,
    reason_codes,
    reference_row,
)
from fraud_detection.features import add_behavioural_features
from fraud_detection.ieee.data import IEEE_DIR, load_raw, to_log
from fraud_detection.ieee.features import FAMILY_DESCRIPTIONS, feature_families
from fraud_detection.ieee.models import build_ieee_models
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

log = logging.getLogger(__name__)

IEEE_REPORTS = REPORTS / "ieee"
IEEE_TEST_START = "2018-04-01"       # train Dec-Mar, test Apr-May


def train_and_evaluate_ieee(data: pd.DataFrame | None = None,
                            src: Path = IEEE_DIR,
                            reports_dir: Path | None = IEEE_REPORTS,
                            chargeback_fee: float = CHARGEBACK_FEE,
                            cost_fp: float = COST_FALSE_POSITIVE,
                            cv_folds: int = CV_FOLDS,
                            test_start: str | None = IEEE_TEST_START,
                            label_delay_days: float = LABEL_DELAY_DAYS,
                            threshold_folds: int = THRESHOLD_FOLDS,
                            models: list[str] | None = None,
                            importance_sample: int = 60_000,
                            random_state: int = RANDOM_STATE) -> dict:
    """Run the full protocol on IEEE-CIS. ``data`` takes a raw joined frame
    (as returned by :func:`load_raw`) instead of reading ``src``."""
    raw = load_raw(src) if data is None else data
    log.info("IEEE-CIS: %s transactions, %.2f%% fraud", f"{len(raw):,}",
             raw["isFraud"].mean() * 100)
    enriched = add_behavioural_features(to_log(raw)).sort_values(
        ["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    train, embargo, test, cut = split_with_embargo(enriched, test_start,
                                                   label_delay_days)
    y_train = train["is_fraud"]

    candidates = build_ieee_models(random_state)
    if models:
        candidates = {k: candidates[k] for k in models}
    best_name, results, oof, folds = select_model(
        candidates, train, y_train, cv_folds,
        gap=cv_gap_rows(train, label_delay_days, cv_folds))

    threshold_cv, target_rate, recent = tune_alert_rate(
        y_train, oof[best_name], folds[best_name],
        train["amount"].to_numpy() + chargeback_fee, cost_fp, cv_folds,
        threshold_folds)
    best_model = candidates[best_name].fit(train, y_train)
    threshold, threshold_method = deploy_threshold(
        best_model, embargo, target_rate, threshold_cv, threshold_folds)

    proba = best_model.predict_proba(test)[:, 1]
    ev = evaluate_test(test, proba, threshold, threshold_cv, chargeback_fee,
                       cost_fp, groups=test["customer_id"], seed=random_state)
    flagged = ev["flagged"]
    results[best_name].update(ev["ranking"])
    selection_diagnostics(candidates, best_name, train, y_train, test,
                          test["is_fraud"], results)
    if "gradient_boosting_vesta_only" in results and "gradient_boosting" in results:
        lift = (results["gradient_boosting"]["cv_pr_auc"]
                - results["gradient_boosting_vesta_only"]["cv_pr_auc"])
        log.info("Behaviour features add %+.4f CV PR-AUC over Vesta's columns",
                 lift)

    cost = ev["business_cost"]
    summary = {
        "data": {
            "source": "IEEE-CIS Fraud Detection (Vesta), labelled train files",
            "transactions": len(enriched),
            "clients": int(enriched["customer_id"].nunique()),
            "train_transactions": len(train), "test_transactions": len(test),
            "train_fraud_rate": round(float(y_train.mean()), 5),
            "test_fraud_rate": round(float(test["is_fraud"].mean()), 5),
            "train_end": str(train["timestamp"].max().date()),
            "label_delay_days": label_delay_days,
            "test_start": str(cut.date()),
            "test_end": str(test["timestamp"].max().date()),
        },
        "models": results,
        "best_model": best_name,
        **ev["operating_point"],
        "business_cost": {**cost, "threshold_method": threshold_method,
                          "threshold_tuned_on_folds": threshold_folds,
                          "target_alert_rate": round(target_rate, 5)},
        "confidence_intervals_95": ev["confidence_intervals_95"],
        "alert_budget": ev["alert_budget"],
        "unseen_clients": unseen_client_metrics(
            test, proba, flagged, known=set(train["customer_id"])
            | set(embargo["customer_id"])),
        "episodes": episode_metrics(test, flagged),
        "recall_by_product": recall_by_group(test, flagged, "ProductCD"),
        "performance_by_month": by_month(test, proba, flagged),
        "rules_baseline": compare_with_rules(test, proba),
    }
    ep, unseen, rb = (summary["episodes"], summary["unseen_clients"],
                      summary["rules_baseline"])
    log.info("Compromised clients detected: %.1f%% of %d | %.1f%% of fraud "
             "value lost before the first alert", ep["detection_rate"] * 100,
             ep["episodes"], ep["value_lost_before_detection"] * 100)
    log.info("Clients new in the test period: PR-AUC %s, recall %s (%s txns)",
             unseen["pr_auc"], unseen["recall"], f"{unseen['transactions']:,}")
    log.info("Rule engine: %s alerts, recall %.1f%% | model at same volume: "
             "recall %.1f%%", f"{rb['alerts']:,}", rb["rules"]["recall"] * 100,
             rb["model_same_alerts"]["recall"] * 100)

    # explainability by feature family: hundreds of correlated columns make
    # per-column importances meaningless
    features_step = best_model.named_steps["features"]
    families = feature_families(features_step.get_feature_names_out())
    sample = test.sample(min(importance_sample, len(test)),
                         random_state=random_state)
    importance = grouped_permutation_importance(
        best_model[1:], features_step.transform(sample), sample["is_fraud"],
        families, n_repeats=3, seed=random_state)
    summary["feature_family_importance"] = [
        {"feature": f, "importance": round(v, 4)} for f, v in importance.items()]
    log.info("Feature families by importance: %s",
             {k: round(v, 3) for k, v in list(importance.items())[:6]})

    reference = reference_row(features_step.transform(
        train.sample(min(50_000, len(train)), random_state=random_state)),
        one_hot_groups=())
    alerts = sample_alerts(test, proba, best_model, reference, families)

    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        alerts.to_csv(reports_dir / "sample_alerts.csv", index=False)

    return {
        "summary": summary, "model": best_model, "threshold": threshold,
        "threshold_cv": threshold_cv, "sample_alerts": alerts,
        "train": train.assign(oof_proba=oof[best_name], threshold_fold=recent),
        "test": test.assign(proba=proba, flagged=flagged.astype(int)),
        "y_test": test["is_fraud"], "proba": proba,
    }


def unseen_client_metrics(test, proba, flagged, known: set) -> dict:
    """Performance on clients with no transaction before the test period."""
    new = ~test["customer_id"].isin(known).to_numpy()
    y = test["is_fraud"].to_numpy()[new]
    if not y.any():
        return {"transactions": int(new.sum()), "pr_auc": None, "recall": None}
    return {
        "transactions": int(new.sum()),
        "fraud_rate": round(float(y.mean()), 5),
        "pr_auc": round(float(average_precision_score(y, proba[new])), 4),
        "recall": round(float(flagged[new][y == 1].mean()), 3),
        "precision": round(float(y[flagged[new]].mean()), 3)
        if flagged[new].any() else None,
    }


def recall_by_group(test, flagged, column: str) -> dict:
    """Transaction and value recall of labelled fraud per category."""
    out = {}
    y = test["is_fraud"].to_numpy() == 1
    for value in sorted(test[column].dropna().unique()):
        rows = (test[column] == value).to_numpy() & y
        if rows.any():
            out[str(value)] = {
                "transactions": int(rows.sum()),
                "recall": round(float(flagged[rows].mean()), 3),
                "value_recall": round(value_recall(rows, flagged,
                                                   test["amount"]), 3),
            }
    return out


def by_month(test, proba, flagged) -> dict:
    month = test["timestamp"].dt.to_period("M").astype(str).to_numpy()
    out = {}
    for m in np.unique(month):
        rows = month == m
        y = test["is_fraud"].to_numpy()[rows]
        out[m] = {
            "transactions": int(rows.sum()),
            "fraud_rate": round(float(y.mean()), 5),
            "pr_auc": round(float(average_precision_score(y, proba[rows])), 4)
            if y.any() else None,
            "recall": round(float(flagged[rows][y == 1].mean()), 3)
            if y.any() else None,
        }
    return out


def sample_alerts(test, proba, model, reference, families,
                  n: int = 15) -> pd.DataFrame:
    """Highest-scored test transactions with family-level reason codes."""
    top = test.assign(score=proba).nlargest(n, "score")
    reasons = reason_codes(model, top, reference, groups=families)
    return pd.DataFrame({
        "TransactionID": top["TransactionID"],
        "timestamp": top["timestamp"],
        "amount": top["amount"],
        "ProductCD": top["ProductCD"],
        "score": top["score"].round(3),
        "reasons": [describe(r, FAMILY_DESCRIPTIONS | {"other": "other"})
                    for r in reasons],
        "is_fraud": top["is_fraud"],
    })
