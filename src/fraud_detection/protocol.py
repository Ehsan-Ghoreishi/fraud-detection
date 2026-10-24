"""The evaluation protocol, as reusable steps shared by every data track.

1. :func:`split_with_embargo` — out-of-time test period, plus a label-delay
   embargo: the last days before deployment are visible but unlabelled.
2. :func:`select_model` — forward-chaining CV with the same embargo as a gap;
   best out-of-fold PR-AUC wins.
3. :func:`tune_alert_rate` — cost-optimal alert rate on the most recent folds.
4. :func:`deploy_threshold` — that alert rate re-expressed on the final
   model's scores for the embargo window.
5. :func:`evaluate_test` — score the test period once: ranking, cost, regret
   versus a hindsight threshold, customer-clustered confidence intervals.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from fraud_detection.evaluation import (
    bootstrap_ci,
    choose_threshold,
    cost_curve,
    rate_matched_threshold,
    value_recall,
)
from fraud_detection.validation import (
    apply_label_delay,
    forward_chaining_oof,
    out_of_time_split,
)

log = logging.getLogger(__name__)


def _span_days(df: pd.DataFrame) -> float:
    """Covered period in (fractional) days, at least one hour."""
    span = df["timestamp"].max() - df["timestamp"].min()
    return max(span.total_seconds() / 86_400, 1 / 24)


def split_with_embargo(enriched: pd.DataFrame, test_start,
                       label_delay_days: float):
    """Return ``(train, embargo, test, cut)`` from a time-sorted frame.

    ``train`` holds rows whose labels are known at deployment (``cut``);
    ``embargo`` is the unlabelled window just before it.
    """
    before, test, cut = out_of_time_split(enriched, test_start)
    train = apply_label_delay(before, cut, label_delay_days)
    embargo = before.loc[before.index.difference(train.index)]
    log.info("Train: %s txns to %s | %g-day label-delay embargo | "
             "Test: %s txns from %s (out of time)", f"{len(train):,}",
             train["timestamp"].max().date(), label_delay_days,
             f"{len(test):,}", cut.date())
    return train, embargo, test, cut


def cv_gap_rows(train: pd.DataFrame, label_delay_days: float,
                cv_folds: int) -> int:
    """The label-delay embargo expressed in rows, for forward-chaining CV."""
    gap = int(round(label_delay_days * len(train) / _span_days(train)))
    # on short histories a full embargo leaves no room for the folds; keep
    # at least 5% of the rows for the first (earliest) training window
    n = len(train)
    max_gap = n - cv_folds * (n // (cv_folds + 1)) - int(0.05 * n)
    if gap > max_gap:
        log.warning("CV label-delay gap capped at %d rows (wanted %d): "
                    "training history is short", max_gap, gap)
        gap = max_gap
    return gap


def select_model(candidates: dict, X_train, y_train, cv_folds: int,
                 gap: int):
    """Forward-chaining CV for every candidate; best out-of-fold PR-AUC wins.

    Returns ``(best_name, results, oof, folds)`` where ``oof[name]`` and
    ``folds[name]`` are per-row out-of-fold scores and fold indices.
    """
    results, oof, folds = {}, {}, {}
    for name, model in candidates.items():
        oof[name], folds[name] = forward_chaining_oof(
            model, X_train, y_train, cv_folds, gap=gap, return_folds=True)
        seen = ~np.isnan(oof[name])
        pr_auc = average_precision_score(y_train[seen], oof[name][seen])
        roc_auc = roc_auc_score(y_train[seen], oof[name][seen])
        results[name] = {"cv_pr_auc": round(pr_auc, 4),
                         "cv_roc_auc": round(roc_auc, 4)}
        log.info(f"{name:>22} | CV PR-AUC {pr_auc:.4f} | CV ROC-AUC {roc_auc:.4f}")
    best_name = max(results, key=lambda k: results[k]["cv_pr_auc"])
    log.info(f"\nBest model by CV PR-AUC: {best_name}")
    return best_name, results, oof, folds


def tune_alert_rate(y_train, oof, folds, fn_cost, cost_fp: float,
                    cv_folds: int, threshold_folds: int):
    """Cost-optimal threshold and alert rate on the most recent CV folds.

    Fold models are trained on less (and older) history than the final
    model, so their scores sit on a different scale and a raw score
    threshold does not transfer; the alert rate does. Returns
    ``(threshold_cv, target_rate, recent_mask)``.
    """
    y_train = np.asarray(y_train)
    recent = folds >= cv_folds - threshold_folds
    if not y_train[recent].any():           # too little fraud: use all folds
        recent = folds >= 0
    threshold_cv, _ = choose_threshold(y_train[recent], oof[recent],
                                       cost_fn=np.asarray(fn_cost)[recent],
                                       cost_fp=cost_fp)
    target_rate = float((oof[recent] >= threshold_cv).mean())
    return threshold_cv, target_rate, recent


def deploy_threshold(model, X_embargo, target_rate: float,
                     threshold_cv: float, threshold_folds: int):
    """Place the cut-off at ``target_rate`` on the embargo window's scores.

    Those transactions are already seen at deployment time but unlabelled,
    so the rule uses only information available then. Falls back to the
    CV-scale threshold when there is no embargo window.
    """
    if len(X_embargo) and 0 < target_rate < 1:
        threshold = rate_matched_threshold(
            model.predict_proba(X_embargo)[:, 1], target_rate)
        method = "alert_rate_transfer"
    else:
        threshold, method = threshold_cv, "cv_score"
    log.info(f"Threshold {threshold:.4f} ({method}): cost-optimal alert rate "
             f"{target_rate:.2%} from the last {threshold_folds} folds "
             f"(CV-scale threshold {threshold_cv:.4f})")
    return threshold, method


def evaluate_test(test: pd.DataFrame, proba, threshold: float,
                  threshold_cv: float, chargeback_fee: float, cost_fp: float,
                  groups, seed: int) -> dict:
    """Score the out-of-time test period once.

    ``test`` needs ``is_fraud``, ``amount`` and ``timestamp``. Returns the
    test metrics, operating point, business cost (including regret versus
    the hindsight-optimal threshold) and clustered confidence intervals.
    """
    y_test = test["is_fraud"]
    flagged = proba >= threshold
    fn_cost = test["amount"].to_numpy() + chargeback_fee

    ranking = {
        "test_pr_auc": round(average_precision_score(y_test, proba), 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "test_brier": round(brier_score_loss(y_test, proba), 5),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, flagged).ravel()
    test_days = _span_days(test)
    recall, precision = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    caught_value = value_recall(y_test, flagged, test["amount"])
    log.info(f"Test PR-AUC {ranking['test_pr_auc']:.3f} | "
             f"recall {recall:.1%} | precision {precision:.1%} | "
             f"fraud value caught {caught_value:.1%} | "
             f"{(tp + fp) / test_days:.0f} alerts/day")

    costs = cost_curve(y_test, proba, fn_cost, cost_fp,
                       thresholds=[threshold, 0.5])["cost"]
    cost_at_threshold, cost_at_default = map(float, costs)
    no_model_cost = float(fn_cost[y_test.to_numpy() == 1].sum())
    # regret: how much a perfectly hindsight-tuned threshold would have saved
    oracle_threshold, oracle_table = choose_threshold(y_test, proba, fn_cost,
                                                      cost_fp)
    oracle_cost = float(oracle_table["cost"].min())
    cv_scale_cost = float(cost_curve(y_test, proba, fn_cost, cost_fp,
                                     thresholds=[threshold_cv])["cost"][0])
    log.info(f"Test cost €{cost_at_threshold:,.0f} at {threshold:.3f} "
             f"(€{cost_at_default:,.0f} at 0.50, €{no_model_cost:,.0f} "
             f"with no model) | regret vs hindsight-optimal {oracle_threshold:.3f}: "
             f"€{cost_at_threshold - oracle_cost:,.0f}")

    # resample whole customers: a fraud burst on one card is one event
    ci = bootstrap_ci(y_test, proba, threshold, fn_cost, cost_fp, seed=seed,
                      groups=groups)
    log.info("95%% CI  PR-AUC %s | recall %s | precision %s",
             ci["pr_auc"], ci["recall"], ci["precision"])

    return {
        "ranking": ranking,
        "operating_point": {
            "cost_optimal_threshold": round(threshold, 4),
            "recall_fraud": round(recall, 3),
            "precision_fraud": round(precision, 3),
            "value_recall": round(caught_value, 3),
            "flagged_rate": round(float(flagged.mean()), 4),
            "alerts_per_day": round((tp + fp) / test_days, 1),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                                 "fn": int(fn), "tp": int(tp)},
        },
        "business_cost": {
            "chargeback_fee": chargeback_fee,
            "cost_false_positive": cost_fp,
            "test_cost_at_threshold": cost_at_threshold,
            "test_cost_at_default_0_5": cost_at_default,
            "test_cost_without_model": no_model_cost,
            "test_savings_vs_no_model": no_model_cost - cost_at_threshold,
            "threshold_cv_scale": threshold_cv,
            "test_cost_at_cv_scale_threshold": cv_scale_cost,
            "test_oracle_threshold": oracle_threshold,
            "test_cost_at_oracle": oracle_cost,
            "threshold_regret": cost_at_threshold - oracle_cost,
        },
        "confidence_intervals_95": ci,
        "flagged": flagged,
    }
