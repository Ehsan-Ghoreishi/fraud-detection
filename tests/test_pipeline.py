import json

import pandas as pd
import pytest

from fraud_detection.cli import main as cli
from fraud_detection.features import FEATURES
from fraud_detection.model import train_and_evaluate
from fraud_detection.scoring import load_model, save_model, score


@pytest.fixture(scope="module")
def trained(small_log, tmp_path_factory):
    reports = tmp_path_factory.mktemp("reports")
    out = train_and_evaluate(data=small_log, reports_dir=reports, cv_folds=3,
                             test_start=None,
                             models=["logistic_regression", "gradient_boosting"])
    return out, reports


def test_metrics_file_is_complete(trained):
    out, reports = trained
    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics == out["summary"]
    assert metrics["best_model"] in metrics["models"]
    best = metrics["models"][metrics["best_model"]]
    assert {"test_pr_auc", "cv_pr_auc"} <= best.keys()
    assert sum(metrics["confusion_matrix"].values()) == len(out["y_test"])
    assert {d["feature"] for d in metrics["feature_importance"]} == set(FEATURES)


def test_breakdowns_cover_typologies_and_months(trained):
    out, _ = trained
    summary, test = out["summary"], out["test"]
    by_type = summary["recall_by_typology"]
    assert set(by_type) == set(test.fraud_type) - {"none"}
    assert all(0 <= v["recall"] <= 1 for v in by_type.values())
    months = set(test.timestamp.dt.to_period("M").astype(str))
    assert set(summary["performance_by_month"]) == months
    hidden = summary["unreported_fraud"]
    assert 0 <= hidden["flagged"] <= hidden["transactions"]


def test_threshold_regret_is_measured(trained):
    cost = trained[0]["summary"]["business_cost"]
    # the hindsight-optimal threshold can never be beaten on its own data
    assert cost["test_cost_at_oracle"] <= cost["test_cost_at_threshold"]
    assert cost["threshold_regret"] >= 0


def test_episode_metrics_are_reported(trained):
    out, _ = trained
    ep = out["summary"]["episodes"]
    assert ep["episodes"] == out["test"].query("fraud_episode >= 0")\
        .fraud_episode.nunique()
    assert 0 <= ep["detected_on_first_txn"] <= ep["detection_rate"] <= 1
    assert 0 <= ep["value_lost_before_detection"] <= 1


def test_rules_baseline_uses_identical_workload(trained):
    rb = trained[0]["summary"]["rules_baseline"]
    assert rb["alerts"] > 0
    assert set(rb["rules"]) == set(rb["model_same_alerts"]) == {"recall",
                                                                "precision"}


def test_model_beats_random_ranking(trained):
    out, _ = trained
    best = out["summary"]["models"][out["summary"]["best_model"]]
    assert best["test_pr_auc"] > 5 * out["y_test"].mean()


def test_evaluation_is_out_of_time(trained):
    out, _ = trained
    train, test = out["train"], out["test"]
    delay = pd.Timedelta(days=out["summary"]["data"]["label_delay_days"])
    assert train.timestamp.max() < test.timestamp.min() - delay
    assert not set(train.transaction_id) & set(test.transaction_id)
    # forward chaining: the earliest chunk has no out-of-fold prediction
    assert train.oof_proba.isna().iloc[0]
    assert train.oof_proba.notna().iloc[-1]


def test_save_load_and_score_roundtrip(trained, small_log, tmp_path):
    out, _ = trained
    path = save_model(out["model"], out["threshold"], path=tmp_path / "m.joblib",
                      reference=out["reference"])
    bundle = load_model(path)
    scored = score(small_log, bundle)
    assert len(scored) == len(small_log)
    assert (scored.reasons[scored.flag == 1] != "").all()
    assert (scored.reasons[scored.flag == 0] == "").all()
    assert scored.fraud_probability.between(0, 1).all()
    expected = (scored.fraud_probability >= bundle["threshold"]).astype(int)
    # probabilities are rounded for output, so allow boundary disagreements
    assert (scored.flag == expected).mean() > 0.95


def test_scores_match_training_pipeline(trained, small_log):
    out, _ = trained
    bundle = {"pipeline": out["model"], "threshold": out["threshold"]}
    scored = score(small_log, bundle).set_index("transaction_id")
    test = out["test"].set_index("transaction_id")
    assert ((scored.loc[test.index, "fraud_probability"] - test.proba).abs()
            < 1e-4).all()


def test_score_rejects_missing_columns(trained, small_log):
    out, _ = trained
    bundle = {"pipeline": out["model"], "threshold": out["threshold"]}
    with pytest.raises(ValueError, match="merchant_risk"):
        score(small_log.drop(columns="merchant_risk"), bundle)


def test_cli_end_to_end(tmp_path):
    data, reports, model = tmp_path / "t.csv", tmp_path / "r", tmp_path / "m.joblib"
    cli(["-q", "generate", "--customers", "300", "--days", "60",
         "--fraud-episode-rate", "0.1", "--data", str(data)])
    cli(["-q", "train", "--data", str(data), "--reports", str(reports),
         "--model", str(model), "--cv-folds", "3", "--test-start", "2026-02-10",
         "--models", "logistic_regression"])
    cli(["-q", "score", str(data), "-o", str(tmp_path / "s.csv"),
         "--model", str(model)])
    assert {p.name for p in reports.iterdir()} == {
        "metrics.json", "sample_alerts.csv", "pr_curve.png", "confusion_matrix.png",
        "cost_curve.png", "feature_importance.png", "calibration.png",
        "recall_by_typology.png",
    }
    assert (tmp_path / "s.csv").stat().st_size > 0


def test_sample_alerts_explain_themselves(trained):
    out, reports = trained
    alerts = out["sample_alerts"]
    assert alerts.score.is_monotonic_decreasing
    assert (alerts.reasons.str.len() > 0).all()
    assert (reports / "sample_alerts.csv").exists()
