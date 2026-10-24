import json

import pytest

from fraud_detection.cli import main as cli
from fraud_detection.features import FEATURES
from fraud_detection.model import train_and_evaluate
from fraud_detection.scoring import load_model, save_model, score


@pytest.fixture(scope="module")
def trained(small_df, tmp_path_factory):
    reports = tmp_path_factory.mktemp("reports")
    out = train_and_evaluate(data=small_df, reports_dir=reports, cv_folds=3)
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


def test_model_beats_random_ranking(trained):
    out, _ = trained
    best = out["summary"]["models"][out["summary"]["best_model"]]
    assert best["test_pr_auc"] > 5 * out["y_test"].mean()


def test_threshold_is_tuned_on_training_data(trained):
    out, _ = trained
    assert len(out["oof_proba"]) == len(out["y_train"])
    assert not set(out["y_train"].index) & set(out["y_test"].index)


def test_save_load_and_score_roundtrip(trained, small_df, tmp_path):
    out, _ = trained
    path = save_model(out["model"], out["threshold"], path=tmp_path / "m.joblib")
    bundle = load_model(path)
    scored = score(small_df.head(50), bundle)
    assert scored.fraud_probability.between(0, 1).all()
    expected = (scored.fraud_probability >= bundle["threshold"]).astype(int)
    # probabilities are rounded for output, so allow boundary disagreements
    assert (scored.flag == expected).mean() > 0.95


def test_score_rejects_missing_columns(trained, small_df):
    out, _ = trained
    bundle = {"pipeline": out["model"], "threshold": out["threshold"]}
    with pytest.raises(ValueError, match="merchant_risk"):
        score(small_df.drop(columns="merchant_risk"), bundle)


def test_cli_end_to_end(tmp_path):
    data, reports, model = tmp_path / "t.csv", tmp_path / "r", tmp_path / "m.joblib"
    cli(["-q", "generate", "--n", "3000", "--fraud-rate", "0.05",
         "--data", str(data)])
    cli(["-q", "train", "--data", str(data), "--reports", str(reports),
         "--model", str(model), "--cv-folds", "3"])
    cli(["-q", "score", str(data), "-o", str(tmp_path / "s.csv"),
         "--model", str(model)])
    assert {p.name for p in reports.iterdir()} == {
        "metrics.json", "pr_curve.png", "confusion_matrix.png",
        "cost_curve.png", "feature_importance.png",
    }
    assert (tmp_path / "s.csv").stat().st_size > 0
