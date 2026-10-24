import json

import numpy as np
import pandas as pd

from fraud_detection.ulb import (
    PCA,
    REFERENCE_DATE,
    ULBFeatures,
    to_log,
    train_and_evaluate_ulb,
)


def _ulb_like(n=6_000, seed=0):
    """ULB schema: Time (s over 48h), V1-V28, Amount, Class (as OpenML)."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.02
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=PCA)
    df[["V14", "V4"]] += np.c_[-3 * y, 2 * y]          # where the signal is
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_792, n)))
    df["Amount"] = rng.lognormal(3, 1.2, n).round(2)
    df["Class"] = pd.Categorical(np.where(y, "1", "0"))
    return df


def test_to_log_maps_time_and_labels():
    log = to_log(_ulb_like(200))
    assert log.timestamp.min() >= REFERENCE_DATE
    assert log.timestamp.max() < REFERENCE_DATE + pd.Timedelta(days=2)
    assert set(log.is_fraud.unique()) <= {0, 1}
    assert log.transaction_id.is_unique


def test_features_are_pca_amount_and_hour():
    out = ULBFeatures().fit_transform(to_log(_ulb_like(100)))
    assert list(out.columns) == [*PCA, "log_amount", "hour"]
    assert out.hour.between(0, 23).all()


def test_pipeline_end_to_end_on_ulb_schema(tmp_path):
    out = train_and_evaluate_ulb(data=to_log(_ulb_like()), reports_dir=tmp_path,
                                 cv_folds=3)
    s = json.loads((tmp_path / "metrics.json").read_text())
    assert s == out["summary"]
    assert s["models"][s["best_model"]]["test_pr_auc"] > 0.5
    assert s["business_cost"]["threshold_regret"] >= 0
    assert s["feature_importance"][0]["feature"] in {"V14", "V4"}
