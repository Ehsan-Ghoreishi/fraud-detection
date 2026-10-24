import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import BEHAVIOUR_COLUMNS, add_behavioural_features
from fraud_detection.ieee.data import REFERENCE_DATE, load_raw, to_log
from fraud_detection.ieee.features import IEEEFeatures, feature_families
from fraud_detection.ieee.pipeline import train_and_evaluate_ieee
from ieee_fixture import make_ieee_like, split_tables


@pytest.fixture(scope="module")
def raw():
    return make_ieee_like()


def test_load_raw_joins_tables_and_caches(raw, tmp_path):
    txn, ident = split_tables(raw)
    txn.to_csv(tmp_path / "train_transaction.csv", index=False)
    ident.to_csv(tmp_path / "train_identity.csv", index=False)
    df = load_raw(tmp_path)
    assert len(df) == len(txn)                     # left join keeps all
    assert df.DeviceInfo.notna().sum() == len(ident)
    assert df["V1"].dtype == np.float32
    assert (tmp_path / "train_joined.pkl").exists()


def test_load_raw_explains_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="ieee-download"):
        load_raw(tmp_path)


def test_to_log_reconstructs_clients_and_time(raw):
    log = to_log(raw)
    assert log.timestamp.min() >= REFERENCE_DATE
    assert log.timestamp.is_monotonic_increasing
    # client = card1 + addr1 + account start day (day - D1)
    row = log.iloc[0]
    start = int(row.TransactionDT // 86_400 - row.D1)
    addr = -1 if pd.isna(row.addr1) else int(row.addr1)
    assert row.customer_id == f"{row.card1}_{addr}_{start}"
    assert set(log.channel) == {"online", "online_unidentified"}
    assert (log.device_id[log.channel == "online_unidentified"] == "").all()
    fraud = log[log.is_fraud == 1]
    assert (fraud.groupby("fraud_episode").customer_id.nunique() == 1).all()
    assert (log.fraud_episode[log.is_fraud == 0] == -1).all()


def test_ieee_features_learn_encodings_on_fit_data_only(raw):
    enriched = add_behavioural_features(to_log(raw))
    train, test = enriched.iloc[:3000], enriched.iloc[3000:].copy()
    test.loc[test.index[:5], "P_emaildomain"] = "never-seen.example"
    ft = IEEEFeatures().fit(train)
    assert "P_emaildomain" in ft.categorical_ and "C1" in ft.numeric_
    out = ft.transform(test)
    assert (out["P_emaildomain__code"].iloc[:5] == -1).all()
    assert (out["P_emaildomain__freq"].iloc[:5] == 0).all()
    assert out.dtypes.eq(np.float32).all()
    assert {f"bhv_{c}" for c in BEHAVIOUR_COLUMNS} <= set(out.columns)
    assert not any(c.startswith("bhv_")
                   for c in IEEEFeatures(behaviour=False).fit(train)
                   .get_feature_names_out())
    # identifiers and labels never become features
    assert not {"isFraud", "TransactionID", "TransactionDT", "customer_id",
                "fraud_episode"} & set(out.columns)


def test_feature_families_cover_every_column(raw):
    names = IEEEFeatures().fit(add_behavioural_features(to_log(raw)))\
        .get_feature_names_out()
    families = feature_families(names)
    assert sum(len(v) for v in families.values()) == len(names)
    assert "V1" in families["Vesta engineered (V)"]
    assert "P_emaildomain__code" in families["email domain"]
    assert "bhv_new_device" in families["customer behaviour (ours)"]


def test_pipeline_end_to_end_on_ieee_schema(raw, tmp_path):
    out = train_and_evaluate_ieee(
        data=raw, reports_dir=tmp_path, test_start=None, cv_folds=3,
        label_delay_days=5, importance_sample=2_000,
        models=["logistic_regression", "gradient_boosting_vesta_only",
                "gradient_boosting"])
    s = json.loads((tmp_path / "metrics.json").read_text())
    assert s == out["summary"]
    assert set(s["models"]) == {"logistic_regression", "gradient_boosting",
                                "gradient_boosting_vesta_only"}
    best = s["models"][s["best_model"]]
    assert best["test_pr_auc"] > 2 * s["data"]["test_fraud_rate"]
    assert s["business_cost"]["threshold_regret"] >= 0
    assert s["unseen_clients"]["transactions"] > 0
    assert {d["feature"] for d in s["feature_family_importance"]} >= {
        "customer behaviour (ours)", "Vesta engineered (V)"}
    assert (tmp_path / "sample_alerts.csv").exists()
    assert out["train"].timestamp.max() < out["test"].timestamp.min()


def test_ieee_figures_are_written(raw, tmp_path):
    from fraud_detection.plots import make_ieee

    out = train_and_evaluate_ieee(
        data=raw, reports_dir=None, test_start=None, cv_folds=3,
        label_delay_days=5, importance_sample=1_000,
        models=["gradient_boosting"])
    make_ieee(out, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "pr_curve.png", "confusion_matrix.png", "cost_curve.png",
        "feature_family_importance.png", "calibration.png",
        "recall_by_product.png"}
