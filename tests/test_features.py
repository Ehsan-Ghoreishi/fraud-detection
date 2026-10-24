import numpy as np
import pandas as pd

from fraud_detection.features import (
    FEATURES,
    RAW_COLUMNS,
    FraudFeatures,
    engineer_features,
)


def _txn(**overrides):
    row = {"amount": 20.0, "hour": 12, "days_since_last_txn": 1.0,
           "txn_count_1h": 1, "new_device": 0, "foreign_country": 0,
           "distance_from_home": 5.0, "merchant_risk": 0.1}
    row.update(overrides)
    return row


def test_engineered_flags():
    df = pd.DataFrame([
        _txn(),
        _txn(hour=3, txn_count_1h=4, distance_from_home=250.0),
        _txn(amount=900.0, new_device=1, foreign_country=1),
    ])
    out = engineer_features(df, high_amount_log=np.log1p(100))
    assert out.is_night.tolist() == [0, 1, 0]
    assert out.high_velocity.tolist() == [0, 1, 0]
    assert out.geo_risk.tolist() == [0, 1, 1]
    assert out.device_geo_amount_risk.tolist() == [0, 0, 1]
    assert np.allclose(out.log_amount, np.log1p(df.amount))


def test_does_not_mutate_input():
    df = pd.DataFrame([_txn()])
    engineer_features(df)
    assert list(df.columns) == RAW_COLUMNS


def test_transformer_learns_cutoff_from_fit_data_only():
    train = pd.DataFrame([_txn(amount=a) for a in (10.0, 20.0, 30.0)])
    test = pd.DataFrame([_txn(amount=a, new_device=1, foreign_country=1)
                         for a in (1_000.0, 2_000.0, 3_000.0)])
    ft = FraudFeatures().fit(train)
    assert ft.high_amount_log_ == np.log1p(20.0)
    # every test amount is above the *training* median, so all are flagged;
    # a cut-off leaked from the test rows would flag only one
    assert ft.transform(test).device_geo_amount_risk.tolist() == [1, 1, 1]


def test_transformer_output_columns(small_df):
    out = FraudFeatures().fit_transform(small_df[RAW_COLUMNS])
    assert list(out.columns) == FEATURES
    assert list(FraudFeatures().get_feature_names_out()) == FEATURES
