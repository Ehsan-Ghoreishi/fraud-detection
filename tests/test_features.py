import numpy as np
import pandas as pd

from fraud_detection.features import (
    BEHAVIOUR_COLUMNS,
    FEATURES,
    RAW_COLUMNS,
    FraudFeatures,
    add_behavioural_features,
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


# --------------------------------------------------------------------------- #
# Point-in-time behavioural features
# --------------------------------------------------------------------------- #
def _log(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "customer_id", "amount",
                                     "channel", "device_id",
                                     "merchant_category"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.insert(0, "transaction_id", range(len(df)))
    return df


def test_behavioural_features_on_a_known_history():
    log = _log([
        ("2026-01-01 10:00", 1, 20.0, "online", "c1-d0", "grocery"),
        ("2026-01-01 10:20", 1, 2.0, "online", "c1-d0", "digital_goods"),
        ("2026-01-01 10:30", 2, 50.0, "in_store", "", "fuel"),
        ("2026-01-01 10:40", 1, 3.0, "online", "fraud-1", "digital_goods"),
        ("2026-01-02 12:00", 1, 400.0, "online", "fraud-1", "electronics"),
    ])
    f = add_behavioural_features(log)
    c1 = f[f.customer_id == 1]
    assert c1.prior_txn_count.tolist() == [0, 1, 2, 3]
    assert c1.txn_count_1h.tolist() == [0, 1, 2, 0]
    assert c1.small_txn_count_1h.tolist() == [0, 0, 1, 0]
    assert c1.txn_count_24h.tolist() == [0, 1, 2, 0]   # next day is > 24h later
    assert c1.secs_since_last_txn.tolist()[1:] == [1200, 1200, 91200]   # 25h20m
    # fraud-1 was first seen 25h20m earlier, so it is no longer "new"
    assert c1.new_device.tolist() == [1, 1, 1, 0]
    assert c1.new_category.tolist() == [1, 1, 0, 1]
    # past median of [20, 2, 3] is 3 -> 400 / 3
    assert np.isclose(c1.amount_vs_customer_median.iloc[3], 400 / 3)
    assert c1.amount_vs_customer_median.iloc[0] == 1.0     # no history yet
    assert f.loc[f.customer_id == 2, "new_device"].item() == 0  # in-store


def test_behavioural_features_never_look_ahead():
    from fraud_detection.simulation import SimulationConfig, simulate_transactions

    log = simulate_transactions(SimulationConfig(n_customers=150, days=30),
                                seed=4)
    cutoff = log.timestamp.quantile(0.6)
    full = add_behavioural_features(log)
    past_only = add_behavioural_features(log[log.timestamp < cutoff])
    cols = BEHAVIOUR_COLUMNS
    pd.testing.assert_frame_equal(
        full.loc[full.timestamp < cutoff, cols].reset_index(drop=True),
        past_only[cols].reset_index(drop=True),
    )


def test_behavioural_features_preserve_row_order():
    log = _log([
        ("2026-01-01 12:00", 2, 5.0, "in_store", "", "fuel"),
        ("2026-01-01 09:00", 1, 9.0, "in_store", "", "fuel"),
    ])
    assert add_behavioural_features(log).transaction_id.tolist() == [0, 1]
