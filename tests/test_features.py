import numpy as np
import pandas as pd

from fraud_detection.features import (
    BEHAVIOUR_COLUMNS,
    FEATURES,
    RAW_COLUMNS,
    FraudFeatures,
    add_behavioural_features,
)


def _enriched(small_log):
    return add_behavioural_features(small_log)


def test_transformer_output_columns(small_log):
    X = _enriched(small_log)[RAW_COLUMNS]
    out = FraudFeatures().fit_transform(X)
    assert list(out.columns) == FEATURES
    assert list(FraudFeatures().get_feature_names_out()) == FEATURES
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    # exactly one merchant-category indicator is set per row
    assert (out.filter(like="cat_").sum(axis=1) == 1).all()


def test_transformer_learns_category_medians_from_fit_data_only(small_log):
    X = _enriched(small_log)[RAW_COLUMNS]
    train, test = X.iloc[:1000], X.iloc[1000:].assign(amount=lambda d: d.amount * 10)
    ft = FraudFeatures().fit(train)
    expected = train.groupby("merchant_category").amount.median().to_dict()
    assert ft.category_median_ == expected
    # inflating test amounts shifts their ratio to the *training* median;
    # a median re-learned on the test rows would hide the shift
    assert ft.transform(test).log_amount_vs_category.median() > 1.5


def test_transformer_derives_time_and_channel_flags():
    X = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 03:00", "2026-01-01 14:00"]),
        "amount": [10.0, 10.0], "merchant_category": ["grocery", "travel"],
        "merchant_risk": [0.1, 0.2], "channel": ["in_store", "online"],
        "foreign_country": [0, 1], "distance_from_home": [1.0, 900.0],
        **{c: [0.0, 0.0] for c in BEHAVIOUR_COLUMNS},
    }).assign(amount_vs_customer_median=1.0)
    out = FraudFeatures().fit_transform(X)
    assert out.hour.tolist() == [3, 14]
    assert out.is_night.tolist() == [1, 0]
    assert out.is_online.tolist() == [0, 1]
    assert out.cat_travel.tolist() == [0, 1]


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
