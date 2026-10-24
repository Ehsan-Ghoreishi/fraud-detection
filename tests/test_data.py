import pandas as pd
import pytest

from fraud_detection.data import generate_transactions
from fraud_detection.features import RAW_COLUMNS


def test_schema_and_size():
    df = generate_transactions(n=1_000, seed=1)
    assert len(df) == 1_000
    assert set(RAW_COLUMNS + ["is_fraud"]) == set(df.columns)
    assert df.notna().all().all()


def test_fraud_rate_is_respected():
    df = generate_transactions(n=10_000, fraud_rate=0.02, seed=1)
    assert df.is_fraud.sum() == 200


def test_is_deterministic_for_a_seed():
    pd.testing.assert_frame_equal(generate_transactions(n=500, seed=7),
                                  generate_transactions(n=500, seed=7))


def test_value_ranges():
    df = generate_transactions(n=5_000, seed=2)
    assert df.amount.min() >= 0.5
    assert df.hour.between(0, 23).all()
    assert df.merchant_risk.between(0, 1).all()
    assert set(df.new_device.unique()) <= {0, 1}


def test_rows_are_shuffled():
    df = generate_transactions(n=2_000, fraud_rate=0.1, seed=3)
    # fraud rows are generated last; after shuffling they must not all be there
    assert df.is_fraud.iloc[:1_000].sum() > 0


@pytest.mark.parametrize("kwargs", [{"fraud_rate": 0}, {"fraud_rate": 1},
                                    {"fraud_rate": -0.1}, {"n": 1}])
def test_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_transactions(**kwargs)
