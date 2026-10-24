import pandas as pd
import pytest

from fraud_detection.data import LOG_COLUMNS, generate, load_transactions
from fraud_detection.simulation import SimulationConfig


def test_generate_and_load_roundtrip(tmp_path):
    path = tmp_path / "log.csv"
    written = generate(path, SimulationConfig(n_customers=80, days=20), seed=1)
    loaded = load_transactions(path)
    assert pd.api.types.is_datetime64_any_dtype(loaded.timestamp)
    # in-store rows have no device: must come back as "", not NaN
    assert loaded.device_id.notna().all()
    pd.testing.assert_frame_equal(loaded[LOG_COLUMNS], written[LOG_COLUMNS],
                                  check_dtype=False)


def test_load_rejects_incomplete_log(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2026-01-01"], "amount": [1.0]}).to_csv(path)
    with pytest.raises(ValueError, match="customer_id"):
        load_transactions(path)
