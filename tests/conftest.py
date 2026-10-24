import pytest

from fraud_detection.data import generate_transactions


@pytest.fixture(scope="session")
def small_df():
    """A small but non-trivial dataset that keeps the suite fast."""
    return generate_transactions(n=4_000, fraud_rate=0.05, seed=0)
