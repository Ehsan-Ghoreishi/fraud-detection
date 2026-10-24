import pytest

from fraud_detection.simulation import SimulationConfig, simulate_transactions

# small enough for a fast suite, large enough that every typology appears in
# both the training and the out-of-time test period
SMALL = SimulationConfig(n_customers=500, days=90, fraud_episode_rate=0.08,
                         drift_from_day=60)


@pytest.fixture(scope="session")
def small_log():
    return simulate_transactions(SMALL, seed=0)
