import pandas as pd
import pytest

from fraud_detection.simulation import (
    CATEGORIES,
    FRAUD_TYPES,
    SimulationConfig,
    simulate_transactions,
)

SMALL = SimulationConfig(n_customers=400, days=60, fraud_episode_rate=0.08,
                         drift_from_day=40)


@pytest.fixture(scope="module")
def sim():
    return simulate_transactions(SMALL, seed=0)


def test_log_is_time_ordered_with_unique_ids(sim):
    assert sim.timestamp.is_monotonic_increasing
    assert sim.transaction_id.is_unique
    assert sim.timestamp.min() >= pd.Timestamp(SMALL.start)
    assert sim.timestamp.max() < pd.Timestamp(SMALL.start) + pd.Timedelta(days=60)


def test_value_domains(sim):
    assert sim.amount.min() >= 0.5
    assert sim.merchant_risk.between(0, 1).all()
    assert set(sim.merchant_category) <= set(CATEGORIES)
    assert set(sim.fraud_type) <= {"none", *FRAUD_TYPES}
    assert set(sim.channel) == {"online", "in_store"}
    # only online transactions carry a device
    assert (sim.device_id[sim.channel == "in_store"] == "").all()
    assert (sim.device_id[sim.channel == "online"] != "").all()


def test_labels_are_noisy_but_never_invented(sim):
    # unreported fraud exists, but a legit transaction is never labelled fraud
    assert not ((sim.fraud_type == "none") & (sim.is_fraud == 1)).any()
    assert ((sim.fraud_type != "none") & (sim.is_fraud == 0)).any()


def test_every_typology_appears(sim):
    assert set(sim.fraud_type) - {"none"} == set(FRAUD_TYPES)


def test_typology_footprints(sim):
    by = sim[sim.fraud_type != "none"].groupby("fraud_type")
    assert 0 < by.foreign_country.mean()["cloned_card"] < 1
    assert by.distance_from_home.min()["cloned_card"] >= 30
    assert (by.channel.agg(lambda c: (c == "in_store").mean())["cloned_card"]
            == 1)
    assert by.foreign_country.mean()["stealth"] == 0
    assert (by.channel.agg(lambda c: (c == "online").mean())["card_testing"]
            == 1)
    assert by.amount.median()["card_testing"] < by.amount.median()["cloned_card"]


def test_legitimate_traffic_overlaps_with_fraud_signals(sim):
    legit = sim[sim.fraud_type == "none"]
    assert legit.foreign_country.mean() > 0       # travel, cross-border online
    assert (legit.distance_from_home > 400).any()


def test_deterministic_for_a_seed():
    a = simulate_transactions(SMALL, seed=3)
    b = simulate_transactions(SMALL, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_rejects_degenerate_config():
    with pytest.raises(ValueError):
        simulate_transactions(SimulationConfig(days=5))
