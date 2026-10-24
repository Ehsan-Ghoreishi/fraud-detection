import numpy as np

from fraud_detection.evaluation import choose_threshold, cost_curve


def test_cost_curve_counts_errors():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    table = cost_curve(y, proba, cost_fn=100, cost_fp=1,
                       thresholds=[0.05, 0.5, 0.95])
    assert table.fp.tolist() == [2, 1, 0]
    assert table.fn.tolist() == [0, 1, 2]
    assert table.cost.tolist() == [2, 101, 200]


def test_matches_naive_loop():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = rng.random(500)
    table = cost_curve(y, proba, cost_fn=150, cost_fp=5)
    for t, cost in zip(table.threshold, table.cost, strict=True):
        pred = proba >= t
        expected = 5 * (pred & (y == 0)).sum() + 150 * (~pred & (y == 1)).sum()
        assert cost == expected


def test_threshold_moves_with_cost_assumptions():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2_000)
    proba = np.clip(y * 0.3 + rng.random(2_000) * 0.7, 0, 1)
    expensive_misses, _ = choose_threshold(y, proba, cost_fn=1_000, cost_fp=1)
    expensive_alarms, _ = choose_threshold(y, proba, cost_fn=1, cost_fp=1_000)
    assert expensive_misses < expensive_alarms


def test_perfect_separation_has_zero_cost():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold, table = choose_threshold(y, proba)
    assert table.cost.min() == 0
    assert 0.3 < threshold <= 0.8


def test_ties_resolve_to_lowest_threshold():
    y = np.array([0, 1])
    proba = np.array([0.0, 1.0])
    threshold, _ = choose_threshold(y, proba)
    assert threshold == 0.01
