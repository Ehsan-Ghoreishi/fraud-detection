import numpy as np

from fraud_detection.evaluation import (
    bootstrap_ci,
    choose_threshold,
    cost_curve,
    value_recall,
)


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
    threshold, table = choose_threshold(y, proba, cost_fn=150)
    assert table.cost.min() == 0
    assert 0.3 < threshold <= 0.8


def test_ties_resolve_to_lowest_threshold():
    from fraud_detection.evaluation import DEFAULT_THRESHOLDS

    y = np.array([0, 1])
    proba = np.array([0.0, 1.0])
    threshold, _ = choose_threshold(y, proba, cost_fn=150)
    assert threshold == DEFAULT_THRESHOLDS.min() == 0.001


def test_default_grid_is_fine_at_low_scores():
    from fraud_detection.evaluation import DEFAULT_THRESHOLDS

    low = DEFAULT_THRESHOLDS[DEFAULT_THRESHOLDS < 0.05]
    assert len(low) >= 25
    assert np.all(np.diff(DEFAULT_THRESHOLDS) > 0)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(100), np.zeros(1_900)]
    proba = np.clip(y * 0.4 + rng.random(2_000) * 0.6, 0, 1)
    ci = bootstrap_ci(y, proba, threshold=0.5, cost_fn=150, n_boot=200)
    recall = (proba[y == 1] >= 0.5).mean()
    assert ci["recall"][0] <= recall <= ci["recall"][1]
    assert set(ci) == {"pr_auc", "recall", "precision", "cost_vs_baseline"}
    # comparing a threshold with itself must give exactly zero difference
    same = bootstrap_ci(y, proba, threshold=0.5, cost_fn=150,
                        baseline_threshold=0.5,
                        n_boot=50)
    assert same["cost_vs_baseline"] == [0.0, 0.0]


def test_per_transaction_miss_costs():
    y = np.array([1, 1, 0])
    proba = np.array([0.9, 0.2, 0.1])
    table = cost_curve(y, proba, cost_fn=np.array([500.0, 3.0, 0.0]),
                       cost_fp=1, thresholds=[0.05, 0.5, 0.95])
    # at 0.5 only the cheap €3 fraud is missed; at 0.95 both are
    assert table.cost.tolist() == [1, 3, 503]


def test_expensive_frauds_pull_the_threshold_down():
    # catching the fraud means also flagging two legit txns scored the same
    y = np.array([1, 0, 0])
    proba = np.array([0.3, 0.3, 0.3])
    cheap, _ = choose_threshold(y, proba, cost_fn=np.array([1.0, 0, 0]),
                                cost_fp=1)
    pricey, _ = choose_threshold(y, proba, cost_fn=np.array([900.0, 0, 0]),
                                 cost_fp=1)
    assert pricey <= 0.3 < cheap


def test_value_recall():
    y = np.array([1, 1, 0])
    assert value_recall(y, [1, 0, 1], [300.0, 100.0, 50.0]) == 0.75


def test_cluster_bootstrap_is_wider_for_correlated_errors():
    # 40 customers with 25 identical transactions each: errors are perfectly
    # correlated within a customer, so the effective sample is 40, not 1,000
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(40), 25)
    y_cust = rng.random(40) < 0.5
    right = rng.random(40) < 0.7          # is the model right for this customer?
    y = np.repeat(y_cust, 25)
    proba = np.repeat(np.where(right, y_cust, ~y_cust) * 0.8 + 0.1, 25)
    naive = bootstrap_ci(y, proba, 0.5, cost_fn=10, n_boot=300)
    clustered = bootstrap_ci(y, proba, 0.5, cost_fn=10, n_boot=300,
                             groups=groups)
    width = lambda ci: ci["recall"][1] - ci["recall"][0]   # noqa: E731
    assert width(clustered) > 2 * width(naive)


def test_cluster_bootstrap_keeps_groups_whole():
    from fraud_detection.evaluation import _cluster_sampler

    groups = np.array([7, 7, 3, 3, 3, 9])
    idx = _cluster_sampler(groups, np.random.default_rng(0))()
    counts = np.bincount(groups[idx], minlength=10)
    # every drawn customer contributes all of its rows
    assert all(counts[g] % (groups == g).sum() == 0 for g in (3, 7, 9))


def test_episode_metrics_measure_time_to_detect():
    import pandas as pd

    from fraud_detection.evaluation import episode_metrics

    df = pd.DataFrame({
        "fraud_episode": [-1, 1, 1, 1, 2, 2, 3],
        "fraud_type": ["none", "ato", "ato", "ato", "card_testing",
                       "card_testing", "stealth"],
        "timestamp": pd.to_datetime(["2026-01-01"] * 7)
        + pd.to_timedelta([0, 1, 2, 3, 1, 2, 1], unit="h"),
        "amount": [10.0, 100.0, 200.0, 300.0, 2.0, 500.0, 50.0],
    })
    #   episode 1: first two slip through, third caught
    #   episode 2: caught on the first (test) charge
    #   episode 3: never caught
    flagged = [True, False, False, True, True, False, False]
    m = episode_metrics(df, flagged)
    assert m["episodes"] == 3
    assert m["detection_rate"] == round(2 / 3, 3)
    assert m["detected_on_first_txn"] == round(1 / 3, 3)
    assert m["median_txns_before_alert"] == 1.0      # median of [2, 0]
    # lost before detection: 100 + 200 (ep 1) + 50 (ep 3) of 1,152
    assert m["value_lost_before_detection"] == round(350 / 1152, 3)
    assert m["detection_rate_by_typology"] == {"ato": 1.0, "card_testing": 1.0,
                                               "stealth": 0.0}


def test_rate_matched_threshold_preserves_the_alert_rate():
    import pytest

    from fraud_detection.evaluation import rate_matched_threshold

    rng = np.random.default_rng(3)
    final_scores = rng.beta(1, 30, 50_000)       # a different score scale
    t = rate_matched_threshold(final_scores, 0.02)
    assert abs((final_scores >= t).mean() - 0.02) < 0.001
    with pytest.raises(ValueError):
        rate_matched_threshold(final_scores, 0.0)
