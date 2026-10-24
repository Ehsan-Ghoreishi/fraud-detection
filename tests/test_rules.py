import numpy as np
import pandas as pd

from fraud_detection.rules import RULES, compare_with_rules, rule_hits


def _rows():
    return pd.DataFrame({
        "amount": [900.0, 20.0, 3.0, 300.0, 15.0],
        "new_device": [0, 1, 1, 0, 0],
        "amount_vs_customer_median": [5.0, 4.0, 0.1, 6.0, 1.0],
        "small_txn_count_1h": [0, 0, 3, 0, 0],
        "foreign_country": [0, 0, 0, 1, 0],
        "distance_from_home": [1.0, 2.0, 1.0, 2_500.0, 3.0],
        "is_fraud": [0, 1, 1, 1, 0],
    })


def test_each_rule_fires_on_its_pattern():
    hits = rule_hits(_rows())
    assert list(hits.columns) == list(RULES)
    assert hits.high_amount.tolist() == [True, False, False, False, False]
    assert hits.new_device_big_spend.tolist() == [False, True, False, False, False]
    assert hits.card_testing_burst.tolist() == [False, False, True, False, False]
    assert hits.far_abroad_spend.tolist() == [False, False, False, True, False]


def test_model_is_compared_at_the_same_alert_volume():
    proba = np.array([0.1, 0.9, 0.8, 0.7, 0.2])
    result = compare_with_rules(_rows(), proba)
    assert result["alerts"] == 4
    assert result["rules"] == {"recall": 1.0, "precision": 0.75}
    # the model's top four alerts contain all three frauds as well
    assert result["model_same_alerts"]["recall"] == 1.0
    assert result["rule_precision"]["high_amount"] == 0.0
