"""A hand-written rule engine: the baseline a model has to beat.

Most fraud teams start with rules like these. Comparing the model with them
at the *same alert volume* answers the question a fraud manager actually
asks: "for the same analyst workload, how much more fraud do we catch?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {
    "high_amount": lambda d: d["amount"] > 500,
    "new_device_big_spend": lambda d: (d["new_device"] == 1)
    & (d["amount_vs_customer_median"] > 3),
    "card_testing_burst": lambda d: d["small_txn_count_1h"] >= 2,
    "far_abroad_spend": lambda d: (d["foreign_country"] == 1)
    & (d["distance_from_home"] > 1_000) & (d["amount_vs_customer_median"] > 3),
}


def rule_hits(enriched: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule for a log with behavioural features."""
    return pd.DataFrame({name: rule(enriched).to_numpy()
                         for name, rule in RULES.items()}, index=enriched.index)


def compare_with_rules(enriched: pd.DataFrame, proba) -> dict:
    """Rule engine vs model at an identical number of alerts."""
    y = enriched["is_fraud"].to_numpy().astype(bool)
    hits = rule_hits(enriched)
    flagged = hits.any(axis=1).to_numpy()
    n_alerts = int(flagged.sum())
    # model at the same workload: alert on the n highest-scored transactions
    top = np.zeros(len(y), bool)
    top[np.argsort(-np.asarray(proba), kind="stable")[:n_alerts]] = True

    def stats(f):
        return {"recall": round(float(f[y].mean()), 3),
                "precision": round(float(y[f].mean()) if f.any() else 0.0, 3)}

    return {
        "alerts": n_alerts,
        "rules": stats(flagged),
        "model_same_alerts": stats(top),
        "rule_precision": {name: round(float(y[h].mean()), 3) if h.any() else None
                           for name, h in hits.items()},
    }
