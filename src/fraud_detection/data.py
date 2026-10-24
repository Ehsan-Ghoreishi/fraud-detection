"""
Synthetic transaction data generator for fraud detection.

Real payment data is confidential, so this module generates a realistic
synthetic dataset that reproduces the statistical patterns seen in genuine
card-fraud data (heavy class imbalance, fraud concentrated at unusual hours,
higher amounts, new devices, velocity spikes, foreign transactions).

The point is not the data itself but demonstrating that the modelling
pipeline is built on domain-aware features a fraud/risk analyst would design.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fraud_detection.config import DATA_DIR, DATA_PATH, RANDOM_STATE

log = logging.getLogger(__name__)

def generate_transactions(n: int = 60_000, fraud_rate: float = 0.017,
                          seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic but realistic transaction dataset.

    Parameters
    ----------
    n : total number of transactions
    fraud_rate : proportion of fraudulent transactions (real card fraud is ~1-2%)
    seed : random seed for reproducibility
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if not 0 < fraud_rate < 1:
        raise ValueError(f"fraud_rate must be in (0, 1), got {fraud_rate}")

    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_legit = n - n_fraud

    # ---- Legitimate transactions -------------------------------------------
    legit = pd.DataFrame({
        # amounts: most everyday spending is small, log-normal shaped
        "amount": rng.lognormal(mean=3.0, sigma=1.0, size=n_legit).round(2),
        # hour of day: legit activity peaks during waking hours
        "hour": rng.choice(range(24), size=n_legit,
                           p=_daytime_distribution()),
        # days since the account last transacted (recency)
        "days_since_last_txn": rng.exponential(scale=2.0, size=n_legit).round(2),
        # number of transactions in the last hour (velocity)
        "txn_count_1h": rng.poisson(lam=1.2, size=n_legit),
        # is the device new to this customer?
        "new_device": rng.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
        # is the transaction in a foreign country?
        "foreign_country": rng.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        # distance (km) from customer's usual location
        "distance_from_home": rng.exponential(scale=15, size=n_legit).round(1),
        # merchant risk score (0-1), most merchants are low risk
        "merchant_risk": rng.beta(a=2, b=8, size=n_legit).round(3),
        "is_fraud": 0,
    })

    # ---- Fraudulent transactions -------------------------------------------
    # Fraud has systematically different behaviour: odd hours, bigger amounts,
    # new devices, velocity spikes, foreign, far from home, risky merchants.
    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=4.2, sigma=1.2, size=n_fraud).round(2),
        "hour": rng.choice(range(24), size=n_fraud,
                          p=_nighttime_distribution()),
        "days_since_last_txn": rng.exponential(scale=5.0, size=n_fraud).round(2),
        "txn_count_1h": rng.poisson(lam=4.5, size=n_fraud),
        "new_device": rng.choice([0, 1], size=n_fraud, p=[0.45, 0.55]),
        "foreign_country": rng.choice([0, 1], size=n_fraud, p=[0.55, 0.45]),
        "distance_from_home": rng.exponential(scale=120, size=n_fraud).round(1),
        "merchant_risk": rng.beta(a=6, b=3, size=n_fraud).round(3),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    # shuffle so fraud isn't all at the bottom
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # card schemes have a minimum charge; drop implausible micro-amounts
    df["amount"] = df["amount"].clip(0.5, None)
    return df


def _daytime_distribution() -> np.ndarray:
    """Hour-of-day probabilities weighted toward daytime (legit)."""
    weights = np.array([
        0.5, 0.3, 0.2, 0.2, 0.3, 0.5,   # 0-5  (night, low)
        1.0, 2.0, 3.0, 3.5, 4.0, 4.5,   # 6-11 (morning ramp)
        5.0, 4.5, 4.0, 4.0, 4.5, 5.0,   # 12-17 (afternoon peak)
        4.5, 3.5, 3.0, 2.0, 1.5, 1.0,   # 18-23 (evening decline)
    ])
    return weights / weights.sum()


def _nighttime_distribution() -> np.ndarray:
    """Hour-of-day probabilities weighted toward night (fraud)."""
    weights = np.array([
        4.0, 4.5, 5.0, 4.5, 4.0, 3.0,   # 0-5  (night, high)
        2.0, 1.5, 1.0, 1.0, 1.0, 1.0,   # 6-11
        1.0, 1.0, 1.0, 1.5, 1.5, 2.0,   # 12-17
        2.5, 3.0, 3.5, 4.0, 4.0, 4.0,   # 18-23
    ])
    return weights / weights.sum()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DATA_DIR.mkdir(exist_ok=True)
    df = generate_transactions()
    df.to_csv(DATA_PATH, index=False)
    log.info(f"Generated {len(df):,} transactions "
          f"({df.is_fraud.mean():.2%} fraud) -> data/transactions.csv")
