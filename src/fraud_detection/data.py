"""Create and load the transaction log."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraud_detection.config import DATA_PATH, RANDOM_STATE
from fraud_detection.simulation import SimulationConfig, simulate_transactions

log = logging.getLogger(__name__)

LOG_COLUMNS = [
    "transaction_id", "timestamp", "customer_id", "merchant_id",
    "merchant_category", "merchant_risk", "channel", "device_id",
    "foreign_country", "distance_from_home", "amount",
]


def generate(path: Path = DATA_PATH, config: SimulationConfig | None = None,
             seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Simulate a transaction log and write it to ``path`` as CSV."""
    df = simulate_transactions(config, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Generated %s transactions from %s customers (%.2f%% fraud) -> %s",
             f"{len(df):,}", f"{df.customer_id.nunique():,}",
             df.is_fraud.mean() * 100, path)
    return df


def load_transactions(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read a transaction log, restoring dtypes that CSV loses."""
    df = pd.read_csv(path, parse_dates=["timestamp"],
                     dtype={"device_id": str}, keep_default_na=False,
                     na_values={c: [""] for c in ("amount", "merchant_risk")})
    missing = [c for c in LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate()
