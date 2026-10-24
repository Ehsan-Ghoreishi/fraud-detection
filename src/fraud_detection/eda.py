"""
Exploratory data analysis for the transaction dataset.

Produces a single figure that tells the fraud story visually: how fraud
differs from legit traffic across amount, hour, velocity, and geography.
This is the "understand the data before modelling" step a good analyst
never skips.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_detection.config import DATA_PATH, REPORTS
from fraud_detection.features import NIGHT_HOURS

log = logging.getLogger(__name__)

NAVY = "#1F3864"
RED = "#C0392B"

plt.rcParams.update({"figure.dpi": 120, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def plot_overview(df: pd.DataFrame, path: Path) -> None:
    """Four-panel figure contrasting fraud with legitimate traffic."""
    legit = df[df.is_fraud == 0]
    fraud = df[df.is_fraud == 1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("How Fraud Differs from Legitimate Transactions",
                 fontsize=14, fontweight="bold")

    # 1. amount distribution (log scale)
    ax = axes[0, 0]
    bins = np.logspace(0, 3.5, 40)
    ax.hist(legit.amount, bins=bins, alpha=0.6, color=NAVY,
            label="Legit", density=True)
    ax.hist(fraud.amount, bins=bins, alpha=0.6, color=RED,
            label="Fraud", density=True)
    ax.set_xscale("log")
    ax.set_title("Transaction Amount")
    ax.set_xlabel("Amount (€, log scale)")
    ax.set_ylabel("Density")
    ax.legend()

    # 2. hour of day
    ax = axes[0, 1]
    ax.hist(legit.hour, bins=24, alpha=0.6, color=NAVY,
            label="Legit", density=True)
    ax.hist(fraud.hour, bins=24, alpha=0.6, color=RED,
            label="Fraud", density=True)
    ax.set_title("Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Density")
    ax.legend()

    # 3. fraud rate by velocity
    ax = axes[1, 0]
    vel = (df.assign(bucket=pd.cut(df.txn_count_1h, [-1, 0, 1, 2, 3, 100],
                                   labels=["0", "1", "2", "3", "4+"]))
             .groupby("bucket", observed=True).is_fraud.mean() * 100)
    ax.bar(vel.index.astype(str), vel.values, color=NAVY)
    ax.set_title("Fraud Rate by Transactions-per-Hour")
    ax.set_xlabel("Transactions in last hour")
    ax.set_ylabel("Fraud rate (%)")

    # 4. geography
    ax = axes[1, 1]
    geo = pd.DataFrame({
        "New device": [legit.new_device.mean(), fraud.new_device.mean()],
        "Foreign": [legit.foreign_country.mean(), fraud.foreign_country.mean()],
    }, index=["Legit", "Fraud"]) * 100
    geo.plot(kind="bar", ax=ax, color=[NAVY, RED], rot=0)
    ax.set_title("Device & Geography Risk Signals")
    ax.set_ylabel("% of transactions")
    ax.legend(title="")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path)
    plt.close(fig)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Headline fraud-vs-legit statistics, one row per signal."""
    night = df["hour"].between(*NIGHT_HOURS)
    g = df.assign(night=night).groupby("is_fraud")
    table = pd.DataFrame({
        "Median amount (€)": g["amount"].median(),
        "Night transactions": g["night"].mean(),
        "New device": g["new_device"].mean(),
        "Foreign country": g["foreign_country"].mean(),
    }).T
    table.columns = ["legit", "fraud"]
    return table


def main(data_path: Path = DATA_PATH, reports_dir: Path = REPORTS) -> None:
    df = pd.read_csv(data_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(df, reports_dir / "eda_overview.png")
    log.info(f"EDA figure written to {reports_dir / 'eda_overview.png'}")

    log.info("\n--- Fraud vs Legit summary ---")
    table = summarise(df)
    for name, row in table.iterrows():
        fmt = "€{:.0f}" if "amount" in name else "{:.1%}"
        log.info(f"{name:<20} | legit {fmt.format(row.legit):>6} | "
              f"fraud {fmt.format(row.fraud):>6}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
