"""
Exploratory data analysis for the transaction dataset.

Produces a single figure that tells the fraud story visually: how fraud
differs from legit traffic across amount, hour, velocity, and geography.
This is the "understand the data before modelling" step a good analyst
never skips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fraud_detection.config import DATA_PATH, REPORTS

NAVY = "#1F3864"
RED = "#C0392B"

plt.rcParams.update({"figure.dpi": 120, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    df = pd.read_csv(DATA_PATH)
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
    fig.savefig(REPORTS / "eda_overview.png")
    print("EDA figure written to reports/eda_overview.png")

    # quick printed summary
    print("\n--- Fraud vs Legit summary ---")
    print(f"Median amount   | legit €{legit.amount.median():.0f} | "
          f"fraud €{fraud.amount.median():.0f}")
    print(f"Night txns (%)  | legit {legit.hour.between(0,5).mean():.1%} | "
          f"fraud {fraud.hour.between(0,5).mean():.1%}")
    print(f"New device (%)  | legit {legit.new_device.mean():.1%} | "
          f"fraud {fraud.new_device.mean():.1%}")
    print(f"Foreign (%)     | legit {legit.foreign_country.mean():.1%} | "
          f"fraud {fraud.foreign_country.mean():.1%}")


if __name__ == "__main__":
    main()
