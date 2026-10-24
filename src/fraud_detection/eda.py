"""
Exploratory data analysis for the transaction log.

Produces one figure that tells the fraud story visually — fraud is unusual
*for that customer*, comes in bursts, and its mix shifts over time — plus a
per-typology summary table. This is the "understand the data before
modelling" step a good analyst never skips.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_detection.config import DATA_PATH, REPORTS
from fraud_detection.data import load_transactions
from fraud_detection.features import NIGHT_HOURS, add_behavioural_features
from fraud_detection.simulation import FRAUD_TYPES

log = logging.getLogger(__name__)

NAVY = "#1F3864"
RED = "#C0392B"
TYPE_COLORS = dict(zip(FRAUD_TYPES, ["#C0392B", "#E67E22", "#2E86AB", "#7F8C8D"],
                       strict=True))

plt.rcParams.update({"figure.dpi": 120, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def plot_overview(df: pd.DataFrame, path: Path) -> None:
    """Four-panel figure contrasting fraud with legitimate traffic.

    ``df`` is a log enriched with add_behavioural_features. Fraud here means
    true fraud (``fraud_type``), reported or not.
    """
    is_fraud = df.fraud_type != "none"
    legit, fraud = df[~is_fraud], df[is_fraud]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    fig.suptitle("How Fraud Differs from Legitimate Transactions",
                 fontsize=14, fontweight="bold")

    # 1. amount relative to the customer's own history; histogram of the
    # log so that bin widths (and therefore densities) are comparable
    ax = axes[0, 0]
    bins = np.linspace(-2, 2, 41)
    for part, color, label in ((legit, NAVY, "Legit"), (fraud, RED, "Fraud")):
        ax.hist(np.log10(part.amount_vs_customer_median), bins=bins, alpha=0.6,
                color=color, label=label, density=True)
    ax.set_xticks([-2, -1, 0, 1, 2], ["÷100", "÷10", "usual", "×10", "×100"])
    ax.set_title("Amount vs the Customer's Usual Spend")
    ax.set_xlabel("Amount relative to the customer's past median")
    ax.set_ylabel("Density")
    ax.legend()

    # 2. new-device rate for each typology
    ax = axes[0, 1]
    groups = ["legit", *FRAUD_TYPES]
    rate = df.groupby(df.fraud_type.replace("none", "legit")).new_device\
        .mean().reindex(groups) * 100
    ax.bar(range(len(groups)), rate.values,
           color=[NAVY] + [TYPE_COLORS[t] for t in FRAUD_TYPES])
    ax.set_xticks(range(len(groups)), [g.replace("_", "\n") for g in groups])
    ax.set_title("Transactions from a Device First Seen < 24h Ago")
    ax.set_ylabel("% of transactions")

    # 3. fraud rate by velocity
    ax = axes[1, 0]
    labels = ["0", "1", "2", "3-4", "5+"]
    buckets = pd.cut(df.txn_count_1h, [-1, 0, 1, 2, 4, 1_000], labels=labels)
    vel = is_fraud.groupby(buckets, observed=False).mean() * 100
    ax.bar(range(len(labels)), vel.values, color=NAVY)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_title("Fraud Rate by Transactions in the Previous Hour")
    ax.set_xlabel("Customer's transactions in the previous hour")
    ax.set_ylabel("Fraud rate (%)")

    # 4. typology mix over time (drift)
    ax = axes[1, 1]
    month = fraud.timestamp.dt.to_period("M")
    mix = pd.crosstab(month, fraud.fraud_type, normalize="index")[FRAUD_TYPES]
    mix.index = mix.index.strftime("%b")
    bottom = np.zeros(len(mix))
    for ftype in FRAUD_TYPES:
        ax.bar(range(len(mix)), mix[ftype] * 100, bottom=bottom,
               color=TYPE_COLORS[ftype], label=ftype.replace("_", " "))
        bottom += mix[ftype].to_numpy() * 100
    ax.set_xticks(range(len(mix)), mix.index)
    ax.set_title("Fraud Typology Mix by Month")
    ax.set_ylabel("% of fraud transactions")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path)
    plt.close(fig)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Headline statistics per typology (and legit), one row per group."""
    g = df.assign(night=df.timestamp.dt.hour.between(*NIGHT_HOURS),
                  online=df.channel == "online").groupby("fraud_type")
    table = pd.DataFrame({
        "transactions": g.size(),
        "reported": g.is_fraud.mean(),
        "median_amount": g.amount.median(),
        "amount_vs_usual": g.amount_vs_customer_median.median(),
        "online": g.online.mean(),
        "new_device": g.new_device.mean(),
        "foreign": g.foreign_country.mean(),
        "night": g.night.mean(),
    })
    return table.loc[["none", *FRAUD_TYPES]].rename(index={"none": "legit"})


def main(data_path: Path = DATA_PATH, reports_dir: Path = REPORTS) -> None:
    df = add_behavioural_features(load_transactions(data_path))
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(df, reports_dir / "eda_overview.png")
    log.info(f"EDA figure written to {reports_dir / 'eda_overview.png'}")

    log.info("\n--- Legit vs fraud typologies ---")
    with pd.option_context("display.width", 120,
                           "display.float_format", "{:.2f}".format):
        log.info(summarise(df).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
