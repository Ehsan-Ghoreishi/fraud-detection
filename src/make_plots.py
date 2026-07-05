"""
Generate the figures used in the README and any presentation:
  1. Precision-Recall curve (the right curve for imbalanced fraud data)
  2. Confusion matrix at the cost-optimal threshold
  3. Cost-vs-threshold curve showing why we don't use 0.5
  4. Feature importance / top fraud signals
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, confusion_matrix

from fraud_model import train_and_evaluate, choose_threshold

REPORTS = Path(__file__).resolve().parent.parent / "reports"
NAVY = "#1F3864"
ACCENT = "#2E86AB"
RED = "#C0392B"

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


def main():
    global summary
    out = train_and_evaluate()
    y_test, proba, threshold = out["y_test"], out["proba"], out["threshold"]
    summary = out["summary"]

    _pr_curve(y_test, proba)
    _confusion(y_test, proba, threshold)
    _cost_curve(y_test, proba)
    _feature_importance(summary)
    print("\nFigures written to reports/")


def _pr_curve(y_test, proba):
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = summary_ap(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color=NAVY, lw=2.5,
            label=f"Model (PR-AUC = {ap:.3f})")
    baseline = y_test.mean()
    ax.axhline(baseline, color=RED, ls="--", lw=1.5,
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("Precision (flags that are real fraud)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(REPORTS / "pr_curve.png")
    plt.close(fig)


def _confusion(y_test, proba, threshold):
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    cmap = LinearSegmentedColormap.from_list("navy", ["#EAF0F7", NAVY])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap=cmap)
    labels = ["Legit", "Fraud"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val:,}\n({val/total:.1%})",
                    ha="center", va="center",
                    color="white" if val > cm.max() / 2 else NAVY,
                    fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png")
    plt.close(fig)


def _cost_curve(y_test, proba):
    _, cost_table = choose_threshold(y_test, proba)
    best_t = cost_table.loc[cost_table["cost"].idxmin(), "threshold"]
    best_c = cost_table["cost"].min()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(cost_table["threshold"], cost_table["cost"],
            color=ACCENT, lw=2.5)
    ax.axvline(best_t, color=RED, ls="--", lw=1.5,
               label=f"Cost-optimal = {best_t:.2f}")
    ax.axvline(0.5, color="gray", ls=":", lw=1.5, label="Default = 0.50")
    ax.scatter([best_t], [best_c], color=RED, zorder=5)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected business cost (€)")
    ax.set_title("Why We Don't Use the Default Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "cost_curve.png")
    plt.close(fig)


def _feature_importance(summary):
    fi = summary.get("feature_importance")
    if not fi:
        return
    fi = fi[:8][::-1]
    names = [d["feature"] for d in fi]
    vals = [d["importance"] for d in fi]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(names, vals, color=NAVY)
    ax.set_xlabel("Importance")
    ax.set_title("Top Fraud Signals")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png")
    plt.close(fig)


def summary_ap(y_test, proba):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_test, proba)


# make `summary` importable inside _pr_curve without recompute juggling
summary = {}
if __name__ == "__main__":
    main()
