"""
Generate the figures used in the README and any presentation:
  1. Precision-Recall curve (the right curve for imbalanced fraud data)
  2. Confusion matrix at the cost-optimal threshold
  3. Cost-vs-threshold curve showing why we don't use 0.5
  4. Feature importance / top fraud signals
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from fraud_detection.config import REPORTS
from fraud_detection.evaluation import cost_curve
from fraud_detection.model import train_and_evaluate

log = logging.getLogger(__name__)

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


def make_all(out: dict, reports_dir: Path = REPORTS) -> None:
    """Write every report figure from the output of train_and_evaluate."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = out["summary"]
    cost = summary["business_cost"]
    plot_pr_curve(out["y_test"], out["proba"], reports_dir / "pr_curve.png")
    plot_confusion(out["y_test"], out["proba"], out["threshold"],
                   reports_dir / "confusion_matrix.png")
    plot_cost_curve(out["y_train"], out["oof_proba"],
                    out["y_test"], out["proba"], out["threshold"],
                    cost["cost_false_negative"], cost["cost_false_positive"],
                    reports_dir / "cost_curve.png")
    plot_feature_importance(summary.get("feature_importance", []),
                            reports_dir / "feature_importance.png")


def main():
    make_all(train_and_evaluate())
    log.info("\nFigures written to reports/")


def plot_pr_curve(y_test, proba, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
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
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y_test, proba, threshold, path: Path) -> None:
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    cmap = LinearSegmentedColormap.from_list("navy", ["#EAF0F7", NAVY])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap=cmap)
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
    fig.savefig(path)
    plt.close(fig)


def plot_cost_curve(y_train, oof_proba, y_test, proba, threshold,
                    cost_fn, cost_fp, path: Path) -> None:
    """Cost per 1,000 transactions on CV (where tuned) and test data."""
    cv_table = cost_curve(y_train, oof_proba, cost_fn, cost_fp)
    test_table = cost_curve(y_test, proba, cost_fn, cost_fp)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(cv_table["threshold"], cv_table["cost"] / len(y_train) * 1000,
            color=ACCENT, lw=2.5, label="Train (out-of-fold) — used to tune")
    ax.plot(test_table["threshold"], test_table["cost"] / len(y_test) * 1000,
            color=NAVY, lw=1.5, alpha=0.7, label="Test (held out)")
    ax.axvline(threshold, color=RED, ls="--", lw=1.5,
               label=f"Chosen = {threshold:.2f}")
    ax.axvline(0.5, color="gray", ls=":", lw=1.5, label="Default = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost per 1,000 txns (€)")
    ax.set_title("Business Cost vs Decision Threshold")
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(feature_importance: list[dict], path: Path) -> None:
    if not feature_importance:
        return
    fi = feature_importance[:8][::-1]
    names = [d["feature"] for d in fi]
    vals = [d["importance"] for d in fi]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(names, vals, color=NAVY)
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    ax.set_title("Top Fraud Signals (permutation importance)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
