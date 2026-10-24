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
from sklearn.calibration import calibration_curve
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
    plot_cost_curve(out["train"], out["test"], out["threshold"],
                    cost["chargeback_fee"], cost["cost_false_positive"],
                    reports_dir / "cost_curve.png",
                    threshold_cv=out.get("threshold_cv"))
    plot_feature_importance(summary.get("feature_importance", []),
                            reports_dir / "feature_importance.png")
    plot_calibration(out["y_test"], out["proba"], reports_dir / "calibration.png")
    if "recall_by_typology" in summary:
        plot_typology_recall(summary["recall_by_typology"],
                             reports_dir / "recall_by_typology.png")


def make_ieee(out: dict, reports_dir: Path) -> None:
    """Report figures for the IEEE-CIS track."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary, cost = out["summary"], out["summary"]["business_cost"]
    plot_pr_curve(out["y_test"], out["proba"], reports_dir / "pr_curve.png")
    plot_confusion(out["y_test"], out["proba"], out["threshold"],
                   reports_dir / "confusion_matrix.png")
    plot_cost_curve(out["train"], out["test"], out["threshold"],
                    cost["chargeback_fee"], cost["cost_false_positive"],
                    reports_dir / "cost_curve.png",
                    threshold_cv=out["threshold_cv"])
    plot_feature_importance(summary["feature_family_importance"],
                            reports_dir / "feature_family_importance.png",
                            title="Fraud Signals by Feature Family")
    plot_calibration(out["y_test"], out["proba"], reports_dir / "calibration.png")
    plot_typology_recall(summary["recall_by_product"],
                         reports_dir / "recall_by_product.png",
                         title="Recall by Product (test set)")


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


def plot_cost_curve(train, test, threshold, chargeback_fee, cost_fp,
                    path: Path, threshold_cv: float | None = None) -> None:
    """Cost per 1,000 transactions on CV (where tuned) and test data.

    ``train`` carries ``oof_proba`` and ``threshold_fold`` (rows of the
    folds the threshold was tuned on); ``test`` carries ``proba``. A missed
    fraud costs amount + fee.
    """
    tr = train[train["threshold_fold"]]
    cv_table = cost_curve(tr["is_fraud"], tr["oof_proba"],
                          tr["amount"] + chargeback_fee, cost_fp)
    test_table = cost_curve(test["is_fraud"], test["proba"],
                            test["amount"] + chargeback_fee, cost_fp)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(cv_table["threshold"], cv_table["cost"] / len(tr) * 1000,
            color=ACCENT, lw=2.5, label="Recent CV folds — used to tune")
    ax.plot(test_table["threshold"], test_table["cost"] / len(test) * 1000,
            color=NAVY, lw=1.5, alpha=0.7, label="Test (out of time)")
    if threshold_cv is not None:
        ax.axvline(threshold_cv, color=ACCENT, ls="-.", lw=1.2,
                   label=f"CV optimum (fold-model scale) = {threshold_cv:.3f}")
    ax.axvline(threshold, color=RED, ls="--", lw=1.5,
               label=f"Deployed = {threshold:.3f}")
    ax.axvline(0.5, color="gray", ls=":", lw=1.5, label="Default = 0.50")
    ax.set_xscale("log")
    ax.set_xlabel("Decision threshold (log scale)")
    ax.set_ylabel("Expected cost per 1,000 txns (€)")
    ax.set_title("Business Cost vs Decision Threshold")
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_typology_recall(recall_by_typology: dict, path: Path,
                         title: str = "Recall by Fraud Typology (test set)"
                         ) -> None:
    """Share of each fraud typology caught at the chosen threshold."""
    names = list(recall_by_typology)
    recall = [recall_by_typology[n]["recall"] * 100 for n in names]
    counts = [recall_by_typology[n]["transactions"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.barh([n.replace("_", " ") for n in names][::-1], recall[::-1],
                   color=NAVY)
    for bar, r, n in zip(bars, recall[::-1], counts[::-1], strict=True):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{r:.0f}%  (n={n})", va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Fraud transactions caught (%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(feature_importance: list[dict], path: Path,
                            title: str = "Top Fraud Signals") -> None:
    if not feature_importance:
        return
    fi = feature_importance[:8][::-1]
    names = [d["feature"] for d in fi]
    vals = [d["importance"] for d in fi]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(names, vals, color=NAVY)
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(y_test, proba, path: Path) -> None:
    """Reliability diagram: predicted probability vs observed fraud rate."""
    # uniform bins: with ~2% fraud, quantile bins put every fraud in one bin
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10,
                                            strategy="uniform")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1.5,
            label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, color=NAVY, lw=2, marker="o",
            label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration (test set)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
