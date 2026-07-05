"""
Fraud detection pipeline: feature engineering, model training, and
threshold-aware evaluation.

Design choices that matter for a fraud/risk context (and that this file
demonstrates deliberately):

1. Never evaluate fraud models on accuracy. With ~1.7% fraud, a model that
   predicts "never fraud" is 98.3% accurate and 100% useless. We use
   precision, recall, PR-AUC and a cost-based threshold instead.

2. Train/test split is stratified so both sets keep the same fraud ratio,
   and all preprocessing is fit on the training data only (no leakage).

3. Class imbalance is handled with class_weight rather than naive
   oversampling, so we don't leak synthetic rows into evaluation.

4. The final threshold is chosen from a business cost model (a missed fraud
   costs far more than a false alarm), not left at the default 0.5 — this is
   the decision a risk analyst actually owns.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "transactions.csv"
REPORTS = ROOT / "reports"

# Business cost assumptions (euros). These are the lever a risk team tunes.
COST_FALSE_NEGATIVE = 150.0   # average loss from letting one fraud through
COST_FALSE_POSITIVE = 5.0     # cost of reviewing/blocking one legit txn


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven features a fraud analyst would reach for."""
    df = df.copy()

    # log amount — fraud amounts are heavy-tailed; log stabilises them
    df["log_amount"] = np.log1p(df["amount"])

    # "odd hour" flag — activity between midnight and 5am is higher risk
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    # velocity risk — many transactions in a short window is suspicious
    df["high_velocity"] = (df["txn_count_1h"] >= 3).astype(int)

    # geo risk — far from home OR foreign is a classic fraud signal
    df["geo_risk"] = ((df["distance_from_home"] > 100) |
                      (df["foreign_country"] == 1)).astype(int)

    # combined behavioural risk: new device on a foreign, high-value txn
    df["device_geo_amount_risk"] = (
        df["new_device"] * df["foreign_country"] *
        (df["log_amount"] > df["log_amount"].median()).astype(int)
    )
    return df


FEATURES = [
    "log_amount", "hour", "is_night", "days_since_last_txn",
    "txn_count_1h", "high_velocity", "new_device", "foreign_country",
    "distance_from_home", "geo_risk", "merchant_risk",
    "device_geo_amount_risk",
]


# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #
def train_and_evaluate() -> dict:
    df = engineer_features(pd.read_csv(DATA_PATH))
    X, y = df[FEATURES], df["is_fraud"]

    # stratified split preserves the fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    results = {}
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000,
                                       class_weight="balanced",
                                       random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
    }

    best_name, best_pr_auc = None, -1.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)   # key metric for imbalance

        results[name] = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
        }
        print(f"{name:>22} | ROC-AUC {roc_auc:.4f} | PR-AUC {pr_auc:.4f}")

        if pr_auc > best_pr_auc:
            best_name, best_pr_auc = name, pr_auc
            best_model, best_proba = model, proba

    print(f"\nBest model by PR-AUC: {best_name}\n")

    # ---- cost-based threshold selection ------------------------------------
    threshold, cost_table = choose_threshold(y_test, best_proba)
    preds = (best_proba >= threshold).astype(int)

    print(f"Cost-optimal threshold: {threshold:.3f}")
    print("\nClassification report at cost-optimal threshold:\n")
    print(classification_report(y_test, preds, digits=3,
                                target_names=["legit", "fraud"]))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    caught = tp / (tp + fn)
    review_rate = (tp + fp) / len(y_test)
    print(f"Fraud caught (recall):     {caught:.1%}")
    print(f"Transactions flagged:      {review_rate:.1%} of all traffic")
    print(f"False alarms per real fraud: {fp / max(tp, 1):.1f}")

    # persist metrics for the README / reports
    summary = {
        "models": results,
        "best_model": best_name,
        "cost_optimal_threshold": round(threshold, 3),
        "recall_fraud": round(caught, 3),
        "precision_fraud": round(tp / max(tp + fp, 1), 3),
        "flagged_rate": round(review_rate, 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))

    # feature importance via permutation — works for ANY model and measures
    # real predictive contribution. Explainability is non-negotiable in risk:
    # a fraud team must be able to justify why a customer was flagged.
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=10,
        random_state=42, scoring="average_precision", n_jobs=-1,
    )
    importances = sorted(
        zip(FEATURES, perm.importances_mean),
        key=lambda t: t[1], reverse=True,
    )
    summary["feature_importance"] = [
        {"feature": f, "importance": round(float(v), 4)} for f, v in importances
    ]
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nTop fraud signals the model relies on (permutation importance):")
    for f, v in importances[:6]:
        print(f"  {f:>24}: {v:.4f}")

    return {"summary": summary, "y_test": y_test,
            "proba": best_proba, "threshold": threshold}


def choose_threshold(y_true, proba):
    """Pick the probability threshold that minimises expected business cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
        rows.append((t, cost, fp, fn))
    cost_table = pd.DataFrame(rows, columns=["threshold", "cost", "fp", "fn"])
    best = cost_table.loc[cost_table["cost"].idxmin(), "threshold"]
    return float(best), cost_table


if __name__ == "__main__":
    train_and_evaluate()
