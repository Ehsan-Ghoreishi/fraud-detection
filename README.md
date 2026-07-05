# Fraud Detection on Payment Transactions

An end-to-end machine-learning pipeline that flags fraudulent card
transactions, built the way a **fraud / risk analyst** would actually approach
it: domain-driven features, honest evaluation for imbalanced data, and a
**business-cost-based decision threshold** rather than a default 0.5.

> Built by **Seyedmohammad (Ehsan) Ghoreishi** — a data professional focused on
> risk, fraud, and financial analytics. This project demonstrates the exact
> skills used in Customer Risk / Anti-Financial-Crime (AFC) analytics roles at
> fintechs and banks.

---

## Why this project

Most beginner fraud projects report "99% accuracy" and stop there. On data
where fraud is ~1.7% of transactions, **a model that never predicts fraud is
already 98.3% accurate — and completely useless.** The interesting problem in
fraud is the trade-off between *catching fraud* and *not annoying real
customers*, and choosing where to draw that line based on cost.

This project is built around that reality.

---

## Headline results

On a held-out test set (25% of data, never seen in training):

| Metric | Value | What it means |
|---|---|---|
| **Fraud caught (recall)** | **98.8%** | Of all real fraud, we catch nearly all of it |
| **Precision** | 77.8% | Of everything we flag, ~4 in 5 are truly fraud |
| **PR-AUC** | 0.990 | Ranking quality on the imbalanced positive class |
| **ROC-AUC** | 0.9998 | Overall separability |
| **Traffic flagged** | 2.2% | Only a small slice needs review |
| **False alarms per real fraud** | 0.3 | Very low customer friction |

Concretely, on 15,000 test transactions the model missed only **3 frauds out
of 255** while flagging just 72 legitimate transactions for review.

![Precision-Recall Curve](reports/pr_curve.png)

---

## The idea that makes it a *risk* project, not just a *model*

A missed fraud and a false alarm do **not** cost the same. Letting fraud
through costs real money (chargebacks, losses); a false alarm just costs a
quick review. I encode that asymmetry directly and pick the decision threshold
that **minimises expected business cost** instead of accepting the default:

```
Cost = (false negatives × €150) + (false positives × €5)
```

![Cost vs Threshold](reports/cost_curve.png)

The optimal threshold lands at **0.65**, not 0.50 — and it's the single most
important decision a risk analyst owns. Change the cost assumptions and the
threshold moves; the pipeline recomputes it automatically.

---

## Understanding the data first (EDA)

Before modelling, I looked at *how* fraud actually behaves. The signal is
strong and intuitive — fraud skews toward nights, new devices, foreign
countries, higher amounts, and velocity spikes:

![EDA Overview](reports/eda_overview.png)

| Signal | Legit | Fraud |
|---|---|---|
| Median amount | €20 | €70 |
| Night transactions (00–05h) | 3.2% | 41.6% |
| New device | 5.0% | 53.6% |
| Foreign country | 7.9% | 43.8% |

---

## What the model relies on (explainability)

In a regulated bank you must be able to **justify why a customer was flagged**.
I use permutation importance (works for any model, measures real predictive
contribution) rather than opaque built-in scores:

![Feature Importance](reports/feature_importance.png)

Merchant risk and distance-from-home dominate — consistent with how real card
fraud presents.

![Confusion Matrix](reports/confusion_matrix.png)

---

## Engineering choices worth calling out

- **Stratified train/test split** so both sets keep the true fraud ratio.
- **No data leakage** — scaling and all preprocessing are fit on training data
  only, inside an sklearn `Pipeline`.
- **Imbalance handled with `class_weight="balanced"`**, not naive oversampling
  that would leak synthetic rows into evaluation.
- **Two models compared** (logistic regression vs random forest) and selected
  by **PR-AUC**, the correct metric for rare-event detection.
- **Reproducible** — fixed seeds, one command runs the whole thing.

---

## Project structure

```
fraud-detection/
├── src/
│   ├── generate_data.py   # realistic synthetic transactions (real fraud patterns)
│   ├── eda.py             # exploratory analysis + overview figure
│   ├── fraud_model.py     # feature engineering, training, cost-based evaluation
│   └── make_plots.py      # all reporting figures
├── reports/               # generated metrics.json + figures
├── data/                  # generated dataset (git-ignored in practice)
├── requirements.txt
└── README.md
```

> **Note on data:** real payment data is confidential, so this repo generates a
> synthetic dataset that reproduces the statistical patterns of genuine card
> fraud (imbalance, night-time skew, velocity, geography). The modelling
> pipeline is identical to what you'd run on real data — only the source
> changes.

---

## Run it yourself

```bash
pip install -r requirements.txt

python src/generate_data.py   # create the dataset
python src/eda.py             # exploratory figures
python src/fraud_model.py     # train + evaluate + metrics.json
python src/make_plots.py      # all reporting figures
```

Everything is deterministic; you'll reproduce the numbers above exactly.

---

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib`

## Skills demonstrated

Imbalanced classification · feature engineering · cost-sensitive decision
thresholds · model evaluation (PR-AUC, precision/recall, confusion matrix) ·
explainability (permutation importance) · leakage-free pipelines ·
reproducible, documented code — the core toolkit for **fraud, risk, and
financial-crime analytics**.
