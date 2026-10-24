# Model Card — Card-Transaction Fraud Classifier

## Overview

| | |
|---|---|
| **Task** | Binary classification: is a card transaction fraudulent? |
| **Output** | Fraud probability and a 0/1 *review* flag at a cost-tuned threshold |
| **Selected model** | Logistic regression (`class_weight="balanced"`) on 12 engineered features |
| **Alternative evaluated** | Random forest (300 trees, depth 12) |
| **Selection criterion** | 5-fold stratified out-of-fold PR-AUC on the training split |
| **Artefact** | `models/fraud_model.joblib` — fitted pipeline + threshold + package version |

## Intended use

A portfolio demonstration of how a fraud / risk analyst builds, evaluates and
hands over a transaction-screening model. The flag is meant to route
transactions to **human review**, not to block customers automatically.

**Not intended for** production decisions on real customers. The model has
only ever seen synthetic data (see below).

## Data

Synthetic transactions from `fraud_detection.data.generate_transactions`
(60,000 rows, 1.7% fraud, seed 42). Fraud and legitimate rows are drawn from
different distributions for amount, hour, recency, velocity, device,
geography and merchant risk.

**Implication:** the classes are far more separable than in real payment data
(ROC-AUC ≈ 0.9998). The *method* transfers to real data; the *scores* do not.
Expect substantially lower PR-AUC on genuine transactions, where fraudsters
imitate normal behaviour.

## Features

Raw inputs (`RAW_COLUMNS`): amount, hour, days since last transaction,
transactions in the last hour, new device, foreign country, distance from
home, merchant risk.

Engineered (`FraudFeatures` transformer, fitted on training data only):
log amount, night flag (00–05h), high-velocity flag (≥3 txns/hour), geo-risk
flag (>100 km or foreign), and a new-device × foreign × high-amount
interaction whose amount cut-off is the *training* median.

## Evaluation protocol

1. Stratified 75/25 train/test split.
2. On the training split only: 5-fold out-of-fold predictions for each
   candidate → pick the best PR-AUC → tune the threshold that minimises
   `150 € × FN + 5 € × FP` on those predictions.
3. Refit the chosen pipeline on the full training split, score the test set
   **once** at the frozen threshold.

## Results (test set, 15,000 transactions, 255 frauds)

| Metric | Value |
|---|---|
| PR-AUC | 0.990 |
| ROC-AUC | 0.9998 |
| Recall | 97.6% (249 / 255) |
| Precision | 84.1% |
| Flagged traffic | 2.0% |
| Threshold | 0.77 |
| Cost at threshold / at 0.5 | €1,135 / €1,005 |

The tuned threshold does not beat 0.5 on this test sample; the cost curve is
flat between ~0.5 and ~0.8 and the difference is within the noise of ~250
positive cases. Source of truth: `reports/metrics.json`.

## Explainability

Permutation importance (drop in PR-AUC when a feature is shuffled) is
reported in `reports/metrics.json` and `reports/feature_importance.png`.
Merchant risk and distance from home dominate. Logistic regression
coefficients are also directly inspectable on the saved pipeline.

## Limitations and risks

- **Synthetic data**: no concept drift, no adversarial adaptation, no
  label delay or chargeback lag.
- **Static cost model**: a flat €150 per missed fraud ignores transaction
  amount; an amount-weighted cost would be more realistic.
- **No fairness assessment**: there are no customer attributes in the data.
  On real data, flag rates should be checked across customer segments,
  and features such as `foreign_country` reviewed for disparate impact.
- **Calibration** has not been assessed; treat probabilities as scores.

## Maintenance

Retrain and re-tune the threshold whenever cost assumptions change:
`fraud-detection train --cost-fn <€> --cost-fp <€>`.
