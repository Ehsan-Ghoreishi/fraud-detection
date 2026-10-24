# Model Card — Card-Transaction Fraud Classifier

## Overview

| | |
|---|---|
| **Task** | Binary classification: is a card transaction fraudulent? |
| **Output** | Fraud score, a 0/1 *review* flag at a cost-tuned threshold, and reason codes |
| **Selected model** | `HistGradientBoostingClassifier` (`class_weight="balanced"`) on 27 features |
| **Alternatives evaluated** | Logistic regression, random forest |
| **Selection criterion** | Forward-chaining (4-fold) out-of-fold PR-AUC on the training period |
| **Artefact** | `models/fraud_model.joblib` — fitted pipeline, threshold, reason-code reference, package version |

## Intended use

A portfolio demonstration of how a fraud / risk analytics team builds,
evaluates and hands over a transaction-screening model. The flag routes
transactions to **human review**; it is not meant to decline customers
automatically.

**Not intended for** decisions on real customers: the model has only seen
simulated data.

## Data

Simulated with `fraud_detection.simulation` (seed 42): 2,500 customers,
1 January – 30 June 2026, 257,573 transactions, 0.53% labelled fraud.

- Customers have individual spend levels, active hours, online affinity,
  devices (25% replace their phone mid-period) and trips abroad (15%).
- Four fraud typologies: card testing, account takeover, cloned card
  (60% abroad, 40% domestic far from home) and stealth.
- **Drift**: from day 120 account takeover rises from 25% to 40% of
  episodes and card-test amounts rise from €0.5–3 to €1–9.
- **Label noise**: 3–5% of fraud (20% of stealth fraud) is unreported and
  labelled legitimate; `fraud_type` keeps the ground truth for analysis and
  is never a model input.

Simulated data is still cleaner than reality: no merchant compromise
events, no first-party fraud, no chargeback delay, and fraud typologies
are drawn from known distributions.

## Features

**Point-in-time behaviour** (`add_behavioural_features`, earlier
transactions of the same customer only — verified by a no-look-ahead test):
prior transaction count, seconds since last transaction, transactions and
spend in the previous hour / day, small (< €10) charges in the previous
hour, amount vs the customer's past median, device age, new device
(< 24h), first purchase in a merchant category.

**Transaction attributes** (`FraudFeatures`): log amount, amount vs the
category median (learned on training data), hour, night flag, channel,
foreign country, log distance from home, merchant risk score, one-hot
merchant category.

## Evaluation protocol

1. **Out-of-time split** at 1 May: test 1 May – 30 Jun (86,798 txns).
2. **Label-delay embargo**: labels arrive with a delay, so a model deployed
   on 1 May can only learn from transactions up to 16 April (150,895 txns).
   17–30 April remains available as unlabelled history and as the
   threshold anchor (step 5).
3. Forward-chaining CV (4 folds) on the labelled period, with the same
   14-day embargo as a gap between every training and validation window →
   candidate with the best out-of-fold PR-AUC.
4. On the two most recent folds, find the threshold minimising
   `Σ(amount + €20) over missed fraud + €5 × false alarms` and record its
   **alert rate** (1.26%).
5. Refit on all labelled data. Fold models score on a different scale
   (less history), so the deployed threshold is the score at that alert
   rate on the final model's predictions for the embargo window
   (0.065, vs 0.025 on the fold-model scale).
6. Score the test period **once**. Intervals: 1,000 bootstrap resamples
   of *customers* (fraud transactions on one card are not independent).

## Results (test period)

| Metric | Value | 95% clustered CI |
|---|---|---|
| PR-AUC | 0.748 | 0.690 – 0.792 |
| ROC-AUC | 0.966 | |
| Recall (transactions) | 83.2% (386 / 464) | 78.7% – 86.6% |
| Recall (fraud value) | 92.8% | |
| Precision | 35.4% | 29.6% – 40.2% |
| Alerts per day | 18 | |
| Threshold | 0.065 | |
| Cost at threshold / at 0.5 / no model | €9.5k / €12.8k / €71.6k | saving vs 0.5: €1.1k – €5.8k |
| Threshold regret vs hindsight optimum (0.12) | €616 | |
| Brier score | 0.0026 | |

**Per attack** (158 fraud episodes in the test period): 65.8% detected;
account takeover 98.5%, card testing 100%, cloned card 100%, stealth 1.9%.
46% are stopped on the first fraudulent transaction, the median detected
attack has zero fraud transactions before the first alert, and 8.6% of
fraud value is lost before detection. All 29 card-testing attacks alert
before the cash-out.

**By typology** (transaction recall): card testing 88%, account takeover
90%, cloned card 100% (n = 46), stealth 2%.

**By month** (PR-AUC): May 0.72, June 0.78. The first month after the
tactics shift is the weakest.

**Rule-engine benchmark**: at the rule engine's 2,118 alerts, rules catch
62% of fraud and the model 88%.

**Unreported fraud**: 8 of 24 unlabelled fraud transactions are flagged.
In production these would surface as "false positives" that are actually
fraud.

Source of truth: `reports/metrics.json`.

## Explainability

- **Per alert**: reason codes (`explain.py`). Each feature group is reset
  to a typical training value and the groups that lower the score most are
  reported in plain language. Sample: `reports/sample_alerts.csv`.
- **Global**: permutation importance on the test period
  (`reports/feature_importance.png`); device age, merchant risk, distance
  and the new-device flag lead.

## Calibration

![Calibration](../reports/calibration.png)

Class weighting makes the model over-predict: mid-range scores correspond
to observed fraud rates well below the score. Ranking, the cost-tuned flag
and reason codes are unaffected, but the output is a **score, not a
probability**. Wrap the pipeline in `CalibratedClassifierCV` (and re-tune
the threshold) if calibrated probabilities are needed, e.g. for
expected-loss reporting.

## Limitations and risks

- **Stealth fraud is not detected.** Transactions that match the
  customer's own behaviour need other controls (cardholder notifications,
  dispute analytics).
- **Device dependence.** Device age is the strongest signal; device-ID
  spoofing or shared devices would weaken the model considerably.
- **Simulated data** (see above); absolute numbers will not transfer.
- **Cold start.** New customers have little history, so behaviour features
  are uninformative for their first transactions.
- **Fairness** not assessed: there are no customer demographics in the
  data. On real data, alert rates should be compared across customer
  segments and features like `foreign_country` reviewed for disparate
  impact (e.g. on migrant customers who often transact abroad).
- **Static costs.** Fixed review cost and chargeback fee; no customer
  lifetime-value effect of false declines.
- **Threshold anchoring assumes a stable fraud rate** between the embargo
  window and deployment. A sudden attack wave changes the score
  distribution, and with it the alert rate the threshold was set for.
  Monitor alert volume daily.

## Maintenance

Monitor monthly PR-AUC and recall by typology; retrain when they fall or
when the typology mix changes. Re-derive the threshold whenever cost
assumptions change:
`fraud-detection train --chargeback-fee <€> --cost-fp <€>`.
