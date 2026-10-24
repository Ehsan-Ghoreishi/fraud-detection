# Model Card — Card-Transaction Fraud Classifier

## Overview

| | |
|---|---|
| **Task** | Binary classification: is a card transaction fraudulent? |
| **Output** | Fraud score, a 0/1 *review* flag at a cost-tuned threshold, and reason codes |
| **Candidates** | Logistic regression, random forest, `HistGradientBoostingClassifier` (class-weighted) |
| **Selection criterion** | Forward-chaining CV PR-AUC, computed per fold and averaged; every fold model sees ≥ 50% of the history |
| **Selected (simulated track)** | Random forest on 27 features |
| **Selected (ULB real data)** | Gradient boosting on 30 features |
| **Selected (IEEE-CIS real data)** | Gradient boosting on Vesta's columns plus behaviour features |
| **Artefact** | `models/fraud_model.joblib` — fitted pipeline, threshold, reason-code reference, package version |

## Intended use

A portfolio demonstration of how a fraud / risk analytics team builds,
evaluates and hands over a transaction-screening model. The flag routes
transactions to **human review**; it is not meant to decline customers
automatically. **Not intended for** decisions on real customers.

## Data

The same protocol runs on three tracks.

| Track | Source | Size | Use |
|---|---|---|---|
| Simulated | `fraud_detection.simulation`, seed 42 | 257,573 txns, 2,500 customers, Jan–Jun 2026, 0.53% fraud | Ground truth per typology, attack and unreported fraud |
| ULB (real) | Dal Pozzolo et al., OpenML 42175 | 284,807 txns over 48h, 0.17% fraud | Real-data check of the protocol |
| IEEE-CIS (real) | Vesta, Kaggle competition (licence required) | 590,540 txns, Dec 2017 – May 2018, 3.5% fraud | Full pipeline on real data, incl. behaviour features |

**Simulated portfolio.** Customers have individual spend levels, active
hours, online affinity, devices (25% replace their phone) and trips abroad
(15%). There are four fraud typologies: card testing, account takeover,
cloned card (60% abroad) and stealth. From day 120 account takeover rises
from 25% to 40% of episodes and card-test amounts rise from €0.5–3 to €1–9.
3–5% of fraud (20% of stealth) is unreported. `fraud_type` and
`fraud_episode` are analysis-only ground truth, never model inputs. The
data is still cleaner than reality: no merchant compromise events, no
first-party fraud.

**ULB.** Real European card transactions; features are PCA components
(V1–V28), time and amount, with no card or customer ID.

**IEEE-CIS.** Real e-commerce transactions with ~430 anonymised Vesta
columns. Time is anchored at 2017-12-01 (the common convention), and a
client is reconstructed as card1 + addr1 + account start day (day − D1).

## Features

**Point-in-time behaviour** (`add_behavioural_features`; earlier
transactions of the same customer only, verified by a no-look-ahead test):
prior transaction count, seconds since last transaction, transactions and
spend in the previous hour / day, small (< €10) charges in the previous
hour, amount vs the customer's past median, device age, new device (< 24h),
first purchase in a merchant category. These are used on the simulated
track and, on reconstructed clients, on IEEE-CIS.

**Transaction attributes**: amount, hour, channel, geography, merchant
risk and category (simulated); PCA components, amount and hour (ULB);
Vesta's columns with category codes and frequencies learned on training
data (IEEE-CIS).

## Evaluation protocol

1. **Out-of-time split**: the last part of the period is the test set
   (simulated: May–June; ULB: last 12 hours; IEEE-CIS: April–May).
2. **Label-delay embargo**: labels arrive late, so the days just before the
   test period are excluded from training (14 days; 6 hours on the 48-hour
   ULB data). They remain available as unlabelled history and as the
   threshold anchor.
3. **Forward-chaining CV** (4 folds, same embargo as a gap). Validation
   folds are sized so every fold model trains on ≥ 50% of the history.
   PR-AUC is computed per fold and averaged, because fold models score on
   different scales and pooling their predictions mixes those scales.
4. **Threshold**: on the two most recent folds, minimise
   `Σ(amount + €20) over missed fraud + €5 × false alarms`; keep the
   resulting **alert rate**.
5. **Refit** on all labelled data; the deployed threshold is the score at
   that alert rate on the final model's predictions for the embargo window.
6. **Score the test period once.** Intervals: 1,000 bootstrap resamples of
   customers (transactions on ULB, which has no customer IDs). Runner-up
   models are also scored on test as a diagnostic, never for decisions.
   The ranking is also read at fixed alert budgets (0.5%, 1%, 2%, 5% of
   traffic), the volumes a fixed-size review team can work.

## Results — simulated track (test: May–June)

| Metric | Value | 95% clustered CI |
|---|---|---|
| PR-AUC | 0.652 | 0.586 – 0.704 |
| ROC-AUC | 0.958 | |
| Recall (transactions) | 82.5% (383 / 464) | 77.9% – 86.3% |
| Recall (fraud value) | 92.3% | |
| Precision | 36.8% | 30.5% – 41.8% |
| Alerts per day | 17 | |
| Threshold | 0.131 (CV scale 0.20; alert rate 1.15%) | |
| Cost at threshold / at 0.5 / no model | €9.7k / €18.7k / €71.6k | saving vs 0.5: €5.2k – €13.0k |
| Threshold regret vs hindsight optimum (0.12) | €62 | |
| Brier score | 0.0032 | |

**Model selection noise.** CV PR-AUC: random forest 0.761, gradient
boosting 0.753, logistic regression 0.586. Per-fold values for the
selected model range 0.63–0.88. On test, gradient boosting (diagnostic)
scores 0.748 vs 0.652 for the selected random forest: a 0.008 CV margin
does not separate the two.

**Per attack** (158 episodes): 63.9% detected. By typology: account
takeover 95.5%, card testing 100%, cloned card 100%, stealth 0%. 44.9% are
stopped on the first fraudulent transaction, and 9.7% of fraud value is
lost before the first alert. All 29 card-testing attacks alert before the
cash-out.

**By month** (PR-AUC): May 0.61, June 0.71. The first month after the
tactics shift is the weakest.

**Rule-engine benchmark**: at the rule engine's 2,118 alerts, rules catch
62% of fraud and the model 87%.

**Unreported fraud**: 6 of 24 unlabelled fraud transactions are flagged.

## Results — ULB real data (test: last 12 hours)

| Metric | Value |
|---|---|
| PR-AUC (gradient boosting) | 0.777 (95% CI 0.705 – 0.842) |
| PR-AUC (logistic regression, diagnostic) | 0.744 |
| Recall / precision | 83.5% / 18.0% |
| Recall (fraud value) | 75.8% |
| Cost at threshold / at 0.5 / CV-scale threshold / no model | €6.5k / €5.1k / €5.1k / €18.7k |

Threshold anchoring **fails** on ULB: the 6-hour anchor window (06:00–12:00
on day 2) had a fraud rate of 0.075%, a third of the training period's, so
the transferred threshold flags too much legitimate traffic and costs
€1.4k more than 0.5. See limitations.

## Results — IEEE-CIS

## Results — IEEE-CIS real data (test: April–May 2018)

Train to 17 March 2018 (371,853 txns), 14-day embargo, test 175,998 txns;
217,779 reconstructed clients.

| Metric | Value |
|---|---|
| PR-AUC (gradient boosting) | 0.515 (95% clustered CI 0.488 – 0.542); CV 0.522 |
| ROC-AUC | 0.892 |
| Ablation without behaviour features | CV 0.521, test 0.516 |
| Logistic regression (diagnostic) | CV 0.359, test 0.233 |
| Clients unseen before the test period (104,922 txns) | PR-AUC 0.559, recall 89% |
| Compromised accounts detected | 90.5% of 2,613; 88% on the first fraud |
| Deployed threshold | 0.18: recall 85.4%, precision 11.2%, 26% of traffic flagged |
| Cost at threshold / at 0.5 / no model | €334k / €440k / €1.07M; regret €9.1k |
| At a 1% / 5% alert budget | recall 25.5% / 55.8%, precision 88% / 39% |
| Rule engine at equal volume (8,024 alerts) | rules 9%, model 54% recall |

Importance is dominated by Vesta's V and C families. Our behaviour features
carry 0.3% of it and add +0.001 CV PR-AUC: Vesta's C/D columns already hold
per-card counts and time deltas. The cost-optimal threshold flags 26% of
traffic, which no review team can work; the alert-budget figures are the
operational ones.

## Explainability

- **Per alert**: reason codes (`explain.py`). Each feature group is reset
  to a typical training value, and the groups that lower the score most are
  reported in plain language. Correlated features (device age and
  new-device flag) form one group. Sample: `reports/sample_alerts.csv` (the IEEE-CIS sample is generated
  locally only, as it contains licensed rows).
- **Global**: permutation importance on the test period; on IEEE-CIS by
  feature family, since hundreds of correlated columns make per-column
  importance meaningless.

## Calibration

![Calibration](../reports/calibration.png)

Class weighting makes the model over-predict: mid-range scores correspond
to observed fraud rates below the score. Ranking, the cost-tuned flag and
reason codes are unaffected, but the output is a **score, not a
probability**. Wrap the pipeline in `CalibratedClassifierCV` (and re-tune
the threshold) if calibrated probabilities are needed.

## Limitations and risks

- **Threshold anchoring needs a representative window.** The alert-rate
  transfer assumes the anchor window has the same fraud rate and mix as
  deployment. It works with a two-week window (simulated regret €62) and
  fails with six hours of a strongly time-of-day-dependent day (ULB). Use
  an anchor spanning at least one full weekly cycle and monitor alert
  volume daily.
- **Cost-optimal is not capacity-feasible.** With cheap reviews (€5)
  and a 3.4% fraud rate (IEEE-CIS), the cost-optimal threshold flags 26%
  of traffic. Set the threshold from review capacity, or price reviews to
  include analyst time, and report recall at that alert budget.
- **Feature engineering can be redundant.** Behaviour features that are
  the strongest signals on the simulated data add nothing on IEEE-CIS,
  whose vendor columns already encode them. The ablation candidate is kept
  for that reason.
- **Model selection is noisy** when candidates are within fold-to-fold
  variation. Prefer the simpler or more stable model in that case, or
  ensemble the candidates.
- **Stealth fraud is not detected.** Transactions that match the
  customer's own behaviour need other controls (cardholder notifications,
  dispute analytics).
- **Device dependence** (simulated track): device age is the strongest
  signal; device-ID spoofing or shared devices would weaken it.
- **Label propagation** (IEEE-CIS): labels spread across a compromised
  account, so overall scores partly reflect recognising known accounts.
  Hence the unseen-client metrics.
- **Cold start**: new customers have little history, so behaviour features
  are uninformative for their first transactions.
- **Fairness** not assessed: there are no customer demographics. On real
  deployments, alert rates should be compared across customer segments and
  features like `foreign_country` reviewed for disparate impact.
- **Static costs**: fixed review cost and chargeback fee; no lifetime-value
  effect of false declines.

## Maintenance

Monitor daily alert volume, monthly PR-AUC and recall by typology; retrain
when they fall or the typology mix changes. Re-derive the threshold
whenever cost assumptions change:
`fraud-detection train --chargeback-fee <€> --cost-fp <€>`.
