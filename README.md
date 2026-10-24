# Fraud Detection on Payment Transactions

[![CI](https://github.com/Ehsan-Ghoreishi/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Ehsan-Ghoreishi/fraud-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end fraud detection system for card transactions, built the way a
**fraud / risk analytics team** works: point-in-time customer features,
out-of-time evaluation, a decision threshold set by business cost, a
benchmark against a rule engine, and a reason code on every alert.

> Built by **Seyedmohammad (Ehsan) Ghoreishi** — a data professional focused on
> risk, fraud, and financial analytics.

---

## Results at a glance

Trained on labelled history up to 16 April, deployed on 1 May, evaluated on
**May–June** — two months the model never saw, after fraudsters changed
tactics:

| | Value | 95% CI |
|---|---|---|
| **Attacks stopped** (account takeover / card testing / cloned card) | **99% / 100% / 100%** | |
| **Fraudulent value caught** | **93%** | |
| Fraud value lost before the first alert | 8.6% | |
| Fraud transactions caught (recall) | 83% | 79–87% |
| Precision of alerts | 35% | 30–40% |
| PR-AUC (fraud rate 0.5%) | 0.75 | 0.69–0.79 |
| Analyst workload | 18 alerts/day | |
| **Loss + review cost** | **€9.5k** vs €71.6k with no model | |
| **vs rule engine, same workload** | **88% vs 62% recall** | |

Confidence intervals come from 1,000 bootstrap resamples **of customers**:
fraud arrives in bursts on the same card, so resampling single transactions
would make the intervals look tighter than they are.

---

## Why this project is different

Most fraud portfolio projects score near-perfectly on a random split of a
public dataset. Production fraud doesn't work like that, and this project is
built around the parts that make it hard:

| Reality | How it is handled here |
|---|---|
| "Unusual" is relative to the customer | Behavioural features vs the customer's own history |
| Features must exist at authorisation time | Point-in-time: each row sees only *earlier* transactions — [tested](tests/test_features.py) |
| Models are used on the future | Out-of-time test set + forward-chaining CV |
| Labels arrive weeks late | 14-day label-delay embargo before the test period and inside CV |
| Fraudsters adapt | Typology mix and tactics drift in the test period |
| Not all fraud is reported | Label noise; unreported fraud tracked separately |
| A €900 miss ≠ a €2 miss | Missed-fraud cost = amount + chargeback fee |
| A fraud team counts stopped *attacks* | Detection per attack, transactions and € lost before the first alert |
| Retrained models score on a new scale | Threshold carried over as an alert rate, regret vs hindsight reported |
| "Better than what we have?" | Benchmarked against a rule engine at equal alert volume |
| Analysts need a *why* | Reason codes for every alert |

---

## The data: a simulated card portfolio

Real payment data is confidential, so [`simulation.py`](src/fraud_detection/simulation.py)
simulates **2,500 customers over six months (~258k transactions, 0.53%
fraud)**. Each customer has a spend level, preferred hours, devices (phones
get replaced), online habits and sometimes a trip abroad, so legitimate
traffic overlaps heavily with classic fraud signals. Fraud comes in four
typologies:

| Typology | Footprint |
|---|---|
| **Card testing** | Bursts of €0.50–9 online charges from a bot device, then a gift-card / electronics cash-out |
| **Account takeover** | New device, several purchases at 3–11× the customer's usual spend |
| **Cloned card** | In-store spending far from home, often abroad |
| **Stealth** | One purchase that looks like the customer's own |

From May the mix shifts toward account takeover and card testers raise their
test amounts to dodge small-amount rules.

![EDA Overview](reports/eda_overview.png)

---

## Approach

```
transaction log ──► point-in-time behaviour features ──► out-of-time split
                    (velocity, spend vs own median,     train ≤ 16 Apr │ embargo │ test May–Jun
                     device age, small-charge bursts)   (labels known) │ 17–30 Apr│
                                                                       ▼
   forward-chaining CV (with embargo gap) ──► pick model by PR-AUC ──► cost-optimal
   (3 candidates)                                                      alert rate on
                                                                       recent folds
                                                                       │
   refit on all labelled data ──► place threshold at that alert rate on│the embargo
                                  window's scores (no labels needed)   ▼
   score test once ──► metrics · attack-level detection · rule benchmark ·
                       typology/month breakdown · reason codes · clustered CIs
```

**Models compared** (forward-chaining CV PR-AUC): logistic regression 0.50,
random forest 0.75, **gradient boosting 0.75** — tree ensembles win because
fraud lives in interactions ("new device *and* far above usual spend").

![Precision-Recall Curve](reports/pr_curve.png)

---

## The decision a risk analyst owns: the threshold

A missed fraud costs its **amount plus a €20 chargeback fee**; a false alarm
costs **€5** of analyst time and customer friction.

A subtlety most projects miss: cross-validation models are trained on less,
older history than the final model, so their scores sit on a **different
scale**. Their cost-optimal threshold (0.025) would make the final model
flag 2.15% of traffic instead of the intended 1.26%, 70% more alerts. What does transfer is the cost-optimal
**alert rate** (1.26%). The deployed threshold is the score at that rate on
the final model's predictions for the embargo window, transactions already
seen on deployment day but not yet labelled:

![Cost vs Threshold](reports/cost_curve.png)

| Threshold | Test cost |
|---|---|
| Default 0.5 | €12,775 |
| Raw CV threshold 0.025 | €11,396 |
| **Deployed 0.065 (alert-rate transfer)** | **€9,548** |
| Best possible in hindsight (0.12) | €8,932 |

The remaining **regret is €616**, and the saving against 0.5 is significant
(95% CI €1.1k–€5.8k). Change the assumptions and the pipeline re-derives
everything:

```bash
fraud-detection train --chargeback-fee 40 --cost-fp 12
```

---

## Better than rules?

Four rules a fraud team typically starts with (high amount, new device with
unusual spend, card-testing burst, spending far abroad) raise 2,118 alerts in
the test period. Given **the same 2,118 alerts**, the model catches:

| | Recall | Precision |
|---|---|---|
| Rule engine | 62% | 14% |
| **Model** | **88%** | **19%** |

---

## Where it works — and where it doesn't

Measured per **attack** (all fraud transactions on one card in one
episode), the question a fraud team reports on:

| Typology | Attacks stopped |
|---|---|
| Account takeover | 98.5% |
| Card testing | 100% |
| Cloned card | 100% |
| Stealth | 1.9% |

46% of all attacks are stopped on their **first** fraudulent transaction,
and only 8.6% of fraud value is lost before the first alert. All 29
card-testing attacks in the test period raised an alert during the small
test charges, before the cash-out.

![Recall by Typology](reports/recall_by_typology.png)

Stealth fraud is almost never caught. It looks like the customer's own
spending, so no transaction-level model can separate it. That is a finding
for the business (e.g. cardholder push notifications), not something to
tune away. Performance also dips in May, the first month after the tactics
shift (PR-AUC 0.72 vs 0.78 in June).

---

## Every alert explains itself

Model-agnostic reason codes reset one feature group at a time to a typical
value and report the ones that drive the score
([sample](reports/sample_alerts.csv)):

| Amount | Category | Reasons | Truth |
|---|---|---|---|
| €325.57 | gift cards | recently first-seen device; small test charges in the last hour; large amount | card testing |
| €149.11 | jewelry | far from home; large amount; high spend in the last day | cloned card |
| €278.63 | electronics | recently first-seen device; large amount; unusual amount for this customer | card testing |

Global explanation via permutation importance (device age, merchant risk,
distance and new-device flags lead):

![Feature Importance](reports/feature_importance.png)

---

## Run it yourself

```bash
pip install -e ".[dev]"

make all        # simulate data, EDA, train, evaluate, save model + reports (~1.5 min)
make test       # 58 tests
```

Or step by step:

```bash
fraud-detection generate                        # data/transactions.csv
fraud-detection eda                             # reports/eda_overview.png
fraud-detection train                           # metrics.json, figures, models/fraud_model.joblib
fraud-detection score log.csv -o scored.csv     # fraud_probability, flag, reasons
```

Everything is seeded; `make all` reproduces every number above.

---

## Project structure

```
src/fraud_detection/
├── simulation.py   # customer-level transaction simulator (typologies, drift, label noise)
├── features.py     # point-in-time behaviour features + FraudFeatures transformer
├── validation.py   # out-of-time split, label-delay embargo, forward-chaining CV
├── models.py       # logistic regression, random forest, gradient boosting
├── evaluation.py   # cost curve, threshold transfer, clustered bootstrap, attack-level metrics
├── rules.py        # rule-engine baseline
├── explain.py      # per-alert reason codes
├── model.py        # training & evaluation pipeline
├── scoring.py      # model bundle persistence, batch scoring
├── eda.py · plots.py · cli.py · data.py · config.py
tests/              # 58 tests incl. look-ahead, leakage and embargo checks
notebooks/          # step-by-step walkthrough
reports/            # metrics.json, figures, sample alerts
docs/model_card.md  # intended use, evaluation, limitations
```

See the [model card](docs/model_card.md) for limitations and the
[changelog](CHANGELOG.md) for how the project evolved.

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `pytest` ·
`ruff` · `GitHub Actions`

## Skills demonstrated

Fraud typologies and behavioural feature design · point-in-time feature
engineering · out-of-time and forward-chaining validation with a label-delay
embargo · attack-level detection metrics · threshold transfer · imbalanced
classification · gradient boosting · cost-sensitive decision thresholds ·
rule-engine benchmarking · explainability (reason codes, permutation
importance) · uncertainty quantification (customer-clustered bootstrap) ·
calibration analysis ·
tested, packaged, CI-checked code.

## License

Released under the [MIT License](LICENSE).
