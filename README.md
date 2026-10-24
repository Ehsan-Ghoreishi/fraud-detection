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

Trained on January–April, evaluated on **May–June** — two months the model
never saw, after fraudsters changed tactics:

| | Value | 95% CI |
|---|---|---|
| **Fraud transactions caught** (recall) | **83%** | 80–86% |
| **Fraudulent value caught** | **92%** | |
| Precision of alerts | 40% | 38–42% |
| PR-AUC (fraud rate 0.5%) | 0.73 | 0.70–0.77 |
| Analyst workload | 16 alerts/day | |
| **Loss + review cost** | **€9.3k** vs €71.6k with no model | |
| **vs rule engine, same workload** | **87% vs 62% recall** | |

Confidence intervals come from 1,000 stratified bootstrap resamples of the
test period.

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
| Fraudsters adapt | Typology mix and tactics drift in the test period |
| Not all fraud is reported | Label noise; unreported fraud tracked separately |
| A €900 miss ≠ a €2 miss | Missed-fraud cost = amount + chargeback fee |
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
                    (velocity, spend vs own median,       train Jan–Apr │ test May–Jun
                     device age, small-charge bursts)                   │
                                                                        ▼
         forward-chaining CV ──► pick model by PR-AUC ──► tune threshold on
         (3 candidates)                                    out-of-fold cost
                                                                        │
                                                                        ▼
           score test once ──► metrics · rule benchmark · typology/month breakdown
                                · reason codes · bootstrap CIs
```

**Models compared** (forward-chaining CV PR-AUC): logistic regression 0.49,
random forest 0.72, **gradient boosting 0.75** — tree ensembles win because
fraud lives in interactions ("new device *and* far above usual spend").

![Precision-Recall Curve](reports/pr_curve.png)

---

## The decision a risk analyst owns: the threshold

A missed fraud costs its **amount plus a €20 chargeback fee**; a false alarm
costs **€5** of analyst time and customer friction. The threshold that
minimises that cost is found on out-of-fold training predictions and then
frozen:

![Cost vs Threshold](reports/cost_curve.png)

The cost-optimal threshold is **0.09**, far below the default 0.5. On the
test period it saves **€3.2k** compared with 0.5 (95% CI €1.3k–€5.5k, so
not noise). Change the assumptions and the pipeline re-derives it:

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
| **Model** | **87%** | **19%** |

---

## Where it works — and where it doesn't

![Recall by Typology](reports/recall_by_typology.png)

Stealth fraud is almost never caught. It looks like the customer's own
spending, so no transaction-level model can separate it. That is a finding
for the business (e.g. cardholder push notifications), not something to
tune away. Performance also dips in May, the first month after the tactics
shift (PR-AUC 0.71 vs 0.76 in June).

---

## Every alert explains itself

Model-agnostic reason codes reset one feature group at a time to a typical
value and report the ones that drive the score
([sample](reports/sample_alerts.csv)):

| Amount | Category | Reasons | Truth |
|---|---|---|---|
| €325.57 | gift cards | recently first-seen device; small test charges in the last hour; unusual amount for this customer | card testing |
| €149.11 | jewelry | far from home; large amount; risky merchant category | cloned card |
| €269.43 | travel | recently first-seen device; high spend in the last day; large amount | account takeover |

Global explanation via permutation importance (device age, merchant risk,
distance and new-device flags lead):

![Feature Importance](reports/feature_importance.png)

---

## Run it yourself

```bash
pip install -e ".[dev]"

make all        # simulate data, EDA, train, evaluate, save model + reports (~1.5 min)
make test       # 45 tests
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
├── validation.py   # out-of-time split, forward-chaining CV
├── models.py       # logistic regression, random forest, gradient boosting
├── evaluation.py   # cost curve, threshold, bootstrap CIs, value recall
├── rules.py        # rule-engine baseline
├── explain.py      # per-alert reason codes
├── model.py        # training & evaluation pipeline
├── scoring.py      # model bundle persistence, batch scoring
├── eda.py · plots.py · cli.py · data.py · config.py
tests/              # 45 tests incl. look-ahead and leakage checks
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
engineering · out-of-time and forward-chaining validation · imbalanced
classification · gradient boosting · cost-sensitive decision thresholds ·
rule-engine benchmarking · explainability (reason codes, permutation
importance) · uncertainty quantification (bootstrap) · calibration analysis ·
tested, packaged, CI-checked code.

## License

Released under the [MIT License](LICENSE).
