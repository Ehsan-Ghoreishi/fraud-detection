# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-10-24

Addresses the questions an experienced fraud reviewer would ask about
1.0.0.

### Added
- **Label-delay embargo**: the 14 days before the test period are not
  used for training (labels would not be known yet at deployment), and
  forward-chaining CV leaves the same gap between every training and
  validation window. `--label-delay-days`.
- **Attack-level metrics**: share of fraud episodes detected (overall and
  per typology), share stopped on the first fraudulent transaction,
  median transactions before the first alert, fraud value lost before
  detection. Needs the new analysis-only `fraud_episode` column.
- **Customer-clustered bootstrap**: confidence intervals resample whole
  customers, since fraud transactions on one card are correlated.
- **Threshold regret**: test cost of the deployed threshold vs the
  hindsight-optimal one, plus the raw CV-scale threshold for comparison.

### Fixed
- **Threshold transfer**: fold models score on a different scale from
  the final model, so the raw CV threshold over-flagged by 70% (regret
  €2.5k). The cost-optimal *alert rate* is now carried over and placed on
  the final model's scores for the unlabelled embargo window (regret
  €616). Tuned on the two most recent folds (`--threshold-folds`).
- Threshold grid is log-spaced at low scores, where optima for rare fraud
  lie.
- The CV embargo is capped only when the folds genuinely don't fit.

### Changed
- Test results under the stricter protocol: PR-AUC 0.748, recall 83.2%,
  precision 35.4%, 92.8% of fraud value caught, 18 alerts/day; account
  takeover / card testing / cloned card attacks detected 98.5% / 100% /
  100%; 88% vs 62% recall against the rule engine.

## [1.0.0] — 2026-10-24

The synthetic data was too easy (ROC-AUC 0.9998) to say anything about
real fraud detection. This release rebuilds the project around realistic
data and production evaluation practice.

### Added
- **Customer-level simulator** (`simulation.py`): customer habits, travel,
  phone changes, shopping bursts; four fraud typologies (card testing,
  account takeover, cloned card, stealth); drift in the last two months;
  unreported fraud.
- **Point-in-time behavioural features**: velocity windows, spend vs the
  customer's own median, device age, small-charge bursts, first purchase
  in a category — with a test proving there is no look-ahead.
- **Out-of-time evaluation** (train Jan–Apr, test May–Jun) and
  **forward-chaining CV** (`validation.py`).
- **Gradient boosting** candidate (now the selected model).
- **Amount-weighted costs**: a missed fraud costs its amount plus a
  chargeback fee; value recall and alerts/day reported.
- **Rule-engine baseline** compared at equal alert volume (`rules.py`).
- **Reason codes** for every alert (`explain.py`), saved with the model,
  added by `score`, and sampled in `reports/sample_alerts.csv`.
- Recall by typology, performance by month and unreported-fraud metrics;
  `recall_by_typology.png`.

### Changed
- Headline results are now realistic: test PR-AUC 0.733, recall 83%,
  precision 40%, 92% of fraud value caught, 87% vs 62% recall against the
  rule engine at the same workload.
- With amount-weighted costs the tuned threshold (0.09) beats 0.5 by
  €3.2k on the test period (95% CI €1.3k–€5.5k).
- CLI: `generate` takes `--customers/--days/--fraud-episode-rate`;
  `train` takes `--chargeback-fee`, `--test-start`, `--models`
  (`--cost-fn` removed).
- EDA, notebook, model card and README rewritten.

### Removed
- The independent-rows generator `generate_transactions()`.

### Fixed
- Undefined monthly metrics are written as `null`, not NaN.
- Forward-chaining folds without fraud are skipped instead of crashing.

## [0.3.0] — 2026-10-24

### Added
- 95% bootstrap confidence intervals for PR-AUC, recall, precision and the
  tuned-vs-default cost difference (`confidence_intervals_95`).
- Calibration analysis: Brier score in `metrics.json` and
  `reports/calibration.png`.
- MIT license, pre-commit hooks, Dependabot.

### Fixed
- `metrics.json` ends with a newline.

## [0.2.0] — 2026-10-24

### Fixed
- **Threshold tuned on the test set.** The cost-optimal threshold is now
  chosen on out-of-fold training predictions and applied unchanged to the
  test set. Reported threshold moves from 0.65 to 0.77.
- **Model selected on the test set.** Candidates are compared by 5-fold
  cross-validated PR-AUC on the training split; only the winner is scored
  on test.
- **Feature leakage.** The high-amount cut-off in `device_geo_amount_risk`
  was the median of the full dataset; it is now learned on training data
  by the `FraudFeatures` transformer.
- Permutation importances are computed on the engineered features the
  classifier actually sees.
- Importing the EDA/plot modules no longer disables inline notebook plots.

### Added
- Installable `fraud_detection` package (`pyproject.toml`, src layout).
- `fraud-detection` CLI: `generate`, `eda`, `train`, `score`, with
  `--cost-fn` / `--cost-fp` to rerun the threshold analysis.
- Model bundle persistence (pipeline + threshold) and batch scoring.
- `business_cost` block in `metrics.json`: test-set cost at the chosen
  threshold vs the default 0.5.
- pytest suite, ruff configuration, GitHub Actions CI, Makefile.
- Model card (`docs/model_card.md`).

### Changed
- Headline test metrics: recall 98.8% → 97.6%, precision 77.8% → 84.1%,
  flagged traffic 2.2% → 2.0%. The earlier numbers were optimistically
  biased by the leakage above.
- Cost curve figure shows the training (tuning) and test curves side by
  side, per 1,000 transactions.
- Threshold search is vectorised and takes costs as arguments.
- Library code logs via `logging` instead of `print`.

## [0.1.0] — 2026-07-05

- Initial pipeline: synthetic data, EDA, logistic regression vs random
  forest, cost-based threshold, report figures and notebook.
