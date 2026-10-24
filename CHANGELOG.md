# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.3.0] — 2026-10-24

First full run on the IEEE-CIS data.

### Added
- **IEEE-CIS results** (`reports/ieee/`): out-of-time test on April–May
  2018, PR-AUC 0.515 (95% CI 0.488–0.542), ROC-AUC 0.892. Clients unseen
  before the test period score PR-AUC 0.559. 90.5% of 2,613 compromised
  accounts are detected.
- **Alert-budget table** for every track (`alert_budget_table`): recall,
  precision and value caught when only the top 0.5% / 1% / 2% / 5% of
  scores are reviewed.

### Findings
- The point-in-time behaviour features add +0.001 CV PR-AUC on IEEE-CIS,
  because Vesta's C/D columns already encode them.
- The cost-optimal threshold flags 26% of IEEE-CIS traffic, so the
  alert-budget view is the operational one.

### Changed
- IEEE-CIS sample alerts are no longer committed, because they hold rows of
  licensed data.

## [1.2.0] — 2026-10-24

Real data. The evaluation protocol now runs on two real datasets next to
the simulated portfolio, and running it on real data exposed three flaws
in the protocol itself, fixed here.

### Added
- **ULB real-data track** (`ulb.py`, `make ulb`): 284,807 European card
  transactions from OpenML 42175; out-of-time test on the last 12 hours.
  Test PR-AUC 0.777 (95% CI 0.705–0.842).
- **IEEE-CIS real-data track** (`ieee/`, `make ieee`): Kaggle download,
  client reconstruction (card1 + addr1 + account start day), point-in-time
  behaviour features on real clients, an ablation without them, metrics on
  clients unseen before the test period, family-level importance and
  reason codes. Tested on an IEEE-CIS-schema fixture; needs a Kaggle
  licence to run.
- `protocol.py`: the evaluation protocol as reusable steps shared by all
  tracks.
- Runner-up models' test scores as diagnostics (`test_pr_auc_diagnostic`)
  and per-fold CV scores (`cv_pr_auc_by_fold`).
- Grouped permutation importance and custom reason groups in `explain.py`.

### Fixed
- **CV scores pooled across folds** mixed fold models' score scales:
  shifting one fold boundary by 188 rows moved CV PR-AUC from 0.75 to 0.59.
  CV is now scored per fold and averaged.
- **The first CV fold model trained on ~7 days** and dominated the
  average. Every fold model now trains on ≥ 50% of the history.
- Periods were measured in whole days (embargo gap, alerts per day), which
  broke on short histories.
- Reason codes listed device age and the new-device flag as two reasons and
  called EUR 2 test charges a "large amount".

### Changed
- Simulated-track results under the corrected protocol: random forest
  selected (CV 0.761 vs 0.753 for gradient boosting); test PR-AUC 0.652,
  recall 82.5%, 92.3% of fraud value caught, threshold regret €62. The
  diagnostic shows gradient boosting at 0.748 on test: the CV margin was
  noise, and the selection rule was deliberately not changed after seeing
  it.

### Known limitation
- Alert-rate threshold anchoring fails on ULB: the 6-hour anchor window had
  a third of the training fraud rate, and the deployed threshold costs
  €6.5k vs €5.1k at 0.5. Anchoring needs a window covering a full cycle.

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
