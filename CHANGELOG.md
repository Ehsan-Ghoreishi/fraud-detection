# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
