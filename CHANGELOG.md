# Changelog

All notable changes to SmallML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-01

Domain-agnostic refactor. SmallML is now a general-purpose Bayesian
transfer-learning framework: the domain is determined entirely by the user's
input data, not by internal assumptions. **This is a breaking release.**

### Added
- **Layer 1 in the package**: `PriorExtractor` (LightGBM + SHAP) extracts priors
  (β₀, Σ₀) from a user-supplied reference dataset. New core deps: `lightgbm`,
  `shap`.
- **Selectable likelihood / task**: `Pipeline(task=...)` supports `"binary"`,
  `"regression"` (Gaussian), and `"count"` (Poisson).
- **Locally-adaptive conformal intervals** for regression/count: Conformalized
  Quantile Regression (CQR, `smallml.layer3.cqr.ConformalQuantileRegressor`) is
  the default; when there is too little training data to fit the quantile
  regressors reliably (`< cqr_min_train` pooled rows), the pipeline falls back to
  locally-adaptive **normalized split conformal** (scaled by the Bayesian
  posterior std). Both yield per-observation interval widths, so `confidence_flag`
  reflects genuine local uncertainty. Configurable via `Pipeline(cqr_min_train=...)`.
- **Standard five-field output contract** for every task/domain:
  `point_prediction`, `posterior_distribution`, `credible_lower`/`credible_upper`,
  `conformal_set`, `confidence_flag`.
- **`FeaturePreprocessor`** and `align_features` for arbitrary schemas (scaling,
  categorical encoding, imputation, reference/target intersection with warnings).
- **User-defined entities** via `entity_col` (single DataFrame) or a dict.
- **Minimum-data enforcement**: reference ≥ 5× target, J ≥ 5 (warn < 10),
  n_j ≥ 20.

### Changed
- Public `fit`/`predict` interface preserved, but parameters are now generic and
  domain-neutral: `data` (was `sme_data`), required `target_col` (no `churned`
  default), `entity_col`/`entity_id` (were `sme_id`), `reference_data`/`priors`.
- Conformal predictor is task-aware: prediction *sets* (binary) or *intervals*
  (regression/count).
- All domain-specific language (`sme`, `churn`, `industry`, `customer`,
  `business`; `mu_industry`→`mu_population`, `sigma_industry`→`sigma_population`)
  removed from the codebase.

### Removed
- Built-in churn domain data (`priors_churn.pkl`, `feature_aliases.json`) — the
  framework ships no domain data.
- `use_pretrained_priors` flag (replaced by `reference_data` / `priors`).

### Migration
- Replace `Pipeline().fit(sme_data, target_col='churned')` with
  `Pipeline(task='binary').fit(data, target_col='...', entity_col='...',
  reference_data=...)`.
- Read predictions from the new contract columns (e.g. `point_prediction`
  instead of `prediction`).

## [0.1.4] - 2025-12-14

### Fixed
- Relaxed dependency version constraints for better compatibility
- numpy: Changed from >=2.3.0 to >=1.24.0,<2.2.0 (compatible with TensorFlow, Google Colab)
- pandas: Changed from >=2.3.0 to >=2.0.0 (compatible with Google Colab)
- scikit-learn: Changed from >=1.7.0 to >=1.3.0 (wider compatibility)
- scipy: Changed from >=1.16.0 to >=1.10.0 (wider compatibility)
- arviz: Changed from >=0.22.0 to >=0.18.0 (wider compatibility)

## [0.1.3] - 2025-12-14

### Changed
- Added shields.io download badge for faster data display
- Both pepy.tech and shields.io badges now shown for comparison

## [0.1.2] - 2025-12-14

### Changed
- Added download statistics badge to README
- Improved README formatting and documentation
- Updated GitHub URLs for consistency

## [0.1.1] - 2025-12-14

### Fixed
- Added missing `dependencies` section to pyproject.toml
- Package dependencies (pandas, numpy, pymc, etc.) now install automatically with the package

## [0.1.0] - 2025-12-10

### Added
- Initial release of SmallML package
- `Pipeline` class for end-to-end Bayesian predictive analytics
- Hierarchical Bayesian inference (Layer 2) using PyMC
  - Multi-entity pooling for sharing statistical strength
  - Informed priors from transfer learning
  - Automatic MCMC convergence validation (R̂ < 1.01, ESS > 400)
- Conformal prediction (Layer 3) for distribution-free uncertainty
  - Split-conformal calibration
  - Prediction sets with coverage guarantees
- Pre-trained priors from 100K+ public observations
- Comprehensive documentation and examples
- Unit test suite with 20+ tests
- Validation test using synthetic research datasets

### Features
- **Small Data Optimization**: Works with 50-200 observations per entity
- **Multi-Entity Learning**: Pools information across 3-10 business entities
- **Uncertainty Quantification**: Bayesian credible intervals + conformal prediction sets
- **Production Ready**: <30 min training, <100ms inference
- **Automatic Validation**: Built-in convergence checks and performance metrics
- **Easy API**: Just 5 lines from data to predictions

### Documentation
- README.md with quickstart guide
- examples/quickstart.py with complete workflow
- API documentation via docstrings
- PACKAGE_STATUS.md tracking implementation progress

### Dependencies
- Python ≥3.9
- PyMC ≥5.0.0
- ArviZ ≥0.22.0
- pandas ≥2.3.0
- numpy ≥2.3.0
- scikit-learn ≥1.7.0
- scipy ≥1.16.0

[0.1.0]: https://github.com/seemyon/smallml/releases/tag/v0.1.0
