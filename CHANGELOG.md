# Changelog

All notable changes to SmallML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
