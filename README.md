# SmallML: Bayesian Transfer Learning for Small Data

[![PyPI version](https://badge.fury.io/py/smallml.svg)](https://badge.fury.io/py/smallml)
[![Downloads_1](https://static.pepy.tech/badge/smallml)](https://pepy.tech/project/smallml)
[![Downloads_2](https://img.shields.io/pypi/dm/smallml)](https://pypi.org/project/smallml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Build production-grade machine learning models with just **50-200 observations per business entity**.

SmallML combines transfer learning, hierarchical Bayesian inference, and conformal prediction to enable SMEs to achieve reliable predictive analytics despite limited data.

## 🎯 Key Features

- **Works with tiny datasets**: 50-200 observations per entity, 3-10 entities total
- **Single-entity mode**: Have only one store? No problem - works with just one dataset
- **Transfer learning**: Extracts knowledge from 100K+ public observations (pre-trained priors included)
- **Hierarchical pooling**: Shares statistical strength across multiple business entities
- **Uncertainty guarantees**: Bayesian credible intervals + distribution-free prediction sets
- **Production-ready**: <30 minutes training, <100ms inference, automatic convergence validation

## 🚀 Quick Start

### Installation

```bash
pip install smallml
```
### Development Version

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/seemyon/smallml@main
```

### Basic Usage (5 lines of code!)

```python
from smallml import Pipeline
import pandas as pd

# Your data: dict of {entity_name: dataframe}
sme_data = {
    'store_1': pd.read_csv('store_1.csv'),  # 80 customers
    'store_2': pd.read_csv('store_2.csv'),  # 120 customers
    'store_3': pd.read_csv('store_3.csv'),  # 95 customers
    # ... 3-10 stores total
}

# Create and fit pipeline (automatically validates convergence)
pipeline = Pipeline()
pipeline.fit(sme_data, target_col='churned')

# Make predictions with uncertainty
predictions = pipeline.predict(new_customers, sme_id='store_1')
print(predictions)
#    prediction  bayesian_std  bayesian_lower_90  bayesian_upper_90  conformal_set  conformal_set_size
# 0       0.23          0.12                0.04               0.42            {0}                   1
# 1       0.78          0.15                0.51               0.95            {1}                   1
# 2       0.51          0.21                0.18               0.84          {0,1}                   2  # Uncertain!
```

**[See full tutorial →](examples/quickstart.py)**

### Single-Entity Mode (New!)

Have only one business location? SmallML now supports single-entity mode for users with just one dataset:

```python
from smallml import Pipeline
import pandas as pd

# Load your single dataset (e.g., your one coffee shop)
my_customers = pd.read_csv('my_customers.csv')

# Create pipeline with pre-trained priors (strongly recommended for single-entity)
pipeline = Pipeline(use_pretrained_priors=True)

# Pass DataFrame directly - no dict needed!
pipeline.fit(my_customers, target_col='churned')

# Predict - no sme_id needed!
predictions = pipeline.predict(new_customers)
```

**Key differences from multi-entity mode:**
- Pre-trained priors are strongly recommended (they provide the statistical strength that pooling would offer)
- Accuracy may be 5-10% lower than multi-entity mode
- As your business grows (more locations), you can switch to multi-entity mode for better accuracy

**[See single-entity tutorial →](examples/single_entity_quickstart.py)**

## 📚 How It Works

SmallML uses a two-layer architecture:

1. **Layer 2 (Hierarchical Bayesian)**: Pools information across J entities using PyMC NUTS sampler
   - Uses pre-trained priors from 100K+ public observations
   - Returns full posterior distributions, not just point estimates
   - Automatic convergence validation (R̂ < 1.01, ESS > 400)

2. **Layer 3 (Conformal Prediction)**: Provides distribution-free uncertainty
   - Split-conformal calibration for coverage guarantees
   - Returns prediction sets: {0} (certain), {1} (certain), or {0,1} (uncertain)
   - Empirical coverage typically 87-93% for 90% target

## 📊 Performance Expectations

- **Prediction Accuracy**: 75-85% AUC on churn with 100 customers per entity
- **Conformal Coverage**: 87-93% empirical for 90% target intervals
- **Training Time**: 15-30 minutes for J=5 entities with 100 customers each
- **Inference**: <100ms per prediction
- **Convergence**: R̂ < 1.01, ESS > 400 (automatically validated)

## 🧪 Requirements

### Data Requirements

**Multi-Entity Mode (Recommended):**
- **Minimum**: 3 entities, 30 observations per entity
- **Recommended**: 5+ entities, 50+ observations per entity

**Single-Entity Mode:**
- **Minimum**: 1 entity, 50 observations
- **Recommended**: Use pre-trained priors for best accuracy

**General:**
- **Use Case**: Binary classification (churn, conversion, etc.)
- **Features**: Numerical + categorical (automatically handled)

### Input Format
```python
sme_data = {
    'entity_1': pd.DataFrame({
        'feature_1': [...],      # Numerical or categorical
        'feature_2': [...],
        'feature_3': [...],
        'churned': [0, 1, 0, ...]  # Binary target (0/1)
    }),
    'entity_2': pd.DataFrame({...}),
    # ... 3-10 entities
}
```

### Python Requirements
- **Python**: 3.9 or higher
- **Dependencies**: PyMC ≥5.0, ArviZ ≥0.22.0, pandas ≥2.3, numpy ≥2.3, scikit-learn ≥1.7, scipy ≥1.16

## 📖 Documentation

- **Installation Guide**: See above for basic installation
- **Quickstart Tutorial**: `examples/quickstart.py`
- **API Reference**: Check docstrings in `smallml.pipeline.Pipeline`
- **Research Paper**: See `docs/` for technical details

## 🔬 Research & Reproducibility

This package is the production-ready version of the SmallML research framework. For research code, paper reproduction, and detailed technical documentation, see:
- **Research Code**: `src/` directory
- **Reproduction Scripts**: `scripts/` directory
- **Technical Docs**: `docs/` directory
- **Original README**: See existing README.md for research details

## 🎓 Citation

If you use SmallML in your research, please cite:

```bibtex
@software{smallml2025,
  title = {SmallML: Bayesian Transfer Learning for Small-Data Predictive Analytics},
  author = {Leontev, Semen},
  year = {2025},
  url = {https://github.com/seemyon/smallml},
}
```

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🔗 Links

- **GitHub**: https://github.com/seemyon/smallml
- **Issues**: https://github.com/seemyon/smallml/issues
- **Paper**: https://arxiv.org/abs/2511.14049

---

**SmallML: Empowering small businesses with reliable ML despite limited data.**

## ⚙️ Advanced Usage

### Evaluating Model Performance

```python
# Evaluate on test data
X_test, y_test = load_test_data()
metrics = pipeline.evaluate(X_test, y_test, sme_id='store_1')

print(f"AUC: {metrics['auc']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
print(f"Conformal Coverage: {metrics['conformal_coverage']:.3f}")  # Should be ~0.90
print(f"Mean Set Size: {metrics['mean_set_size']:.2f}")  # 1.0 = certain, 2.0 = uncertain
```

### Checking MCMC Convergence

```python
# Get convergence diagnostics
diagnostics = pipeline.get_convergence_diagnostics()
print(diagnostics)
#        parameter  r_hat    ess_bulk  ess_tail
# 0       mu[0]     1.003    1845      2103
# 1       mu[1]     1.002    1923      2247
# ...

# All R̂ should be < 1.01, ESS should be > 400
```

### Saving and Loading Pipelines

```python
# Save fitted pipeline
pipeline.save('models/my_pipeline.pkl')

# Load later
from smallml import Pipeline
pipeline = Pipeline.load('models/my_pipeline.pkl')
predictions = pipeline.predict(new_data)
```

### Quick Mode for Prototyping

```python
# Faster MCMC (fewer iterations) for testing
pipeline = Pipeline(quick_mode=True)
pipeline.fit(sme_data, target_col='churned')  # Takes ~5-10 min instead of 15-30

# For production, use default settings:
pipeline = Pipeline(quick_mode=False)  # More reliable convergence
```

## ❓ FAQ

**Q: What if I don't have pre-trained priors?**
A: You'll need to add your own priors to `smallml/data/priors_churn.pkl`. The package structure is ready, and you can copy your existing priors there. The file should contain `{'beta_0': np.ndarray, 'Sigma_0': np.ndarray}`.

**Q: Can I use this for regression instead of classification?**
A: Currently SmallML focuses on binary classification. Regression support is planned for future versions.

**Q: What if MCMC doesn't converge?**
A: The pipeline automatically validates convergence. If it fails, try:
- Use `quick_mode=False` for more MCMC iterations
- Ensure you have at least 50 observations per entity
- Check that features are properly normalized

**Q: How do I interpret conformal sets?**
A:
- `{0}` = Certain prediction: will NOT churn
- `{1}` = Certain prediction: WILL churn
- `{0,1}` = Uncertain prediction: could go either way

**Q: Can I use this with just one entity (one store/business)?**
A: Yes! SmallML now supports single-entity mode. Just pass your DataFrame directly to `fit()` and use `Pipeline(use_pretrained_priors=True)` for best results. See the [single-entity tutorial](examples/single_entity_quickstart.py).

**Q: What's the difference between single-entity and multi-entity mode?**
A: Multi-entity mode pools statistical strength across your stores/branches, improving accuracy. Single-entity mode relies on pre-trained priors instead. Expect ~5-10% lower accuracy with single-entity, but it's still useful for small businesses with one location.

**Q: Can I use this with just 2 entities?**
A: The package will warn but still work. Consider either: (a) adding more entities for better pooling, or (b) using single-entity mode with pre-trained priors if pooling benefit is minimal.
