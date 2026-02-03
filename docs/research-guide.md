# SmallML Research Guide

This guide is for **researchers and ML engineers** who want to:
- Apply SmallML to new domains (healthcare, finance, manufacturing, etc.)
- Train custom priors on their own large datasets
- Understand the underlying methodology
- Extend or modify the framework

For business users who just want to predict churn, see [Business Quickstart](business-quickstart.md).

## Architecture Overview

SmallML implements a three-layer Bayesian transfer learning framework:

```
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 1: TRANSFER LEARNING (Your responsibility in custom domains) │
│ - Train CatBoost/XGBoost on large public dataset                   │
│ - Extract SHAP-based priors (β₀, Σ₀)                               │
│ - These priors encode "what features matter" in your domain        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    Prior extraction
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 2: HIERARCHICAL BAYESIAN INFERENCE (HierarchicalBayesianModel)│
│ - PyMC NUTS MCMC sampler                                            │
│ - Three-level hierarchy: Population → Entity → Observation          │
│ - Non-centered parameterization for MCMC stability                  │
│ - Partial pooling across entities                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
              Posterior predictive sampling
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 3: CONFORMAL PREDICTION (ConformalPredictor)                 │
│ - Distribution-free uncertainty quantification                      │
│ - Split-conformal calibration                                       │
│ - Finite-sample coverage guarantees                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Applying SmallML to a New Domain

### Step 1: Prepare Your Large Public Dataset

You need a "source" dataset with:
- Similar prediction task to your target domain
- 5,000+ observations (more is better)
- Similar feature types (even if names differ)

Example domains and potential source data:
| Target Domain | Potential Public Sources |
|---------------|-------------------------|
| Healthcare readmission | MIMIC-III, eICU |
| Credit default | Kaggle credit datasets, UCI German Credit |
| Manufacturing defects | NASA turbofan, SECOM |
| Employee attrition | IBM HR Analytics |

### Step 2: Train Layer 1 and Extract Priors

```python
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import shap

# Load your large public dataset
public_data = pd.read_csv('large_public_dataset.csv')
X_public = public_data.drop('target', axis=1)
y_public = public_data['target']

# Train CatBoost (or XGBoost, LightGBM)
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=False
)
model.fit(X_public, y_public)

# Extract SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_public)

# Compute priors from SHAP
# β₀ = mean SHAP importance (direction + magnitude)
beta_0 = np.mean(shap_values, axis=0)

# Σ₀ = diagonal covariance from SHAP variance
# (captures uncertainty in feature importance)
sigma_0_diag = np.var(shap_values, axis=0) + 1e-6  # Add small constant for stability
Sigma_0 = np.diag(sigma_0_diag)

# Save priors
priors = {
    'beta_0': beta_0,
    'Sigma_0': Sigma_0,
    'feature_names': list(X_public.columns)
}

import pickle
with open('my_domain_priors.pkl', 'wb') as f:
    pickle.dump(priors, f)
```

### Step 3: Use Your Priors with HierarchicalBayesianModel

```python
from smallml import HierarchicalBayesianModel, ConformalPredictor
import pickle

# Load your custom priors
with open('my_domain_priors.pkl', 'rb') as f:
    priors = pickle.load(f)

# Prepare your small multi-entity data
# Format: {entity_id: {'X': features_df, 'y': target_df}}
sme_datasets = {
    0: {'X': entity_0_features, 'y': entity_0_target},
    1: {'X': entity_1_features, 'y': entity_1_target},
    2: {'X': entity_2_features, 'y': entity_2_target},
}

# Initialize hierarchical model with YOUR priors
model = HierarchicalBayesianModel(
    beta_0=priors['beta_0'],
    Sigma_0=priors['Sigma_0'],
    tau=2.0,  # Controls expected heterogeneity across entities
    random_seed=42
)

# Fit via MCMC
model.fit(
    sme_datasets,
    chains=4,
    draws=2000,
    tune=1000,
    target_accept=0.90
)

# Check convergence
convergence = model.check_convergence()
print(f"Converged: {convergence['all_ok']}")

# Make predictions
predictions = model.posterior_predictive(
    X_new=new_features.values,
    sme_id=0,  # Which entity to predict for
    n_samples=1000
)

print(f"Predicted probability: {predictions['mean']}")
print(f"90% CI: [{predictions['lower_90']}, {predictions['upper_90']}]")
```

### Step 4: Add Conformal Prediction Layer

```python
from smallml import ConformalPredictor

# Split your data: training + calibration
# (You should have done this before fitting the hierarchical model)

# Get predictions on calibration set
cal_predictions = model.posterior_predictive(X_cal.values, sme_id=0)

# Initialize and calibrate conformal predictor
cp = ConformalPredictor(alpha=0.10)  # 90% coverage target
cp.calibrate(
    y_cal=y_cal.values,
    predictions_cal=cal_predictions['mean']
)

# Get prediction sets for new data
test_predictions = model.posterior_predictive(X_test.values, sme_id=0)
prediction_sets = cp.predict_set(test_predictions['mean'])

# Validate coverage
metrics = cp.validate_coverage(y_test.values, test_predictions['mean'])
print(f"Empirical coverage: {metrics['coverage']:.3f}")
```

## Low-Level API Reference

### HierarchicalBayesianModel

```python
from smallml import HierarchicalBayesianModel

model = HierarchicalBayesianModel(
    beta_0,           # Prior mean vector, shape (p,)
    Sigma_0,          # Prior covariance matrix, shape (p, p)
    tau=2.0,          # Between-entity variance prior scale
    random_seed=42
)

# Methods
model.fit(sme_datasets, chains=4, draws=2000, tune=1000, ...)
model.check_convergence()
model.posterior_predictive(X_new, sme_id, n_samples=1000)
model.extract_posterior_means()
model.save_trace(filepath)
```

### ConformalPredictor

```python
from smallml import ConformalPredictor

cp = ConformalPredictor(
    alpha=0.10,                    # Miscoverage rate (1-α = coverage)
    conservative_adjustment=0.0,   # Inflate threshold for small cal sets
    random_seed=42
)

# Methods
cp.calibrate(y_cal, predictions_cal)
cp.predict_set(predictions, return_sets=True)
cp.validate_coverage(y_test, predictions_test)
cp.pooled_calibration(y_cal_dict, predictions_cal_dict)  # Multi-entity
cp.save_calibration(filepath)
cp.load_calibration(filepath)
```

### FeatureMatcher

```python
from smallml import FeatureMatcher

matcher = FeatureMatcher()

# Match user features to pre-trained feature names
matches = matcher.match_features(
    user_features=['days_since_purchase', 'order_count'],
    pretrained_features=['recency', 'frequency', 'monetary']
)
```

### load_pretrained_priors

```python
from smallml import load_pretrained_priors

# Load bundled churn priors (for reference or comparison)
priors = load_pretrained_priors()
print(f"Prior dimensions: {priors['beta_0'].shape}")
print(f"Features: {priors['feature_names']}")
```

## Mathematical Details

### Hierarchical Model Structure

```
Level 1 (Population):
  μ_industry ~ Normal(β₀, √Σ₀)           # Informed by Layer 1 priors
  σ_industry ~ HalfNormal(τ)             # Between-entity variance

Level 2 (Entity-specific):
  β_j ~ Normal(μ_industry, σ_industry)   # Partial pooling

Level 3 (Observations):
  y_ij ~ Bernoulli(logit⁻¹(β_j^T x_ij))  # Likelihood
```

### Non-Centered Parameterization

To avoid Neal's funnel and improve MCMC convergence:

```
β_j_raw ~ Normal(0, 1)
β_j = μ_industry + σ_industry × β_j_raw  # Deterministic transform
```

### Conformal Prediction

Nonconformity score: `s_i = |y_i - p̂_i|`

Calibration: `q̂ = ⌈(n_cal + 1)(1 - α) / n_cal⌉-th quantile of scores`

Prediction set: `C(x) = {y : |y - p̂(x)| ≤ q̂}`

## Tips for Researchers

1. **Prior quality matters**: Garbage priors → garbage posterior. Invest time in Layer 1.

2. **Feature alignment**: Your small dataset features must align with prior features. Use FeatureMatcher or manual mapping.

3. **Check convergence**: Always verify R̂ < 1.01 and ESS > 400 before trusting results.

4. **Calibration set size**: Conformal prediction needs 50+ calibration samples for stable coverage.

5. **Single-entity mode**: With J=1, hierarchical pooling is absent. Priors become critical.

## Citation

If you use SmallML in your research:

```bibtex
@software{smallml2025,
  title = {SmallML: Bayesian Transfer Learning for Small-Data Predictive Analytics},
  author = {Leontev, Semen},
  year = {2025},
  url = {https://github.com/seemyon/smallml},
}
```

## Questions?

- **GitHub Issues**: https://github.com/seemyon/smallml/issues
- **Paper**: https://arxiv.org/abs/2511.14049
