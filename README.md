# SmallML: Bayesian Transfer Learning for Small Data

[![PyPI version](https://badge.fury.io/py/smallml.svg)](https://badge.fury.io/py/smallml)
[![Downloads_1](https://static.pepy.tech/badge/smallml)](https://pepy.tech/project/smallml)
[![Downloads_2](https://img.shields.io/pypi/dm/smallml)](https://pypi.org/project/smallml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Reliable predictions with just **50–500 observations per entity** — for *any*
domain.

SmallML is a **domain-agnostic** Bayesian transfer-learning framework. Like
scikit-learn, it doesn't care whether you're predicting match outcomes, species
detection, or customer behavior — it provides the algorithmic infrastructure for
the small-data Bayesian regime. The domain is determined entirely by *your* data,
not by the framework.

It combines three layers:

1. **Layer 1 — Transfer Learning.** Trains a gradient-boosting model (LightGBM)
   on *your* large reference dataset and converts SHAP attributions into Bayesian
   priors (β₀, Σ₀).
2. **Layer 2 — Hierarchical Bayesian.** Pools information across entities via
   PyMC/NUTS partial pooling, so data-poor entities borrow strength.
3. **Layer 3 — Conformal Prediction.** Wraps outputs in distribution-free,
   finite-sample uncertainty guarantees.

## 🎯 Key Features

- **Domain-agnostic**: bring any tabular schema; no built-in domain data.
- **Three tasks**: binary classification, regression, and count (Poisson).
- **Transfer learning from your own reference data** — no assumptions baked in.
- **Hierarchical pooling** across user-defined entities.
- **Uncertainty guarantees**: Bayesian credible intervals + conformal regions.
- **One output contract** — the same five fields for every task and domain.

## 🚀 Quick Start

```bash
pip install smallml
```

```python
from smallml import Pipeline

# 1) A large reference dataset (yours) and 2) a small target dataset with an
#    entity column. Both share feature columns + the target column.
pipe = Pipeline(task="binary")                 # or "regression" / "count"
pipe.fit(
    target_df,
    target_col="outcome",
    entity_col="group",
    reference_data=reference_df,
)

preds = pipe.predict(new_rows, entity_id="group_A")
print(preds[["point_prediction", "conformal_set", "confidence_flag"]])
```

**Tutorials:** [`examples/quickstart.py`](examples/quickstart.py) (binary),
[`examples/regression_quickstart.py`](examples/regression_quickstart.py),
[`examples/count_quickstart.py`](examples/count_quickstart.py).

## 📦 The Standard Output Contract

Every `predict()` call returns the **same five fields**, regardless of task or
domain:

| Field | Description |
|---|---|
| `point_prediction` | Most likely outcome (probability / value / expected count) |
| `posterior_distribution` | Per-row array of posterior samples (compute any statistic) |
| `credible_lower` / `credible_upper` | Bayesian credible interval at `confidence_level` |
| `conformal_set` | Set `{0}/{1}/{0,1}` (binary) or interval `[lo, hi]` (regression/count) |
| `confidence_flag` | `"HIGH"` (definitive) or `"UNCERTAIN"` (ambiguous) |

## 🧪 Tasks

| `task` | Likelihood | Point output | Conformal region |
|---|---|---|---|
| `binary` | Bernoulli (logit) | probability in [0,1] | set from {0, 1} |
| `regression` | Gaussian | value on the real line | interval `[lo, hi]` |
| `count` | Poisson | expected count ≥ 0 | integer interval |

For **regression/count**, conformal intervals are **locally adaptive** via
Conformalized Quantile Regression (CQR) — interval width varies per observation
with the difficulty of the input. When training data is too small to fit the
quantile regressors reliably, the pipeline automatically falls back to
locally-adaptive normalized split conformal (`Pipeline(cqr_min_train=...)`).

## 📊 Data Requirements

SmallML enforces minimums for reliable transfer + hierarchical inference:

- **Reference dataset** ≥ **5×** the combined target size (when provided).
- **Entities** `J` ≥ **5** (a warning is issued below 10). Use
  `Pipeline(allow_single_entity=True)` to override for single-/few-entity use.
- **Observations per entity** `n_j` ≥ **20**.

If the reference and target feature columns differ, SmallML proceeds on their
**intersection** and warns about dropped columns.

## ⚙️ Advanced Usage

```python
# Task-appropriate evaluation metrics
metrics = pipe.evaluate(X_test, y_test, entity_id="group_A")

# MCMC convergence diagnostics (R̂ < 1.01, ESS > 400)
diagnostics = pipe.get_convergence_diagnostics()

# Faster MCMC for prototyping
pipe = Pipeline(task="binary", quick_mode=True)

# Extract priors yourself (Layer 1 standalone)
from smallml import PriorExtractor
priors = PriorExtractor(task="binary").fit(reference_df, "outcome").get_priors()

# Save / load
pipe.save("models/my_pipeline.pkl")
pipe = Pipeline.load("models/my_pipeline.pkl")
```

## 🧰 Requirements

- **Python**: 3.9+
- **Dependencies**: PyMC ≥5.0, ArviZ ≥0.18, pandas ≥2.0, numpy, scikit-learn
  ≥1.3, scipy ≥1.10, LightGBM ≥4.0, SHAP ≥0.44

## ❓ FAQ

**Q: Where does the reference dataset come from?**
A: Always from you. SmallML ships no built-in domain data. If you have no
reference dataset, omit `reference_data` and SmallML uses weakly informative
priors (no transfer learning).

**Q: Does the output change between domains?**
A: No. The five-field contract is identical for every task and domain.

**Q: What if my reference and target columns differ?**
A: SmallML uses their intersection and warns about dropped columns.

**Q: Can I use regression or count targets?**
A: Yes — set `task="regression"` or `task="count"`.

## 🔬 Research & Reproducibility

The paper-reproduction framework lives in `src/` and `scripts/`; technical docs
in `docs/`. Companion paper: https://arxiv.org/abs/2511.14049.

## 🎓 Citation

```bibtex
@software{smallml2025,
  title = {SmallML: Bayesian Transfer Learning for Small-Data Predictive Analytics},
  author = {Leontev, Semen},
  year = {2025},
  url = {https://github.com/seemyon/smallml},
}
```

## 📝 License

MIT — see [LICENSE](LICENSE).

## 🔗 Links

- **GitHub**: https://github.com/seemyon/smallml
- **Issues**: https://github.com/seemyon/smallml/issues
- **Paper**: https://arxiv.org/abs/2511.14049
