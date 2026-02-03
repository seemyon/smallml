"""
SmallML: Bayesian Transfer Learning for Small-Data Predictive Analytics

Build production-grade ML models with just 50-200 observations per entity
by combining transfer learning, hierarchical Bayesian inference, and
conformal prediction.

Quick Start (Business Users)
----------------------------
>>> from smallml import Pipeline
>>> pipeline = Pipeline(use_pretrained_priors=True)
>>> pipeline.fit(my_data, target_col='churned')
>>> predictions = pipeline.predict(new_customers)

Advanced Usage (Researchers)
----------------------------
>>> from smallml import HierarchicalBayesianModel, ConformalPredictor
>>> from smallml import FeatureMatcher, load_pretrained_priors
>>>
>>> # Build custom pipelines with your own priors
>>> model = HierarchicalBayesianModel(beta_0=my_priors['beta_0'], ...)
"""

from .version import __version__, __author__, __description__
from .pipeline import Pipeline

# Expose lower-level components for researchers
from .layer2.hierarchical_model import HierarchicalBayesianModel
from .layer3.conformal_predictor import ConformalPredictor
from .feature_matcher import FeatureMatcher

# Utility function to load priors
def load_pretrained_priors():
    """
    Load pre-trained priors from package data.

    Returns
    -------
    priors : dict
        Dictionary containing:
        - 'beta_0': Prior mean vector (np.ndarray)
        - 'Sigma_0': Prior covariance matrix (np.ndarray)
        - 'feature_names': List of feature names

    Examples
    --------
    >>> from smallml import load_pretrained_priors, HierarchicalBayesianModel
    >>> priors = load_pretrained_priors()
    >>> model = HierarchicalBayesianModel(
    ...     beta_0=priors['beta_0'],
    ...     Sigma_0=priors['Sigma_0']
    ... )
    """
    import pickle
    from pathlib import Path

    priors_path = Path(__file__).parent / 'data' / 'priors_churn.pkl'
    if not priors_path.exists():
        raise FileNotFoundError(
            f"Pre-trained priors not found at {priors_path}. "
            "You may need to train your own priors using the research framework."
        )

    with open(priors_path, 'rb') as f:
        return pickle.load(f)


__all__ = [
    # High-level API (Business Users)
    'Pipeline',

    # Low-level API (Researchers)
    'HierarchicalBayesianModel',
    'ConformalPredictor',
    'FeatureMatcher',
    'load_pretrained_priors',

    # Metadata
    '__version__',
    '__author__',
    '__description__',
]
