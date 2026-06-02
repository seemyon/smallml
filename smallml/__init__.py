"""
SmallML: Bayesian Transfer Learning for Small-Data Predictive Analytics

Build production-grade ML models with just 50-200 observations per entity
by combining transfer learning, hierarchical Bayesian inference, and
conformal prediction.
"""

from .version import __version__, __author__, __description__
from .pipeline import Pipeline
from .layer1 import PriorExtractor
from .preprocessing import FeaturePreprocessor, align_features

__all__ = [
    "Pipeline",
    "PriorExtractor",
    "FeaturePreprocessor",
    "align_features",
    "__version__",
    "__author__",
    "__description__",
]
