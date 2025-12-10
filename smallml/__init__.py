"""
SmallML: Bayesian Transfer Learning for Small-Data Predictive Analytics

Build production-grade ML models with just 50-200 observations per entity
by combining transfer learning, hierarchical Bayesian inference, and
conformal prediction.
"""

from .version import __version__, __author__, __description__
from .pipeline import SmallMLPipeline

__all__ = [
    'SmallMLPipeline',
    '__version__',
    '__author__',
    '__description__',
]
