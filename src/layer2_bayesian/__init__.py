"""
Layer 2: Hierarchical Bayesian Core

This module implements the hierarchical Bayesian layer of the SmallML framework,
which pools statistical strength across multiple SMEs through partial pooling.
"""

from .sme_data_generator import SMEDataGenerator
from .hierarchical_model import HierarchicalBayesianModel

__all__ = [
    "SMEDataGenerator",
    "HierarchicalBayesianModel",
]
