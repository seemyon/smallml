"""
Layer 3: Conformal Prediction Module

Distribution-free uncertainty quantification for SmallML framework.
Provides coverage guarantees regardless of model correctness.

Author: SmallML Framework
Date: 2025-10-17
"""

from .conformal_predictor import ConformalPredictor
from .prediction_sets import (
    classify_set_type,
    compute_set_metrics,
    interpret_prediction,
    create_decision_matrix_table,
    analyze_prediction_distribution
)

__all__ = [
    'ConformalPredictor',
    'classify_set_type',
    'compute_set_metrics',
    'interpret_prediction',
    'create_decision_matrix_table',
    'analyze_prediction_distribution'
]
