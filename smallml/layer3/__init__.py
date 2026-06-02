"""Layer 3: Conformal Prediction"""

from .conformal_predictor import ConformalPredictor
from .prediction_sets import (
    classify_set_type,
    compute_set_metrics,
    confidence_flag_from_set,
    confidence_flag_from_width,
)

__all__ = [
    "ConformalPredictor",
    "classify_set_type",
    "compute_set_metrics",
    "confidence_flag_from_set",
    "confidence_flag_from_width",
]
