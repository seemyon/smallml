"""
Prediction-Region Analysis and Interpretation Helpers

Domain-agnostic utilities for interpreting conformal prediction regions and
deriving the framework's standard ``confidence_flag`` ("HIGH" / "UNCERTAIN").
These helpers carry no domain assumptions: a prediction set is described purely
in terms of the statistical outcomes it contains.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def classify_set_type(prediction_set: List[int]) -> Tuple[str, str, str]:
    """
    Classify a binary prediction set by type and confidence.

    Parameters
    ----------
    prediction_set : list of int
        e.g. ``[0]``, ``[1]``, ``[0, 1]``, or ``[]``.

    Returns
    -------
    set_type : str
        'singleton_negative', 'singleton_positive', 'doubleton', or 'empty'.
    confidence : str
        'HIGH', 'UNCERTAIN', or 'INVALID'.
    interpretation : str
        Short, domain-agnostic description.

    Examples
    --------
    >>> classify_set_type([1])
    ('singleton_positive', 'HIGH', 'Definitive prediction: outcome = 1')
    >>> classify_set_type([0, 1])
    ('doubleton', 'UNCERTAIN', 'Ambiguous: both outcomes plausible')
    """
    if len(prediction_set) == 0:
        return ("empty", "INVALID", "Invalid: empty set (calibration issue)")
    elif prediction_set == [0]:
        return ("singleton_negative", "HIGH", "Definitive prediction: outcome = 0")
    elif prediction_set == [1]:
        return ("singleton_positive", "HIGH", "Definitive prediction: outcome = 1")
    elif sorted(prediction_set) == [0, 1]:
        return ("doubleton", "UNCERTAIN", "Ambiguous: both outcomes plausible")
    return ("unknown", "INVALID", f"Invalid: unexpected set {prediction_set}")


def confidence_flag_from_set(prediction_set: List[int]) -> str:
    """
    Map a binary conformal set to the standard confidence flag.

    Returns ``"HIGH"`` when the set is a singleton (a definitive prediction) and
    ``"UNCERTAIN"`` otherwise (ambiguous or invalid region).
    """
    return "HIGH" if len(prediction_set) == 1 else "UNCERTAIN"


def confidence_flag_from_width(width: float, threshold: float) -> str:
    """
    Map a per-observation uncertainty *width* to the standard confidence flag.

    Used for regression/count tasks, where there is no singleton notion. The
    width is typically the Bayesian credible-interval width for the observation;
    ``threshold`` is a reference width (e.g. the median credible width on the
    calibration set, computed at fit time).

    Returns ``"HIGH"`` when ``width <= threshold`` (relatively confident), else
    ``"UNCERTAIN"``.
    """
    return "HIGH" if width <= threshold else "UNCERTAIN"


def compute_set_metrics(
    prediction_sets: List[List[int]],
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Aggregate metrics for a collection of binary prediction sets.

    Returns average size, singleton/doubleton/empty fractions, and (if
    ``y_true`` is given) empirical coverage.
    """
    n = len(prediction_sets)
    set_sizes = [len(s) for s in prediction_sets]

    metrics = {
        "avg_size": float(np.mean(set_sizes)),
        "singleton_fraction": float(np.mean([sz == 1 for sz in set_sizes])),
        "doubleton_fraction": float(np.mean([sz == 2 for sz in set_sizes])),
        "empty_fraction": float(np.mean([sz == 0 for sz in set_sizes])),
    }

    if y_true is not None:
        if len(y_true) != n:
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) vs " f"prediction_sets ({n})"
            )
        metrics["coverage"] = float(
            np.mean([y_true[i] in prediction_sets[i] for i in range(n)])
        )

    return metrics
