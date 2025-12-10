"""
Prediction Set Analysis and Interpretation Helpers

Utility functions for analyzing conformal prediction sets and providing
business decision guidance.

"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def classify_set_type(
    prediction_set: List[int]
) -> Tuple[str, str, str]:
    """
    Classify prediction set by type and provide interpretation.

    Parameters
    ----------
    prediction_set : list of int
        Prediction set, e.g., [0], [1], [0, 1], or []

    Returns
    -------
    set_type : str
        Type classification: 'singleton_negative', 'singleton_positive',
        'doubleton', or 'empty'
    confidence_level : str
        'high', 'uncertain', or 'invalid'
    interpretation : str
        Business-friendly interpretation

    Examples
    --------
    >>> classify_set_type([1])
    ('singleton_positive', 'high', 'High confidence: Customer will churn')

    >>> classify_set_type([0, 1])
    ('doubleton', 'uncertain', 'Uncertain: Both outcomes plausible')
    """
    if len(prediction_set) == 0:
        return (
            'empty',
            'invalid',
            'Invalid: Empty set (calibration issue)'
        )
    elif prediction_set == [0]:
        return (
            'singleton_negative',
            'high',
            'High confidence: Customer will not churn'
        )
    elif prediction_set == [1]:
        return (
            'singleton_positive',
            'high',
            'High confidence: Customer will churn'
        )
    elif sorted(prediction_set) == [0, 1]:
        return (
            'doubleton',
            'uncertain',
            'Uncertain: Both outcomes plausible'
        )
    else:
        return (
            'unknown',
            'invalid',
            f'Invalid: Unexpected set {prediction_set}'
        )


def compute_set_metrics(
    prediction_sets: List[List[int]],
    y_true: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute aggregate metrics for collection of prediction sets.

    Parameters
    ----------
    prediction_sets : list of lists
        Collection of prediction sets
    y_true : np.ndarray, optional
        True labels for coverage calculation

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - 'avg_size': Average set size
        - 'singleton_fraction': Fraction of singleton sets
        - 'doubleton_fraction': Fraction of doubleton sets
        - 'empty_fraction': Fraction of empty sets
        - 'coverage': Empirical coverage (if y_true provided)
        - 'singleton_negative_fraction': Fraction {0}
        - 'singleton_positive_fraction': Fraction {1}

    Examples
    --------
    >>> sets = [[0], [1], [0, 1], [0], [1]]
    >>> metrics = compute_set_metrics(sets)
    >>> metrics['avg_size']
    1.2
    """
    n = len(prediction_sets)
    set_sizes = [len(s) for s in prediction_sets]

    metrics = {
        'avg_size': np.mean(set_sizes),
        'singleton_fraction': np.mean([sz == 1 for sz in set_sizes]),
        'doubleton_fraction': np.mean([sz == 2 for sz in set_sizes]),
        'empty_fraction': np.mean([sz == 0 for sz in set_sizes]),
        'singleton_negative_fraction': np.mean([s == [0] for s in prediction_sets]),
        'singleton_positive_fraction': np.mean([s == [1] for s in prediction_sets]),
    }

    # Compute coverage if labels provided
    if y_true is not None:
        if len(y_true) != n:
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) vs "
                f"prediction_sets ({n})"
            )
        coverage_indicators = [
            y_true[i] in prediction_sets[i]
            for i in range(n)
        ]
        metrics['coverage'] = np.mean(coverage_indicators)

    return metrics


def interpret_prediction(
    prediction_set: List[int],
    predicted_probability: float,
    customer_id: Optional[str] = None,
    include_action: bool = True
) -> Dict[str, str]:
    """
    Provide business-friendly interpretation of prediction set.

    Parameters
    ----------
    prediction_set : list of int
        Conformal prediction set
    predicted_probability : float
        Bayesian predicted churn probability
    customer_id : str, optional
        Customer identifier for personalized message
    include_action : bool, optional (default=True)
        Include recommended action in interpretation

    Returns
    -------
    interpretation : dict
        Dictionary with keys:
        - 'set_type': Classification of set
        - 'confidence': 'high', 'uncertain', or 'invalid'
        - 'risk_level': 'high', 'medium', or 'low'
        - 'message': Human-readable interpretation
        - 'action': Recommended business action (if include_action=True)
        - 'priority': Action priority ('highest', 'medium', 'lowest')

    Examples
    --------
    >>> interpret_prediction([1], 0.95, customer_id="Alice")
    {
        'set_type': 'singleton_positive',
        'confidence': 'high',
        'risk_level': 'high',
        'message': 'Alice has 95% predicted churn probability. ...',
        'action': 'Immediate retention intervention',
        'priority': 'highest'
    }
    """
    set_type, confidence, base_interp = classify_set_type(prediction_set)

    # Determine risk level
    if set_type == 'singleton_positive':
        risk_level = 'high'
        priority = 'highest'
        action = (
            "Immediate retention intervention (discount offer, "
            "personal outreach, satisfaction survey)"
        )
    elif set_type == 'doubleton':
        risk_level = 'medium'
        priority = 'medium'
        action = (
            "Gather more information: Deploy in-app survey, monitor "
            "engagement metrics closely, consider A/B test"
        )
    elif set_type == 'singleton_negative':
        risk_level = 'low'
        priority = 'lowest'
        action = "Standard engagement (no special intervention needed)"
    else:  # empty or unknown
        risk_level = 'invalid'
        priority = 'none'
        action = "Recalibrate model before taking action"

    # Build message
    if customer_id:
        customer_ref = f"{customer_id}"
    else:
        customer_ref = "Customer"

    if set_type == 'singleton_positive':
        message = (
            f"{customer_ref} has {predicted_probability:.1%} predicted "
            f"churn probability. Conformal prediction confirms high risk: "
            f"prediction set = {{1}} (definitive churn prediction). "
            f"This customer is highly likely to churn."
        )
    elif set_type == 'singleton_negative':
        message = (
            f"{customer_ref} has {predicted_probability:.1%} predicted "
            f"churn probability. Conformal prediction confirms low risk: "
            f"prediction set = {{0}} (definitive non-churn prediction). "
            f"This customer is unlikely to churn."
        )
    elif set_type == 'doubleton':
        message = (
            f"{customer_ref} has {predicted_probability:.1%} predicted "
            f"churn probability. Conformal prediction indicates uncertainty: "
            f"prediction set = {{0, 1}} (both outcomes plausible). "
            f"The model cannot confidently predict this customer's behavior. "
            f"Recommend gathering more data before acting."
        )
    else:
        message = (
            f"{customer_ref}: Invalid prediction set {prediction_set}. "
            f"This indicates a calibration issue. Do not act on this prediction."
        )

    interpretation = {
        'set_type': set_type,
        'confidence': confidence,
        'risk_level': risk_level,
        'message': message,
        'priority': priority
    }

    if include_action:
        interpretation['action'] = action

    return interpretation


def create_decision_matrix_table() -> pd.DataFrame:
    """
    Create Table 4.14: Prediction Set-Based Decision Matrix.

    Returns
    -------
    table : pd.DataFrame
        Decision matrix mapping prediction sets to actions

    Examples
    --------
    >>> table = create_decision_matrix_table()
    >>> print(table.to_markdown())
    """
    data = {
        'Prediction Set': ['{1}', '{0, 1}', '{0}', '∅'],
        'Churn Risk': ['High', 'Uncertain', 'Low', 'Invalid'],
        'Recommended Action': [
            'Immediate retention campaign',
            'Gather more data, monitor closely',
            'Standard engagement',
            'Recalibrate model'
        ],
        'Resource Allocation': [
            'High budget',
            'Medium budget',
            'Minimal budget',
            'No action'
        ],
        'Expected Precision': ['80-90%', '50-60%', '85-95%', 'N/A']
    }

    return pd.DataFrame(data)


def analyze_prediction_distribution(
    prediction_sets: List[List[int]],
    predictions: np.ndarray,
    q_hat: float
) -> Dict[str, any]:
    """
    Analyze distribution of prediction sets across confidence continuum.

    Parameters
    ----------
    prediction_sets : list of lists
        Collection of prediction sets
    predictions : np.ndarray
        Predicted probabilities
    q_hat : float
        Calibrated threshold

    Returns
    -------
    analysis : dict
        Dictionary with:
        - 'regions': Breakdown by confidence region
        - 'boundary_statistics': Stats near decision boundaries
        - 'efficiency': Efficiency metrics

    Notes
    -----
    Useful for creating Figure 4.4 (Prediction Set Visualization).
    """
    n = len(prediction_sets)

    # Classify predictions by region
    region_low = predictions <= q_hat
    region_high = predictions >= (1 - q_hat)
    region_uncertain = ~region_low & ~region_high

    # Count set types in each region
    regions = {
        'low_confidence_region': {
            'count': region_low.sum(),
            'fraction': region_low.mean(),
            'range': f'[0.00, {q_hat:.2f}]',
            'expected_set': '{0}',
            'actual_distribution': {
                '{0}': np.mean([s == [0] for i, s in enumerate(prediction_sets) if region_low[i]]),
                '{1}': np.mean([s == [1] for i, s in enumerate(prediction_sets) if region_low[i]]),
                '{0,1}': np.mean([sorted(s) == [0, 1] for i, s in enumerate(prediction_sets) if region_low[i]])
            }
        },
        'uncertain_region': {
            'count': region_uncertain.sum(),
            'fraction': region_uncertain.mean(),
            'range': f'({q_hat:.2f}, {1-q_hat:.2f})',
            'expected_set': '{0, 1}',
            'actual_distribution': {
                '{0}': np.mean([s == [0] for i, s in enumerate(prediction_sets) if region_uncertain[i]]) if region_uncertain.sum() > 0 else 0,
                '{1}': np.mean([s == [1] for i, s in enumerate(prediction_sets) if region_uncertain[i]]) if region_uncertain.sum() > 0 else 0,
                '{0,1}': np.mean([sorted(s) == [0, 1] for i, s in enumerate(prediction_sets) if region_uncertain[i]]) if region_uncertain.sum() > 0 else 0
            }
        },
        'high_confidence_region': {
            'count': region_high.sum(),
            'fraction': region_high.mean(),
            'range': f'[{1-q_hat:.2f}, 1.00]',
            'expected_set': '{1}',
            'actual_distribution': {
                '{0}': np.mean([s == [0] for i, s in enumerate(prediction_sets) if region_high[i]]),
                '{1}': np.mean([s == [1] for i, s in enumerate(prediction_sets) if region_high[i]]),
                '{0,1}': np.mean([sorted(s) == [0, 1] for i, s in enumerate(prediction_sets) if region_high[i]])
            }
        }
    }

    # Boundary statistics
    boundary_margin = 0.05
    near_low_boundary = np.abs(predictions - q_hat) < boundary_margin
    near_high_boundary = np.abs(predictions - (1 - q_hat)) < boundary_margin

    boundary_stats = {
        'near_low_boundary_count': near_low_boundary.sum(),
        'near_high_boundary_count': near_high_boundary.sum(),
        'low_boundary': q_hat,
        'high_boundary': 1 - q_hat
    }

    # Efficiency metrics
    efficiency = {
        'singleton_fraction': np.mean([len(s) == 1 for s in prediction_sets]),
        'doubleton_fraction': np.mean([len(s) == 2 for s in prediction_sets]),
        'efficiency_score': np.mean([len(s) == 1 for s in prediction_sets])  # Higher is better
    }

    return {
        'regions': regions,
        'boundary_statistics': boundary_stats,
        'efficiency': efficiency
    }
