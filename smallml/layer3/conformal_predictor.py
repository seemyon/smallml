"""
Conformal Predictor for Distribution-Free Uncertainty Quantification

This module implements Algorithm 4.4 (Split-Conformal Calibration) and
Algorithm 4.5 (Prediction Set Construction) from the SmallML framework.

"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import KFold


class ConformalPredictor:
    """
    Conformal prediction wrapper for hierarchical Bayesian models.

    Provides distribution-free uncertainty quantification with finite-sample
    coverage guarantees. Implements split-conformal and pooled calibration
    strategies for small-data SME contexts.

    Parameters
    ----------
    alpha : float, optional (default=0.10)
        Miscoverage rate. Target coverage = 1 - alpha.
        For alpha=0.10, expect 90% coverage.
    conservative_adjustment : float, optional (default=0.0)
        Conservative inflation factor for threshold.
        q_hat_final = q_hat_empirical × (1 + conservative_adjustment)
        Use 0.1-0.3 for very small calibration sets (<30 samples).
    random_seed : int, optional (default=42)
        Random seed for reproducibility

    Attributes
    ----------
    q_hat_ : float
        Calibrated threshold for prediction set construction
    calibration_scores_ : np.ndarray
        Nonconformity scores from calibration set
    n_cal_ : int
        Number of calibration samples
    empirical_coverage_ : float
        Coverage on calibration set (diagnostic)

    Examples
    --------
    >>> # Initialize predictor
    >>> cp = ConformalPredictor(alpha=0.10)
    >>>
    >>> # Calibrate on held-out data
    >>> cp.calibrate(
    ...     y_cal=y_calibration,
    ...     predictions_cal=bayesian_predictions
    ... )
    >>>
    >>> # Construct prediction sets for new customers
    >>> pred_sets = cp.predict_set(new_predictions)
    >>>
    >>> # Validate coverage on test set
    >>> coverage = cp.validate_coverage(y_test, predictions_test)
    """

    def __init__(
        self,
        alpha: float = 0.10,
        conservative_adjustment: float = 0.0,
        random_seed: int = 42
    ):
        """Initialize conformal predictor."""
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if conservative_adjustment < 0:
            raise ValueError(
                f"conservative_adjustment must be >= 0, "
                f"got {conservative_adjustment}"
            )

        self.alpha = alpha
        self.conservative_adjustment = conservative_adjustment
        self.random_seed = random_seed

        # Placeholders (set during calibration)
        self.q_hat_ = None
        self.calibration_scores_ = None
        self.n_cal_ = None
        self.empirical_coverage_ = None
        self.calibration_metadata_ = {}

    def compute_nonconformity_scores(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute nonconformity scores for binary classification.

        Uses absolute residual score: s_i = |y_i - p̂_i|

        Parameters
        ----------
        y_true : np.ndarray, shape (n,)
            True binary labels (0 or 1)
        predictions : np.ndarray, shape (n,)
            Predicted probabilities p̂(Y=1|X) in [0,1]

        Returns
        -------
        scores : np.ndarray, shape (n,)
            Nonconformity scores

        Notes
        -----
        Higher scores indicate predictions that are more "nonconforming"
        (i.e., confident but wrong predictions).
        """
        # Validate inputs
        if len(y_true) != len(predictions):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) vs "
                f"predictions ({len(predictions)})"
            )
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y_true must contain only 0 or 1")

        # Clip predictions to [0,1] if slightly outside due to numerical precision
        if not np.all((predictions >= 0) & (predictions <= 1)):
            min_pred = predictions.min()
            max_pred = predictions.max()
            # Allow small numerical errors (< 1e-6)
            if min_pred < -1e-6 or max_pred > 1 + 1e-6:
                raise ValueError(
                    f"predictions must be in [0,1], got range [{min_pred:.10f}, {max_pred:.10f}]"
                )
            # Clip small numerical errors
            predictions = np.clip(predictions, 0.0, 1.0)

        # Compute absolute residual scores
        scores = np.abs(y_true - predictions)

        return scores

    def calibrate(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        verbose: bool = True
    ) -> float:
        """
        Calibrate conformal predictor using Algorithm 4.4.

        Computes the (1-alpha)-quantile of calibration scores to determine
        the threshold q̂ for prediction set construction.

        Parameters
        ----------
        y_cal : np.ndarray, shape (n_cal,)
            True labels for calibration set
        predictions_cal : np.ndarray, shape (n_cal,)
            Predicted probabilities for calibration set
        verbose : bool, optional (default=True)
            Print calibration details

        Returns
        -------
        q_hat : float
            Calibrated threshold for prediction sets

        Notes
        -----
        Implements finite-sample correction from Vovk et al. (2005):
        k = ⌈(1-α)(n_cal+1)⌉
        """
        if verbose:
            print(f"\n{'='*60}")
            print("CONFORMAL CALIBRATION (Algorithm 4.4)")
            print(f"{'='*60}")
            print(f"Miscoverage rate α: {self.alpha:.2f}")
            print(f"Target coverage: {1-self.alpha:.1%}")
            print(f"Calibration samples: {len(y_cal)}")

        # Step 1: Compute nonconformity scores
        self.calibration_scores_ = self.compute_nonconformity_scores(
            y_cal, predictions_cal
        )
        self.n_cal_ = len(self.calibration_scores_)

        if verbose:
            print(f"\nNonconformity score statistics:")
            print(f"  Min:    {self.calibration_scores_.min():.4f}")
            print(f"  Q25:    {np.percentile(self.calibration_scores_, 25):.4f}")
            print(f"  Median: {np.median(self.calibration_scores_):.4f}")
            print(f"  Q75:    {np.percentile(self.calibration_scores_, 75):.4f}")
            print(f"  Max:    {self.calibration_scores_.max():.4f}")

        # Step 2: Sort scores
        sorted_scores = np.sort(self.calibration_scores_)

        # Step 3: Compute adjusted quantile (finite-sample correction)
        k = int(np.ceil((1 - self.alpha) * (self.n_cal_ + 1)))
        k = min(k, self.n_cal_)  # Handle edge case

        if verbose:
            print(f"\nQuantile calculation:")
            print(f"  k = ⌈(1-α)(n_cal+1)⌉ = ⌈{(1-self.alpha)*(self.n_cal_+1):.2f}⌉ = {k}")
            print(f"  Using {k}th smallest score (out of {self.n_cal_})")

        # Step 4: Extract threshold
        q_hat_empirical = sorted_scores[k - 1]  # 0-indexed

        if verbose:
            print(f"  Empirical q̂: {q_hat_empirical:.4f}")

        # Step 5: Apply conservative adjustment if needed
        if self.conservative_adjustment > 0:
            self.q_hat_ = q_hat_empirical * (1 + self.conservative_adjustment)
            if verbose:
                print(f"  Conservative adjustment: {self.conservative_adjustment:.2%}")
                print(f"  Final q̂: {self.q_hat_:.4f}")
        else:
            self.q_hat_ = q_hat_empirical

        # Diagnostic: Compute calibration set coverage
        pred_sets_cal = self.predict_set(predictions_cal, return_sets=True)
        coverage_cal = np.mean([
            y_cal[i] in pred_sets_cal[i]
            for i in range(len(y_cal))
        ])
        self.empirical_coverage_ = coverage_cal

        if verbose:
            print(f"\nCalibration set diagnostics:")
            print(f"  Empirical coverage: {coverage_cal:.3f} "
                  f"(target: {1-self.alpha:.3f})")

            # Average set size
            avg_set_size = np.mean([len(s) for s in pred_sets_cal])
            singleton_frac = np.mean([len(s) == 1 for s in pred_sets_cal])
            doubleton_frac = np.mean([len(s) == 2 for s in pred_sets_cal])
            empty_frac = np.mean([len(s) == 0 for s in pred_sets_cal])

            print(f"  Average set size: {avg_set_size:.2f}")
            print(f"  Singleton sets: {singleton_frac:.1%}")
            print(f"  Doubleton sets: {doubleton_frac:.1%}")
            print(f"  Empty sets: {empty_frac:.1%}")

            if coverage_cal < 0.85:
                print(f"\n⚠ WARNING: Coverage {coverage_cal:.3f} < 0.85")
                print(f"  Consider increasing conservative_adjustment")
            elif coverage_cal > 0.95:
                print(f"\n⚠ NOTE: Coverage {coverage_cal:.3f} > 0.95 (over-coverage)")
                print(f"  Acceptable but prediction sets may be less efficient")

        # Store metadata
        self.calibration_metadata_ = {
            'n_cal': self.n_cal_,
            'alpha': self.alpha,
            'q_hat': self.q_hat_,
            'empirical_coverage': coverage_cal,
            'avg_set_size': avg_set_size,
            'singleton_fraction': singleton_frac,
            'doubleton_fraction': doubleton_frac,
            'empty_fraction': empty_frac
        }

        return self.q_hat_

    def predict_set(
        self,
        predictions: Union[np.ndarray, float],
        return_sets: bool = True
    ) -> Union[List[List[int]], List[int]]:
        """
        Construct prediction sets using Algorithm 4.5.

        Parameters
        ----------
        predictions : np.ndarray or float
            Predicted probabilities p̂(Y=1|X).
            Can be single prediction or array.
        return_sets : bool, optional (default=True)
            If True, return list of sets (e.g., [[0], [1], [0,1]])
            If False, return encoded labels (0={0}, 1={1}, 2={0,1}, -1=empty)

        Returns
        -------
        prediction_sets : list
            If return_sets=True: List of lists, each containing {0, 1, or both}
            If return_sets=False: List of integers encoding set type

        Notes
        -----
        Prediction set construction:
        - Include 0 if p̂ ≤ q̂
        - Include 1 if p̂ ≥ 1 - q̂
        """
        if self.q_hat_ is None:
            raise RuntimeError(
                "Predictor not calibrated. Call .calibrate() first."
            )

        # Handle single prediction
        single_input = isinstance(predictions, (int, float))
        if single_input:
            predictions = np.array([predictions])

        # Validate and clip predictions
        if not np.all((predictions >= 0) & (predictions <= 1)):
            min_pred = predictions.min()
            max_pred = predictions.max()
            # Allow small numerical errors (< 1e-6)
            if min_pred < -1e-6 or max_pred > 1 + 1e-6:
                raise ValueError(
                    f"predictions must be in [0,1], got range [{min_pred:.10f}, {max_pred:.10f}]"
                )
            # Clip small numerical errors
            predictions = np.clip(predictions, 0.0, 1.0)

        prediction_sets = []

        for p_hat in predictions:
            pred_set = []

            # Check if 0 is in set: |0 - p̂| ≤ q̂ ⟺ p̂ ≤ q̂
            if p_hat <= self.q_hat_:
                pred_set.append(0)

            # Check if 1 is in set: |1 - p̂| ≤ q̂ ⟺ 1 - p̂ ≤ q̂ ⟺ p̂ ≥ 1 - q̂
            if p_hat >= (1 - self.q_hat_):
                pred_set.append(1)

            prediction_sets.append(pred_set)

        if not return_sets:
            # Encode as integers
            encoded = []
            for s in prediction_sets:
                if len(s) == 0:
                    encoded.append(-1)  # Empty
                elif s == [0]:
                    encoded.append(0)   # {0}
                elif s == [1]:
                    encoded.append(1)   # {1}
                else:  # [0, 1]
                    encoded.append(2)   # {0,1}
            prediction_sets = encoded

        # Return single value if single input
        if single_input:
            return prediction_sets[0]

        return prediction_sets

    def validate_coverage(
        self,
        y_test: np.ndarray,
        predictions_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Validate coverage guarantee on test set.

        Parameters
        ----------
        y_test : np.ndarray, shape (n_test,)
            True labels for test set
        predictions_test : np.ndarray, shape (n_test,)
            Predicted probabilities for test set
        verbose : bool, optional (default=True)
            Print validation results

        Returns
        -------
        metrics : dict
            Dictionary containing:
            - 'coverage': Empirical coverage
            - 'avg_set_size': Average prediction set size
            - 'singleton_fraction': Fraction of singleton sets
            - 'doubleton_fraction': Fraction of doubleton sets
            - 'empty_fraction': Fraction of empty sets
            - 'valid_coverage': True if coverage in [0.87, 0.93] for α=0.10

        Notes
        -----
        Expected coverage range: [0.87, 0.93] for α=0.10 (accounting for
        finite-sample variation).
        """
        if self.q_hat_ is None:
            raise RuntimeError(
                "Predictor not calibrated. Call .calibrate() first."
            )

        if verbose:
            print(f"\n{'='*60}")
            print("COVERAGE VALIDATION")
            print(f"{'='*60}")
            print(f"Test samples: {len(y_test)}")

        # Construct prediction sets
        pred_sets = self.predict_set(predictions_test, return_sets=True)

        # Compute coverage: fraction of test samples where y ∈ C(x)
        coverage_indicators = [
            y_test[i] in pred_sets[i]
            for i in range(len(y_test))
        ]
        coverage = np.mean(coverage_indicators)

        # Set size statistics
        set_sizes = [len(s) for s in pred_sets]
        avg_set_size = np.mean(set_sizes)
        singleton_frac = np.mean([sz == 1 for sz in set_sizes])
        doubleton_frac = np.mean([sz == 2 for sz in set_sizes])
        empty_frac = np.mean([sz == 0 for sz in set_sizes])

        # Check validity (acceptable range for α=0.10 is [0.87, 0.93])
        lower_bound = 1 - self.alpha - 0.03
        upper_bound = 1 - self.alpha + 0.03
        valid_coverage = lower_bound <= coverage <= upper_bound

        if verbose:
            print(f"\nCoverage results:")
            print(f"  Empirical coverage: {coverage:.3f}")
            print(f"  Target coverage:    {1-self.alpha:.3f}")
            print(f"  Acceptable range:   [{lower_bound:.3f}, {upper_bound:.3f}]")
            print(f"  Status: {'✓ PASS' if valid_coverage else '✗ FAIL'}")

            print(f"\nPrediction set statistics:")
            print(f"  Average set size: {avg_set_size:.2f}")
            print(f"  Singleton sets:   {singleton_frac:.1%} (definitive)")
            print(f"  Doubleton sets:   {doubleton_frac:.1%} (uncertain)")
            print(f"  Empty sets:       {empty_frac:.1%} (invalid)")

            if not valid_coverage:
                if coverage < lower_bound:
                    print(f"\n⚠ WARNING: Under-coverage detected!")
                    print(f"  Consider increasing conservative_adjustment")
                    print(f"  or recalibrating with more data")
                else:
                    print(f"\n⚠ NOTE: Over-coverage (acceptable but inefficient)")

            if empty_frac > 0.05:
                print(f"\n⚠ WARNING: {empty_frac:.1%} empty sets")
                print(f"  Consider adjusting q̂ threshold")

        metrics = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'singleton_fraction': singleton_frac,
            'doubleton_fraction': doubleton_frac,
            'empty_fraction': empty_frac,
            'valid_coverage': valid_coverage,
            'target_coverage': 1 - self.alpha,
            'acceptable_range': [lower_bound, upper_bound]
        }

        return metrics

    def pooled_calibration(
        self,
        y_cal_dict: Dict[int, np.ndarray],
        predictions_cal_dict: Dict[int, np.ndarray],
        verbose: bool = True
    ) -> float:
        """
        Pooled calibration across multiple SMEs (Strategy 2 from docs).

        Combines calibration data from J SMEs to increase effective sample
        size while maintaining exchangeability.

        Parameters
        ----------
        y_cal_dict : dict of {sme_id: np.ndarray}
            Calibration labels for each SME
        predictions_cal_dict : dict of {sme_id: np.ndarray}
            Calibration predictions for each SME
        verbose : bool, optional (default=True)
            Print pooling details

        Returns
        -------
        q_hat : float
            Calibrated threshold using pooled data

        Notes
        -----
        Assumes nonconformity score distributions are similar across SMEs
        (reasonable if SMEs are in the same industry).
        """
        if verbose:
            print(f"\n{'='*60}")
            print("POOLED CALIBRATION (Multi-SME Strategy)")
            print(f"{'='*60}")
            print(f"Number of SMEs: {len(y_cal_dict)}")

        # Pool all calibration data
        y_cal_pooled = []
        predictions_cal_pooled = []

        for sme_id in sorted(y_cal_dict.keys()):
            y_cal_pooled.extend(y_cal_dict[sme_id])
            predictions_cal_pooled.extend(predictions_cal_dict[sme_id])

            if verbose:
                print(f"  SME {sme_id}: {len(y_cal_dict[sme_id])} samples")

        y_cal_pooled = np.array(y_cal_pooled)
        predictions_cal_pooled = np.array(predictions_cal_pooled)

        if verbose:
            print(f"Total pooled samples: {len(y_cal_pooled)}")

        # Use standard calibration on pooled data
        return self.calibrate(
            y_cal_pooled,
            predictions_cal_pooled,
            verbose=verbose
        )

    def save_calibration(
        self,
        filepath: Union[str, Path]
    ) -> None:
        """
        Save calibrated predictor to disk.

        Parameters
        ----------
        filepath : str or Path
            Path to save pickle file
        """
        if self.q_hat_ is None:
            raise RuntimeError(
                "Predictor not calibrated. Call .calibrate() first."
            )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        calibration_data = {
            'q_hat': self.q_hat_,
            'alpha': self.alpha,
            'conservative_adjustment': self.conservative_adjustment,
            'calibration_scores': self.calibration_scores_,
            'n_cal': self.n_cal_,
            'empirical_coverage': self.empirical_coverage_,
            'metadata': self.calibration_metadata_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(calibration_data, f)

        print(f"✓ Calibration saved to {filepath}")

    def load_calibration(
        self,
        filepath: Union[str, Path]
    ) -> None:
        """
        Load calibrated predictor from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to pickle file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        with open(filepath, 'rb') as f:
            calibration_data = pickle.load(f)

        self.q_hat_ = calibration_data['q_hat']
        self.alpha = calibration_data['alpha']
        self.conservative_adjustment = calibration_data.get(
            'conservative_adjustment', 0.0
        )
        self.calibration_scores_ = calibration_data['calibration_scores']
        self.n_cal_ = calibration_data['n_cal']
        self.empirical_coverage_ = calibration_data.get('empirical_coverage')
        self.calibration_metadata_ = calibration_data.get('metadata', {})

        print(f"✓ Calibration loaded from {filepath}")
        print(f"  q̂ = {self.q_hat_:.4f}")
        print(f"  α = {self.alpha:.2f}")
        print(f"  Calibration samples: {self.n_cal_}")
