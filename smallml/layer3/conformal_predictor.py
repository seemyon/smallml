"""
Conformal Predictor for Distribution-Free Uncertainty Quantification

Implements split-conformal calibration and prediction-region construction with
finite-sample coverage guarantees. The wrapper is task-aware:

- binary      : prediction *sets* drawn from {0}, {1}, {0, 1}
- regression  : prediction *intervals* [ŷ - q̂, ŷ + q̂]
- count       : prediction *intervals*, clipped to non-negative integers

The calibration procedure (nonconformity scores + finite-sample quantile) is the
same across tasks; only the score definition and the region geometry differ.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Union

VALID_TASKS = ("binary", "regression", "count")


class ConformalPredictor:
    """
    Conformal prediction wrapper for hierarchical Bayesian models.

    Parameters
    ----------
    alpha : float, default=0.10
        Miscoverage rate. Target coverage = 1 - alpha.
    task : {'binary', 'regression', 'count'}, default='binary'
        Determines the nonconformity score and prediction-region geometry.
    conservative_adjustment : float, default=0.0
        Inflate the threshold: ``q_hat = q_hat_empirical * (1 + adjustment)``.
        Useful for very small calibration sets.
    random_seed : int, default=42

    Attributes
    ----------
    q_hat_ : float
        Calibrated threshold.
    calibration_scores_ : np.ndarray
    n_cal_ : int
    empirical_coverage_ : float
        Coverage on the calibration set (diagnostic).
    """

    def __init__(
        self,
        alpha: float = 0.10,
        task: str = "binary",
        conservative_adjustment: float = 0.0,
        random_seed: int = 42,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")
        if conservative_adjustment < 0:
            raise ValueError(
                f"conservative_adjustment must be >= 0, got {conservative_adjustment}"
            )

        self.alpha = alpha
        self.task = task
        self.conservative_adjustment = conservative_adjustment
        self.random_seed = random_seed

        self.q_hat_ = None
        self.calibration_scores_ = None
        self.n_cal_ = None
        self.empirical_coverage_ = None
        self.calibration_metadata_ = {}
        # Locally-adaptive (normalized) regression/count scoring:
        # s = |y - ŷ| / max(sigma(x), sigma_floor_). Set when ``sigma_cal`` is
        # passed to calibrate().
        self.normalized_ = False
        self.sigma_floor_ = None

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def compute_nonconformity_scores(
        self, y_true: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """
        Absolute-residual nonconformity score ``s_i = |y_i - ŷ_i|``.

        For binary tasks ``ŷ`` is the predicted probability of class 1 (and
        labels must be in {0, 1}); for regression/count ``ŷ`` is the predicted
        value / expected count.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)

        if len(y_true) != len(predictions):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) vs "
                f"predictions ({len(predictions)})"
            )

        if self.task == "binary":
            if not np.all((y_true == 0) | (y_true == 1)):
                raise ValueError("binary task: y_true must contain only 0 or 1")
            predictions = self._clip_probabilities(predictions)

        return np.abs(y_true - predictions)

    def calibrate(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        sigma_cal: np.ndarray = None,
        verbose: bool = True,
    ) -> float:
        """
        Calibrate the threshold q̂ as the finite-sample (1-alpha)-quantile of the
        calibration nonconformity scores: ``k = ⌈(1-α)(n_cal+1)⌉``.

        For regression/count, passing ``sigma_cal`` (a per-point scale, e.g. the
        Bayesian posterior std) switches to **locally-adaptive normalized** split
        conformal: scores become ``|y - ŷ| / max(sigma, sigma_floor_)`` and
        intervals become ``ŷ ± q̂ · max(sigma, sigma_floor_)``.
        """
        y_cal = np.asarray(y_cal, dtype=np.float64)
        predictions_cal = np.asarray(predictions_cal, dtype=np.float64)

        if self.task != "binary" and sigma_cal is not None:
            sigma_cal = np.asarray(sigma_cal, dtype=np.float64)
            self.sigma_floor_ = max(float(np.median(sigma_cal)) * 1e-3, 1e-8)
            self.normalized_ = True
            self.calibration_scores_ = np.abs(y_cal - predictions_cal) / np.maximum(
                sigma_cal, self.sigma_floor_
            )
        else:
            self.normalized_ = False
            self.calibration_scores_ = self.compute_nonconformity_scores(
                y_cal, predictions_cal
            )
        self.n_cal_ = len(self.calibration_scores_)

        sorted_scores = np.sort(self.calibration_scores_)
        k = int(np.ceil((1 - self.alpha) * (self.n_cal_ + 1)))
        k = min(k, self.n_cal_)
        q_hat_empirical = sorted_scores[k - 1]

        if self.conservative_adjustment > 0:
            self.q_hat_ = q_hat_empirical * (1 + self.conservative_adjustment)
        else:
            self.q_hat_ = float(q_hat_empirical)

        # Diagnostic coverage on calibration set.
        cov = self.validate_coverage(
            y_cal, predictions_cal, sigma=sigma_cal, verbose=False
        )
        self.empirical_coverage_ = cov["coverage"]
        self.calibration_metadata_ = {
            "n_cal": self.n_cal_,
            "alpha": self.alpha,
            "task": self.task,
            "q_hat": self.q_hat_,
            "empirical_coverage": cov["coverage"],
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print("CONFORMAL CALIBRATION")
            print(f"{'=' * 60}")
            print(f"Task: {self.task}")
            print(f"Target coverage: {1 - self.alpha:.1%}")
            print(f"Calibration samples: {self.n_cal_}")
            print(f"q̂: {self.q_hat_:.4f}")
            print(f"Calibration-set coverage: {cov['coverage']:.3f}")

        return self.q_hat_

    # ------------------------------------------------------------------ #
    # Prediction regions
    # ------------------------------------------------------------------ #

    def predict_region(
        self, predictions: Union[np.ndarray, float]
    ) -> Union[List, np.ndarray]:
        """
        Construct conformal prediction regions.

        Returns prediction *sets* (binary) or prediction *intervals*
        (regression/count). Dispatches on ``task``.
        """
        if self.task == "binary":
            return self.predict_set(predictions)
        return self.predict_interval(predictions)

    def predict_set(
        self,
        predictions: Union[np.ndarray, float],
        return_sets: bool = True,
    ) -> Union[List[List[int]], List[int]]:
        """
        Binary prediction sets (Algorithm 4.5):
        include 0 if p̂ ≤ q̂; include 1 if p̂ ≥ 1 - q̂.
        """
        if self.q_hat_ is None:
            raise RuntimeError("Predictor not calibrated. Call .calibrate() first.")
        if self.task != "binary":
            raise RuntimeError("predict_set is only valid for the binary task")

        single_input = isinstance(predictions, (int, float))
        predictions = np.atleast_1d(np.asarray(predictions, dtype=np.float64))
        predictions = self._clip_probabilities(predictions)

        prediction_sets = []
        for p_hat in predictions:
            pred_set = []
            if p_hat <= self.q_hat_:
                pred_set.append(0)
            if p_hat >= (1 - self.q_hat_):
                pred_set.append(1)
            prediction_sets.append(pred_set)

        if not return_sets:
            encoded = []
            for s in prediction_sets:
                if len(s) == 0:
                    encoded.append(-1)
                elif s == [0]:
                    encoded.append(0)
                elif s == [1]:
                    encoded.append(1)
                else:
                    encoded.append(2)
            prediction_sets = encoded

        return prediction_sets[0] if single_input else prediction_sets

    def predict_interval(
        self,
        predictions: Union[np.ndarray, float],
        sigma: np.ndarray = None,
    ) -> np.ndarray:
        """
        Regression/count prediction intervals.

        Constant-width ``[ŷ - q̂, ŷ + q̂]`` by default, or locally-adaptive
        ``[ŷ - q̂·σ_adj, ŷ + q̂·σ_adj]`` when the predictor was calibrated with a
        per-point ``sigma`` (``normalized_=True``); in that case ``sigma`` is
        required here too. For the count task bounds are rounded and clipped to
        the non-negative integers.

        Returns
        -------
        intervals : np.ndarray, shape (n, 2)
        """
        if self.q_hat_ is None:
            raise RuntimeError("Predictor not calibrated. Call .calibrate() first.")
        if self.task == "binary":
            raise RuntimeError("predict_interval is not valid for the binary task")

        predictions = np.atleast_1d(np.asarray(predictions, dtype=np.float64))

        if self.normalized_:
            if sigma is None:
                raise ValueError(
                    "This predictor was calibrated with normalized scores; "
                    "predict_interval requires a per-point `sigma`."
                )
            sigma_adj = np.maximum(
                np.atleast_1d(np.asarray(sigma, dtype=np.float64)), self.sigma_floor_
            )
            half_width = self.q_hat_ * sigma_adj
        else:
            half_width = self.q_hat_

        lower = predictions - half_width
        upper = predictions + half_width

        if self.task == "count":
            lower = np.clip(np.round(lower), 0, None)
            upper = np.clip(np.round(upper), 0, None)

        return np.column_stack([lower, upper])

    # ------------------------------------------------------------------ #
    # Coverage validation
    # ------------------------------------------------------------------ #

    def validate_coverage(
        self,
        y_test: np.ndarray,
        predictions_test: np.ndarray,
        sigma: np.ndarray = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Empirical coverage on a test set: fraction of points whose true outcome
        lies in the conformal region.

        Returns a dict with at least ``coverage``; binary additionally reports
        set-size statistics, regression/count report ``avg_interval_width``.
        For a normalized regression/count predictor, pass the per-point ``sigma``.
        """
        if self.q_hat_ is None:
            raise RuntimeError("Predictor not calibrated. Call .calibrate() first.")

        y_test = np.asarray(y_test)
        predictions_test = np.asarray(predictions_test, dtype=np.float64)

        if self.task == "binary":
            pred_sets = self.predict_set(predictions_test, return_sets=True)
            coverage = float(
                np.mean([y_test[i] in pred_sets[i] for i in range(len(y_test))])
            )
            set_sizes = np.array([len(s) for s in pred_sets])
            metrics = {
                "coverage": coverage,
                "avg_set_size": float(set_sizes.mean()),
                "singleton_fraction": float(np.mean(set_sizes == 1)),
                "doubleton_fraction": float(np.mean(set_sizes == 2)),
                "empty_fraction": float(np.mean(set_sizes == 0)),
            }
        else:
            intervals = self.predict_interval(predictions_test, sigma=sigma)
            inside = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
            metrics = {
                "coverage": float(np.mean(inside)),
                "avg_interval_width": float(np.mean(intervals[:, 1] - intervals[:, 0])),
            }

        lower_bound = 1 - self.alpha - 0.03
        upper_bound = 1 - self.alpha + 0.03
        metrics["valid_coverage"] = bool(
            lower_bound <= metrics["coverage"] <= upper_bound
        )
        metrics["target_coverage"] = 1 - self.alpha

        if verbose:
            print(f"\n{'=' * 60}")
            print("COVERAGE VALIDATION")
            print(f"{'=' * 60}")
            print(f"  Empirical coverage: {metrics['coverage']:.3f}")
            print(f"  Target coverage:    {1 - self.alpha:.3f}")

        return metrics

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_calibration(self, filepath: Union[str, Path]) -> None:
        """Save the calibrated predictor to a pickle file."""
        if self.q_hat_ is None:
            raise RuntimeError("Predictor not calibrated. Call .calibrate() first.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "q_hat": self.q_hat_,
                    "alpha": self.alpha,
                    "task": self.task,
                    "normalized": self.normalized_,
                    "sigma_floor": self.sigma_floor_,
                    "conservative_adjustment": self.conservative_adjustment,
                    "calibration_scores": self.calibration_scores_,
                    "n_cal": self.n_cal_,
                    "empirical_coverage": self.empirical_coverage_,
                    "metadata": self.calibration_metadata_,
                },
                f,
            )

    def load_calibration(self, filepath: Union[str, Path]) -> None:
        """Load a calibrated predictor from a pickle file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_hat_ = data["q_hat"]
        self.alpha = data["alpha"]
        self.task = data.get("task", "binary")
        self.normalized_ = data.get("normalized", False)
        self.sigma_floor_ = data.get("sigma_floor")
        self.conservative_adjustment = data.get("conservative_adjustment", 0.0)
        self.calibration_scores_ = data["calibration_scores"]
        self.n_cal_ = data["n_cal"]
        self.empirical_coverage_ = data.get("empirical_coverage")
        self.calibration_metadata_ = data.get("metadata", {})

    @staticmethod
    def _clip_probabilities(predictions: np.ndarray) -> np.ndarray:
        """Clip probabilities into [0, 1], rejecting values well outside."""
        if not np.all((predictions >= 0) & (predictions <= 1)):
            lo, hi = predictions.min(), predictions.max()
            if lo < -1e-6 or hi > 1 + 1e-6:
                raise ValueError(
                    f"binary predictions must be in [0,1], got range "
                    f"[{lo:.6f}, {hi:.6f}]"
                )
        return np.clip(predictions, 0.0, 1.0)
