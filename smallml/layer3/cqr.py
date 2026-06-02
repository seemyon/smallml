"""
Conformalized Quantile Regression (CQR) for Layer 3

Implements CQR (Romano, Patterson & Candès, 2019) for the regression and count
tasks. Two quantile regressors (LightGBM, pinball loss) estimate the lower and
upper conditional quantiles; their interval is then conformalized on a held-out
calibration set, yielding **locally-adaptive** prediction intervals with a
finite-sample coverage guarantee of ``1 - alpha``.

CQR is the default regression/count conformal method in :class:`smallml.Pipeline`
when there is enough training data to fit the quantile regressors reliably;
otherwise the pipeline falls back to (locally-adaptive, normalized) split
conformal in :class:`smallml.layer3.conformal_predictor.ConformalPredictor`.
"""

import numpy as np
from lightgbm import LGBMRegressor

VALID_TASKS = ("regression", "count")


class ConformalQuantileRegressor:
    """
    Conformalized Quantile Regression interval estimator.

    Parameters
    ----------
    alpha : float, default=0.10
        Miscoverage rate (target coverage = 1 - alpha). The lower/upper quantile
        regressors are fit at ``alpha/2`` and ``1 - alpha/2``.
    task : {'regression', 'count'}, default='regression'
        For ``'count'`` the interval bounds are rounded and clipped to the
        non-negative integers.
    random_seed : int, default=42
    n_estimators, learning_rate, num_leaves, min_child_samples :
        LightGBM hyperparameters for the quantile regressors (kept modest to
        limit overfitting on small data).

    Attributes
    ----------
    lower_model_, upper_model_ : LGBMRegressor
        Fitted lower/upper quantile regressors.
    q_hat_ : float
        Conformal correction applied symmetrically to the quantile interval.
    n_cal_ : int
        Number of calibration points.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        task: str = "regression",
        random_seed: int = 42,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 5,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")

        self.alpha = alpha
        self.task = task
        self.random_seed = random_seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples

        self.lower_model_ = None
        self.upper_model_ = None
        self.q_hat_ = None
        self.n_cal_ = None

    def fit(self, X_train, y_train, X_cal, y_cal) -> "ConformalQuantileRegressor":
        """
        Fit the quantile regressors on the training split and conformalize their
        interval on the calibration split.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)

        common = dict(
            objective="quantile",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_seed,
            verbosity=-1,
        )
        self.lower_model_ = LGBMRegressor(alpha=self.alpha / 2, **common)
        self.upper_model_ = LGBMRegressor(alpha=1 - self.alpha / 2, **common)
        self.lower_model_.fit(X_train, y_train)
        self.upper_model_.fit(X_train, y_train)

        lo = self.lower_model_.predict(X_cal)
        hi = self.upper_model_.predict(X_cal)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

        # CQR conformity scores (Romano et al., 2019, eq. 6).
        scores = np.maximum(lo - y_cal, y_cal - hi)
        n = len(scores)
        self.n_cal_ = n
        k = int(np.ceil((1 - self.alpha) * (n + 1)))
        k = min(k, n)
        self.q_hat_ = float(np.sort(scores)[k - 1])
        return self

    def predict_interval(self, X) -> np.ndarray:
        """Return locally-adaptive prediction intervals, shape (n, 2)."""
        if self.q_hat_ is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        lo = self.lower_model_.predict(X) - self.q_hat_
        hi = self.upper_model_.predict(X) + self.q_hat_
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        if self.task == "count":
            lo = np.clip(np.round(lo), 0, None)
            hi = np.clip(np.round(hi), 0, None)
        return np.column_stack([lo, hi])

    def validate_coverage(self, X, y, verbose: bool = False) -> dict:
        """Empirical coverage and mean interval width on a labeled set."""
        intervals = self.predict_interval(X)
        y = np.asarray(y, dtype=np.float64)
        inside = (y >= intervals[:, 0]) & (y <= intervals[:, 1])
        metrics = {
            "coverage": float(np.mean(inside)),
            "avg_interval_width": float(np.mean(intervals[:, 1] - intervals[:, 0])),
            "method": "cqr",
        }
        if verbose:
            print(f"CQR coverage: {metrics['coverage']:.3f}")
        return metrics
