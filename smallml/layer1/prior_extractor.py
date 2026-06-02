"""
Layer 1: Domain-Agnostic Prior Extraction (Transfer Learning Foundation)

This module trains a gradient-boosting model (LightGBM) on a large user-supplied
*reference* dataset and converts SHAP (SHapley Additive exPlanations) feature
attributions into Bayesian prior distributions (β₀, Σ₀) for the hierarchical
Bayesian model in Layer 2.

The framework ships **no** built-in domain data: the reference dataset is always
provided by the user, and the priors describe whatever patterns exist in it. The
extraction is identical regardless of domain (e.g. sports, ecology, finance).

Prior-extraction mechanism (unchanged from the original framework, generalized to
a single reference dataset):

  β₀_j = mean(|SHAP_j|) / std(x_j)          # coefficient-scale prior mean
  Σ₀   = diag(σ²_j × (1 + λ))               # diffuse diagonal prior covariance

where σ²_j is the between-partition variance of the per-feature mean |SHAP|.
The original implementation measured this variance *across reference datasets*;
with a single reference dataset we measure it *across K random folds*, which is a
faithful generalization of the same quantity.
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold

VALID_TASKS = ("binary", "regression", "count")


class PriorExtractor:
    """
    Extract Bayesian transfer-learning priors (β₀, Σ₀) from a reference dataset.

    Trains a LightGBM model on the reference data against the user-specified
    target column, computes SHAP feature attributions, and converts them into a
    prior mean vector and (diagonal) prior covariance matrix that Layer 2 uses as
    population-level hyperpriors.

    Parameters
    ----------
    task : {'binary', 'regression', 'count'}, default='binary'
        Prediction task. Controls the gradient-boosting objective:
        - 'binary'     : LGBMClassifier (positive-class SHAP)
        - 'regression' : LGBMRegressor (L2 objective)
        - 'count'      : LGBMRegressor with Poisson objective
    lambda_scale : float, default=1.0
        Prior-variance scaling factor. Σ₀ = diag(σ²_j × (1 + lambda_scale)).
        Higher = more diffuse (conservative) priors. Recommended range [0.5, 2.0].
    n_folds : int, default=5
        Number of folds used to estimate between-partition SHAP variance (Σ₀).
    max_shap_samples : int, default=2000
        Maximum number of reference rows used for SHAP computation (sampled
        without replacement for speed). Set to None to use all rows.
    min_reference_size : int, default=100
        Minimum number of reference rows required for reliable extraction.
    random_seed : int, default=42
        Seed for model training, sampling, and fold assignment.

    Attributes
    ----------
    beta_0_ : np.ndarray, shape (p,)
        Extracted prior mean vector.
    Sigma_0_ : np.ndarray, shape (p, p)
        Extracted diagonal prior covariance matrix.
    feature_names_ : List[str]
        Feature names in column order.
    model_ : LGBMClassifier or LGBMRegressor
        The fitted gradient-boosting model.
    shap_values_ : np.ndarray, shape (n_shap, p)
        SHAP values computed on the (sampled) reference data.
    metadata_ : dict
        Extraction metadata (timestamp, task, sizes, lambda_scale).

    Examples
    --------
    >>> extractor = PriorExtractor(task='binary')
    >>> extractor.fit(reference_df, target_col='outcome')
    >>> priors = extractor.get_priors()
    >>> priors['beta_0'].shape, priors['Sigma_0'].shape
    ((12,), (12, 12))
    """

    def __init__(
        self,
        task: str = "binary",
        lambda_scale: float = 1.0,
        n_folds: int = 5,
        max_shap_samples: Optional[int] = 2000,
        min_reference_size: int = 100,
        random_seed: int = 42,
    ):
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")
        if lambda_scale < 0:
            raise ValueError(f"lambda_scale must be >= 0, got {lambda_scale}")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")

        self.task = task
        self.lambda_scale = lambda_scale
        self.n_folds = n_folds
        self.max_shap_samples = max_shap_samples
        self.min_reference_size = min_reference_size
        self.random_seed = random_seed

        # Set during fit()
        self.beta_0_: Optional[np.ndarray] = None
        self.Sigma_0_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.model_ = None
        self.shap_values_: Optional[np.ndarray] = None
        self.metadata_: Dict = {}

    def fit(
        self,
        reference_data: pd.DataFrame,
        target_col: str,
        feature_names: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ) -> "PriorExtractor":
        """
        Train the base model and extract priors from the reference dataset.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Large reference dataset. Feature columns must be numeric (encode
            categoricals beforehand, e.g. with ``smallml.FeaturePreprocessor``).
        target_col : str
            Name of the target column in ``reference_data``.
        feature_names : sequence of str, optional
            Subset/order of feature columns to use. Defaults to every column
            except ``target_col``.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        self : PriorExtractor
        """
        if not isinstance(reference_data, pd.DataFrame):
            raise TypeError("reference_data must be a pandas DataFrame")
        if target_col not in reference_data.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in reference_data columns"
            )

        if feature_names is None:
            feature_names = [c for c in reference_data.columns if c != target_col]
        feature_names = list(feature_names)
        if len(feature_names) == 0:
            raise ValueError("No feature columns found in reference_data")

        n_ref = len(reference_data)
        if n_ref < self.min_reference_size:
            raise ValueError(
                f"Reference dataset too small for reliable prior extraction: "
                f"{n_ref} rows < minimum {self.min_reference_size}. "
                "Provide a larger reference dataset, or lower "
                "PriorExtractor(min_reference_size=...) at your own risk."
            )

        X = reference_data[feature_names].astype(np.float64)
        y = reference_data[target_col].values

        # Non-numeric features cannot be normalized to coefficient scale.
        if X.isnull().all().any():
            bad = [f for f in feature_names if X[f].isnull().all()]
            raise ValueError(
                f"Reference feature(s) {bad} are entirely non-numeric/NaN. "
                "Encode categorical features before prior extraction."
            )

        self.feature_names_ = feature_names

        if verbose:
            print(f"\n{'=' * 70}")
            print("LAYER 1: PRIOR EXTRACTION (LightGBM + SHAP)")
            print(f"{'=' * 70}")
            print(f"Task: {self.task}")
            print(f"Reference rows: {n_ref:,}")
            print(f"Features (p): {len(feature_names)}")

        # --- Train base gradient-boosting model -----------------------------
        self.model_ = self._fit_base_model(X, y, verbose=verbose)

        # --- Compute SHAP values --------------------------------------------
        X_shap = self._sample_for_shap(X)
        self.shap_values_ = self._compute_shap_values(X_shap, verbose=verbose)

        # --- β₀: coefficient-scale prior means ------------------------------
        self.beta_0_ = self._extract_prior_means(self.shap_values_, X)

        # --- Σ₀: between-fold variance of mean |SHAP| -----------------------
        self.Sigma_0_ = self._extract_prior_covariance(
            self.shap_values_, verbose=verbose
        )

        self.metadata_ = {
            "extraction_timestamp": datetime.now().isoformat(),
            "task": self.task,
            "lambda_scale": self.lambda_scale,
            "n_features": len(feature_names),
            "n_reference_samples": n_ref,
            "n_shap_samples": len(X_shap),
            "n_folds": self.n_folds,
        }

        if verbose:
            print(f"\n✓ Priors extracted for {len(feature_names)} features")
            print(f"  Mean |β₀|: {np.abs(self.beta_0_).mean():.4f}")
            print(f"  Mean prior std: {np.sqrt(np.diag(self.Sigma_0_)).mean():.4f}")
            print(f"{'=' * 70}\n")

        return self

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #

    def get_priors(self) -> Dict:
        """
        Return the extracted priors as a dictionary.

        Returns
        -------
        priors : dict
            ``{'beta_0', 'Sigma_0', 'feature_names', 'lambda_scale', 'metadata'}``
            -- the schema consumed by :class:`smallml.Pipeline`.
        """
        if self.beta_0_ is None:
            raise RuntimeError("Extractor not fitted. Call fit() first.")
        return {
            "beta_0": self.beta_0_,
            "Sigma_0": self.Sigma_0_,
            "feature_names": self.feature_names_,
            "lambda_scale": self.lambda_scale,
            "metadata": self.metadata_,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _fit_base_model(self, X: pd.DataFrame, y: np.ndarray, verbose: bool):
        """Train the LightGBM base model appropriate to the task."""
        common = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.random_seed,
            verbosity=-1,
        )

        if self.task == "binary":
            uniques = np.unique(y[~pd.isnull(y)])
            if not set(uniques.tolist()).issubset({0, 1}):
                raise ValueError(
                    "binary task requires a target containing only {0, 1}; "
                    f"found values {uniques.tolist()}"
                )
            model = LGBMClassifier(objective="binary", **common)
        elif self.task == "regression":
            model = LGBMRegressor(objective="regression", **common)
        else:  # count
            if np.any(y < 0):
                raise ValueError("count task requires non-negative integer target")
            model = LGBMRegressor(objective="poisson", **common)

        if verbose:
            print(f"\n[1/3] Training LightGBM base model ({self.task})...")
        model.fit(X, y)
        if verbose:
            print("  ✓ Base model trained")
        return model

    def _sample_for_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subsample rows for SHAP computation (speed) while keeping enough."""
        if self.max_shap_samples is None or len(X) <= self.max_shap_samples:
            return X
        rng = np.random.RandomState(self.random_seed)
        idx = rng.choice(len(X), size=self.max_shap_samples, replace=False)
        return X.iloc[np.sort(idx)]

    def _compute_shap_values(self, X_shap: pd.DataFrame, verbose: bool) -> np.ndarray:
        """Compute SHAP values, normalizing across shap-version output shapes."""
        if verbose:
            print(f"\n[2/3] Computing SHAP values on {len(X_shap):,} rows...")

        explainer = shap.TreeExplainer(self.model_)
        sv = explainer.shap_values(X_shap)

        # Normalize to a 2-D (n, p) array for the positive class / single output.
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            # shape (n, p, n_classes) -> positive class
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]

        expected = (len(X_shap), X_shap.shape[1])
        if sv.shape != expected:
            raise ValueError(
                f"Unexpected SHAP values shape {sv.shape}, expected {expected}"
            )

        if verbose:
            print(f"  ✓ SHAP values computed: shape {sv.shape}")
        return sv

    def _extract_prior_means(
        self, shap_values: np.ndarray, X: pd.DataFrame
    ) -> np.ndarray:
        """β₀_j = mean(|SHAP_j|) / std(x_j)  (identical to original mechanism)."""
        phi_j = np.abs(shap_values).mean(axis=0)
        std_x = X.std().values
        std_x = np.where(std_x < 1e-10, 1.0, std_x)
        return phi_j / std_x

    def _extract_prior_covariance(
        self, shap_values: np.ndarray, verbose: bool
    ) -> np.ndarray:
        """
        Σ₀ = diag(σ²_j × (1 + λ)) where σ²_j is the between-fold variance of the
        per-feature mean |SHAP|. Generalizes cross-dataset variance to K folds.
        """
        if verbose:
            print(f"\n[3/3] Estimating prior covariance over {self.n_folds} folds...")

        n = len(shap_values)
        n_folds = min(self.n_folds, n)
        abs_shap = np.abs(shap_values)

        if n_folds < 2:
            # Degenerate: not enough rows to estimate variance. Fall back to a
            # diffuse default so priors remain weakly informative.
            warnings.warn(
                "Too few SHAP samples to estimate prior variance; using a "
                "diffuse default covariance."
            )
            p = shap_values.shape[1]
            return np.eye(p) * (1.0 + self.lambda_scale)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        fold_means = []
        for _, fold_idx in kf.split(abs_shap):
            fold_means.append(abs_shap[fold_idx].mean(axis=0))
        fold_means = np.array(fold_means)  # (n_folds, p)

        sigma_sq = fold_means.var(axis=0, ddof=1)
        sigma_sq = np.maximum(sigma_sq, 1e-6)  # avoid overconfident priors
        sigma_sq_scaled = sigma_sq * (1 + self.lambda_scale)

        if verbose:
            print("  ✓ Prior covariance estimated")
        return np.diag(sigma_sq_scaled)
