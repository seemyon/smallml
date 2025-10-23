"""
SHAP-Based Prior Extraction for Bayesian Transfer Learning (Layer 1)

This module implements the bridge between transfer learning (Layer 1) and
hierarchical Bayesian modeling (Layer 2). It extracts Bayesian prior distributions
(β₀, Σ₀) from a trained CatBoost model using SHAP (SHapley Additive exPlanations)
values to quantify feature importance and cross-dataset transferability.

Key Components:
- SHAPPriorExtractor: Main class for prior extraction
- Implements Algorithm 4.2
- Generates Tables 4.6 and 4.7 for white paper
- Validates prior quality through predictive checks

References:
- Section 4.2.3: Prior Distribution Extraction
- Algorithm 4.2: Prior extraction procedure
- Table 4.6: Extracted prior distributions for top features
- Table 4.7: Prior predictive performance validation
"""

from typing import Dict, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings

import shap
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression


class SHAPPriorExtractor:
    """
    Extract Bayesian priors from trained CatBoost model using SHAP values.

    This class bridges deterministic gradient boosting (Layer 1) and probabilistic
    Bayesian inference (Layer 2) by transforming learned patterns into prior
    distributions. It implements Algorithm 4.2 from the SmallML framework.

    The extraction process:
    1. Compute SHAP values on validation data → quantify feature effects
    2. Normalize to coefficient scale → prior means (β₀)
    3. Measure cross-dataset variance → prior uncertainties (Σ₀)
    4. Apply scaling factor → conservative priors
    5. Validate through prior predictive checks

    Parameters
    ----------
    model : CatBoostClassifier
        Trained CatBoost model from Layer 1 transfer learning
    X_train : pd.DataFrame
        Training feature matrix (for normalization statistics)
    lambda_scale : float, default=1.0
        Prior variance scaling factor (1.0 = double empirical variance)
        Range: [0.5, 2.0]. Higher = more diffuse priors.
    random_seed : int, default=42
        Random seed for reproducibility in prior predictive checks

    Attributes
    ----------
    shap_values_ : np.ndarray
        Computed SHAP values (shape: [n_samples, n_features])
    beta_0_ : np.ndarray
        Prior mean vector (shape: [n_features])
    Sigma_0_ : np.ndarray
        Prior covariance matrix (shape: [n_features, n_features])
        Diagonal structure assumes independence between features
    feature_names_ : List[str]
        Feature names in order
    prior_metadata_ : Dict
        Extraction metadata (timestamp, λ, validation metrics)

    Examples
    --------
    >>> from catboost import CatBoostClassifier
    >>> model = CatBoostClassifier()
    >>> model.load_model('models/transfer_learning/catboost_base.cbm')
    >>>
    >>> extractor = SHAPPriorExtractor(model, X_train, lambda_scale=1.0)
    >>> beta_0, Sigma_0 = extractor.extract_priors(X_val, dataset_source_labels)
    >>>
    >>> # Save for hierarchical model
    >>> extractor.save_priors('models/transfer_learning/priors.pkl')
    """

    def __init__(
        self,
        model: CatBoostClassifier,
        X_train: pd.DataFrame,
        lambda_scale: float = 1.0,
        random_seed: int = 42,
    ):
        """
        Initialize SHAP prior extractor.

        Parameters
        ----------
        model : CatBoostClassifier
            Trained CatBoost model
        X_train : pd.DataFrame
            Training data (for computing normalization statistics)
        lambda_scale : float
            Variance scaling factor (default: 1.0)
        random_seed : int
            Random seed for reproducibility
        """
        self.model = model
        self.X_train = X_train
        self.lambda_scale = lambda_scale
        self.random_seed = random_seed

        # Validate inputs
        if not isinstance(model, CatBoostClassifier):
            raise TypeError("model must be a trained CatBoostClassifier")
        if not hasattr(model, "tree_count_") or model.tree_count_ == 0:
            raise ValueError("model must be fitted before prior extraction")
        if lambda_scale < 0.5 or lambda_scale > 2.0:
            warnings.warn(
                f"lambda_scale={lambda_scale} outside recommended range [0.5, 2.0]"
            )

        # Attributes set during extraction
        self.shap_values_: Optional[np.ndarray] = None
        self.beta_0_: Optional[np.ndarray] = None
        self.Sigma_0_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = list(X_train.columns)
        self.prior_metadata_: Dict = {}

        np.random.seed(random_seed)

    def compute_shap_values(
        self,
        X_val: pd.DataFrame,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Compute SHAP values on validation set (Algorithm 4.2, Step 1).

        Uses TreeExplainer for exact SHAP computation (fast for tree models).
        For binary classification, extracts SHAP values for positive class (churn=1).

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation feature matrix (shape: [n_samples, n_features])
        verbose : bool, default=True
            Print progress information

        Returns
        -------
        shap_values : np.ndarray
            SHAP values for each sample and feature (shape: [n_samples, n_features])

        """
        if verbose:
            print("Computing SHAP values...")
            print(f"  Validation set: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")

        # Create TreeExplainer (fast for CatBoost)
        explainer = shap.TreeExplainer(self.model)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_val)

        # For binary classification, CatBoost returns SHAP values for positive class
        # If it returns a list, take the positive class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Validate shape
        expected_shape = (X_val.shape[0], X_val.shape[1])
        if shap_values.shape != expected_shape:
            raise ValueError(
                f"Unexpected SHAP values shape: {shap_values.shape}, "
                f"expected {expected_shape}"
            )

        self.shap_values_ = shap_values

        if verbose:
            print(f"✓ SHAP values computed: shape {shap_values.shape}")
            print(f"  Mean absolute SHAP: {np.abs(shap_values).mean():.4f}")

        return shap_values

    def extract_prior_means(
        self,
        X_val: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Extract prior means from SHAP values (Algorithm 4.2, Steps 2-3).

        Transforms average SHAP values into coefficient-scale priors:
        1. Compute average absolute SHAP per feature: φ_j = mean(|SHAP_j|)
        2. Normalize by feature std dev: β̃_j = φ_j / std(x_j)
        3. Set prior mean: β₀_j = β̃_j

        Parameters
        ----------
        X_val : pd.DataFrame, optional
            Validation data (required if SHAP values not yet computed)
        verbose : bool, default=True
            Print progress information

        Returns
        -------
        beta_0 : np.ndarray
            Prior mean vector (shape: [n_features])

        Raises
        ------
        ValueError
            If SHAP values not computed and X_val not provided
        """
        # Ensure SHAP values exist
        if self.shap_values_ is None:
            if X_val is None:
                raise ValueError(
                    "SHAP values not computed. Call compute_shap_values() first "
                    "or provide X_val."
                )
            self.compute_shap_values(X_val, verbose=verbose)

        if verbose:
            print("\nExtracting prior means (β₀)...")

        # Step 1: Average absolute SHAP per feature
        phi_j = np.abs(self.shap_values_).mean(axis=0)

        # Step 2: Get feature standard deviations from training data
        std_x = self.X_train.std().values

        # Handle zero standard deviations (constant features)
        std_x = np.where(std_x < 1e-10, 1.0, std_x)

        # Step 3: Normalize to coefficient scale
        beta_0 = phi_j / std_x

        self.beta_0_ = beta_0

        if verbose:
            print(f"✓ Prior means extracted: {len(beta_0)} features")
            print(f"  Mean |β₀|: {np.abs(beta_0).mean():.4f}")
            print(f"  Max |β₀|: {np.abs(beta_0).max():.4f}")
            print(f"  Top 3 features by |β₀|:")
            top_indices = np.argsort(np.abs(beta_0))[-3:][::-1]
            for idx in top_indices:
                print(f"    {self.feature_names_[idx]}: {beta_0[idx]:.4f}")

        return beta_0

    def extract_prior_variances(
        self,
        X_val: pd.DataFrame,
        dataset_source: pd.Series,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Extract prior variances from cross-dataset SHAP heterogeneity (Algorithm 4.2, Steps 4-6).

        Measures how consistently each feature's effect transfers across datasets:
        1. Partition validation data by dataset (telco, bank, ecommerce)
        2. Compute dataset-specific SHAP values: φ_j^(k)
        3. Calculate between-dataset variance: σ²_j = Var(φ_j^(1), ..., φ_j^(K))
        4. Apply scaling factor: Σ₀ = diag(σ²_j × (1 + λ))

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation feature matrix
        dataset_source : pd.Series
            Dataset labels ('telco', 'bank', 'ecomm') for each validation sample
        verbose : bool, default=True
            Print progress information

        Returns
        -------
        Sigma_0 : np.ndarray
            Prior covariance matrix (shape: [n_features, n_features])
            Diagonal structure assumes feature independence

        Raises
        ------
        ValueError
            If less than 2 datasets available (need variance)
        """
        if verbose:
            print("\nExtracting prior variances (Σ₀)...")

        # Ensure SHAP values computed
        if self.shap_values_ is None:
            self.compute_shap_values(X_val, verbose=verbose)

        # Get unique datasets
        unique_datasets = dataset_source.unique()
        if len(unique_datasets) < 2:
            raise ValueError(
                f"Need at least 2 datasets for variance estimation, got {len(unique_datasets)}"
            )

        if verbose:
            print(f"  Computing cross-dataset variance across {len(unique_datasets)} datasets")

        # Step 4: Compute dataset-specific SHAP values
        dataset_shap_means = []
        for dataset in unique_datasets:
            mask = (dataset_source == dataset)
            n_samples = mask.sum()

            if n_samples < 10:
                warnings.warn(
                    f"Dataset '{dataset}' has only {n_samples} validation samples. "
                    f"Variance estimates may be unreliable."
                )

            # Average absolute SHAP for this dataset
            phi_j_k = np.abs(self.shap_values_[mask]).mean(axis=0)
            dataset_shap_means.append(phi_j_k)

            if verbose:
                print(f"    {dataset}: {n_samples} samples, mean |SHAP| = {phi_j_k.mean():.4f}")

        # Convert to array (shape: [n_datasets, n_features])
        shap_matrix = np.array(dataset_shap_means)

        # Step 5: Calculate between-dataset variance
        sigma_squared_j = shap_matrix.var(axis=0, ddof=1)  # Use sample variance (n-1)

        # Handle zero variances (perfectly consistent features)
        # Set minimum variance to avoid overconfident priors
        min_variance = 1e-6
        sigma_squared_j = np.maximum(sigma_squared_j, min_variance)

        # Step 6: Apply scaling factor (conservative adjustment)
        sigma_squared_j_scaled = sigma_squared_j * (1 + self.lambda_scale)

        # Construct diagonal covariance matrix
        Sigma_0 = np.diag(sigma_squared_j_scaled)

        self.Sigma_0_ = Sigma_0

        if verbose:
            prior_stds = np.sqrt(np.diag(Sigma_0))
            print(f"✓ Prior variances extracted: {len(prior_stds)} features")
            print(f"  Mean prior std: {prior_stds.mean():.4f}")
            print(f"  Median prior std: {np.median(prior_stds):.4f}")
            print(f"  Top 3 most uncertain features (largest σ₀):")
            top_var_indices = np.argsort(prior_stds)[-3:][::-1]
            for idx in top_var_indices:
                print(f"    {self.feature_names_[idx]}: σ₀ = {prior_stds[idx]:.4f}")

        return Sigma_0

    def extract_priors(
        self,
        X_val: pd.DataFrame,
        dataset_source: pd.Series,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete prior extraction pipeline (Algorithm 4.2).

        Convenience method that runs the full extraction:
        1. Compute SHAP values
        2. Extract prior means (β₀)
        3. Extract prior variances (Σ₀)

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation feature matrix
        dataset_source : pd.Series
            Dataset labels for each validation sample
        verbose : bool, default=True
            Print progress information

        Returns
        -------
        beta_0 : np.ndarray
            Prior mean vector (shape: [n_features])
        Sigma_0 : np.ndarray
            Prior covariance matrix (shape: [n_features, n_features])
        """
        if verbose:
            print("=" * 70)
            print("SHAP-Based Prior Extraction (Algorithm 4.2)")
            print("=" * 70)

        # Step 1: Compute SHAP values
        self.compute_shap_values(X_val, verbose=verbose)

        # Steps 2-3: Extract prior means
        beta_0 = self.extract_prior_means(verbose=verbose)

        # Steps 4-6: Extract prior variances
        Sigma_0 = self.extract_prior_variances(X_val, dataset_source, verbose=verbose)

        # Store metadata
        self.prior_metadata_ = {
            "extraction_timestamp": datetime.now().isoformat(),
            "lambda_scale": self.lambda_scale,
            "n_features": len(beta_0),
            "n_val_samples": len(X_val),
            "n_datasets": len(dataset_source.unique()),
            "datasets": list(dataset_source.unique()),
        }

        if verbose:
            print("\n" + "=" * 70)
            print("Prior Extraction Complete!")
            print("=" * 70)

        return beta_0, Sigma_0

    def prior_predictive_check(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_samples: int = 100,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Validate prior quality through predictive checks (Table 4.7).

        Compares three models on validation data:
        1. Random coefficients: β ~ Normal(0, 1)
        2. Prior-only: β ~ Normal(β₀, Σ₀)
        3. Trained CatBoost: Baseline reference

        A well-calibrated prior should:
        - Outperform random coefficients (AUC_prior >> AUC_random)
        - Underperform full model (AUC_prior < AUC_catboost)

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation labels
        n_samples : int, default=100
            Number of coefficient samples for Monte Carlo averaging
        verbose : bool, default=True
            Print results

        Returns
        -------
        results : Dict[str, float]
            AUC scores for each model type

        Raises
        ------
        ValueError
            If priors not yet extracted
        """
        if self.beta_0_ is None or self.Sigma_0_ is None:
            raise ValueError("Priors not extracted. Call extract_priors() first.")

        if verbose:
            print("\n" + "=" * 70)
            print("Prior Predictive Check (Table 4.7)")
            print("=" * 70)

        # Handle NaN values by filling with 0 (neutral for missing dataset-specific features)
        X_val_filled = X_val.fillna(0)
        X_val_array = X_val_filled.values
        y_val_array = y_val.values

        if verbose and X_val.isna().sum().sum() > 0:
            n_nan = X_val.isna().sum().sum()
            print(f"\n⚠ Filling {n_nan:,} NaN values with 0 for predictive check")
            print("  (NaN values represent dataset-specific features)")

        # Model 1: Random coefficients
        auc_random_list = []
        for _ in range(n_samples):
            beta_random = np.random.normal(0, 1, len(self.beta_0_))
            logits = X_val_array @ beta_random
            probs = 1 / (1 + np.exp(-logits))
            auc = roc_auc_score(y_val_array, probs)
            auc_random_list.append(auc)
        auc_random = np.mean(auc_random_list)

        # Model 2: Prior-only predictions
        auc_prior_list = []
        for _ in range(n_samples):
            beta_prior = np.random.multivariate_normal(
                self.beta_0_, self.Sigma_0_
            )
            logits = X_val_array @ beta_prior
            probs = 1 / (1 + np.exp(-logits))
            auc = roc_auc_score(y_val_array, probs)
            auc_prior_list.append(auc)
        auc_prior = np.mean(auc_prior_list)

        # Model 3: Trained CatBoost (reference)
        # CatBoost handles NaN values internally, so use original X_val
        probs_catboost = self.model.predict_proba(X_val)[:, 1]
        auc_catboost = roc_auc_score(y_val_array, probs_catboost)

        results = {
            "random_coefficients": auc_random,
            "prior_only": auc_prior,
            "trained_catboost": auc_catboost,
        }

        if verbose:
            print(f"\nResults ({n_samples} Monte Carlo samples):")
            print(f"  1. Random coefficients:  AUC = {auc_random:.4f}")
            print(f"  2. Prior-only:           AUC = {auc_prior:.4f}")
            print(f"  3. Trained CatBoost:     AUC = {auc_catboost:.4f}")
            print()
            print("Interpretation:")
            if auc_prior > auc_random + 0.05:
                print("  ✓ Priors encode transferable knowledge (prior >> random)")
            else:
                print("  ⚠ Priors weak (prior ≈ random)")
            if auc_catboost > auc_prior + 0.05:
                print("  ✓ SME data essential (CatBoost >> prior)")
            else:
                print("  ⚠ Priors may be overconfident (CatBoost ≈ prior)")

        return results

    def generate_table_4_6(
        self,
        top_n: int = 5,
        feature_importances: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate Table 4.6: Extracted Prior Distributions for Top Features.

        Parameters
        ----------
        top_n : int, default=5
            Number of top features to include
        feature_importances : pd.DataFrame, optional
            Feature importance from CatBoost (from Table 4.4)
            If None, uses feature magnitudes from β₀

        Returns
        -------
        table_4_6 : pd.DataFrame
            Table with columns: Feature, Importance, Avg SHAP, Prior Mean, Prior Std
        """
        if self.beta_0_ is None or self.Sigma_0_ is None:
            raise ValueError("Priors not extracted. Call extract_priors() first.")

        # Get average SHAP values
        avg_shap = np.abs(self.shap_values_).mean(axis=0)

        # Get prior standard deviations
        prior_stds = np.sqrt(np.diag(self.Sigma_0_))

        # Determine top features
        if feature_importances is not None:
            # Use provided importances (from Table 4.4)
            top_features = feature_importances.head(top_n)
            feature_indices = [
                self.feature_names_.index(f) for f in top_features["Feature"]
            ]
        else:
            # Use magnitude of β₀
            feature_indices = np.argsort(np.abs(self.beta_0_))[-top_n:][::-1]
            top_features = None

        # Build table
        table_data = []
        for i, idx in enumerate(feature_indices):
            feature_name = self.feature_names_[idx]
            row = {
                "Feature": feature_name,
                "Importance (w_j)": (
                    top_features.iloc[i]["Importance"]
                    if top_features is not None
                    else np.abs(self.beta_0_[idx])
                ),
                "Avg SHAP (φ_j)": avg_shap[idx],
                "Prior Mean (β₀_j)": self.beta_0_[idx],
                "Prior Std (√Σ₀_jj)": prior_stds[idx],
            }
            table_data.append(row)

        table_4_6 = pd.DataFrame(table_data)

        return table_4_6

    def save_priors(
        self,
        filepath: Union[str, Path],
        include_metadata: bool = True,
    ) -> None:
        """
        Save extracted priors to pickle file.

        Parameters
        ----------
        filepath : str or Path
            Output path (e.g., 'models/transfer_learning/priors.pkl')
        include_metadata : bool, default=True
            Include extraction metadata in saved file
        """
        if self.beta_0_ is None or self.Sigma_0_ is None:
            raise ValueError("Priors not extracted. Call extract_priors() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        priors_data = {
            "beta_0": self.beta_0_,
            "Sigma_0": self.Sigma_0_,
            "feature_names": self.feature_names_,
            "lambda_scale": self.lambda_scale,
        }

        if include_metadata:
            priors_data["metadata"] = self.prior_metadata_

        with open(filepath, "wb") as f:
            pickle.dump(priors_data, f)

        print(f"✓ Priors saved to {filepath}")

    @staticmethod
    def load_priors(filepath: Union[str, Path]) -> Dict:
        """
        Load priors from pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to priors pickle file

        Returns
        -------
        priors_data : Dict
            Dictionary with keys: beta_0, Sigma_0, feature_names, lambda_scale, metadata
        """
        with open(filepath, "rb") as f:
            priors_data = pickle.load(f)

        print(f"✓ Priors loaded from {filepath}")
        print(f"  β₀ shape: {priors_data['beta_0'].shape}")
        print(f"  Σ₀ shape: {priors_data['Sigma_0'].shape}")

        return priors_data
