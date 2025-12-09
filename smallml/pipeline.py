"""SmallML Pipeline - Main User Interface"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .layer2.hierarchical_model import HierarchicalBayesianModel
from .layer3.conformal_predictor import ConformalPredictor


class SmallMLPipeline:
    """
    End-to-end SmallML pipeline for small-data predictive analytics.

    Combines hierarchical Bayesian inference and conformal prediction
    to enable reliable predictions with 50-200 observations per entity.

    Parameters
    ----------
    use_pretrained_priors : bool, default=True
        If True, loads pre-trained priors from package data.
        If False, priors must be provided during fit().

    quick_mode : bool, default=False
        If True, uses faster MCMC settings for prototyping.
        Set False for production use.

    random_seed : int, default=42
        Random seed for reproducibility.

    Examples
    --------
    >>> from smallml import SmallMLPipeline
    >>>
    >>> # Prepare multi-entity data
    >>> sme_data = {
    ...     'store_1': df1,  # 80 customers with features + 'churned' column
    ...     'store_2': df2,  # 120 customers
    ...     'store_3': df3,  # 95 customers
    ... }
    >>>
    >>> # Fit pipeline (automatic convergence validation)
    >>> pipeline = SmallMLPipeline()
    >>> pipeline.fit(sme_data, target_col='churned')
    >>>
    >>> # Make predictions with uncertainty
    >>> predictions = pipeline.predict(new_customers, sme_id='store_1')
    """

    def __init__(
        self,
        use_pretrained_priors: bool = True,
        quick_mode: bool = False,
        random_seed: int = 42
    ):
        self.use_pretrained_priors = use_pretrained_priors
        self.quick_mode = quick_mode
        self.random_seed = random_seed

        # Load pre-trained priors if requested
        if use_pretrained_priors:
            self.priors = self._load_pretrained_priors()
        else:
            self.priors = None

        # Will be initialized during fit()
        self.hierarchical_model = None
        self.conformal_predictor = None
        self.feature_names = None
        self.target_col = None
        self.sme_names = None

    def _load_pretrained_priors(self) -> Dict:
        """Load pre-trained priors from package data."""
        priors_path = Path(__file__).parent / "data" / "priors_churn.pkl"

        if not priors_path.exists():
            warnings.warn(
                f"Pre-trained priors not found at {priors_path}. "
                "You can add your own priors to smallml/data/priors_churn.pkl "
                "or set use_pretrained_priors=False and provide priors during fit()."
            )
            return None

        with open(priors_path, 'rb') as f:
            priors = pickle.load(f)

        print(f"✓ Loaded pre-trained priors for {len(priors['beta_0'])} features")
        return priors

    def fit(
        self,
        sme_data: Dict[str, pd.DataFrame],
        target_col: str = 'churned',
        calibration_fraction: float = 0.25,
        validate_convergence: bool = True
    ) -> 'SmallMLPipeline':
        """
        Fit the SmallML pipeline on multi-entity data.

        Parameters
        ----------
        sme_data : dict of {str: pd.DataFrame}
            Dictionary mapping entity names to dataframes.
            Each DF must have same features + binary target column.
            Minimum 3 entities, 50+ observations per entity recommended.

        target_col : str, default='churned'
            Name of binary target column (0/1).

        calibration_fraction : float, default=0.25
            Fraction of data reserved for conformal calibration (0.2-0.3).

        validate_convergence : bool, default=True
            If True, raises error if MCMC doesn't converge (R̂ ≥ 1.01).

        Returns
        -------
        self : SmallMLPipeline
            Fitted pipeline.
        """
        # Validate inputs
        self._validate_sme_data(sme_data, target_col)
        self.target_col = target_col
        self.sme_names = list(sme_data.keys())

        # Extract feature names
        first_df = list(sme_data.values())[0]
        self.feature_names = [c for c in first_df.columns if c != target_col]

        print(f"\n{'='*70}")
        print(f"SmallML Pipeline: Fitting on {len(sme_data)} entities")
        print(f"{'='*70}\n")

        # Split data into training and calibration
        train_data, cal_data = self._split_train_calibration(
            sme_data, calibration_fraction
        )

        # Fit Layer 2 (Hierarchical Bayesian)
        print("\n[Layer 2] Fitting Hierarchical Bayesian Model...")
        self._fit_hierarchical(train_data)

        # Validate convergence
        if validate_convergence:
            self._validate_convergence()

        # Calibrate Layer 3 (Conformal Prediction)
        print("\n[Layer 3] Calibrating Conformal Predictor...")
        self._fit_conformal(cal_data)

        print(f"\n{'='*70}")
        print("✓ SmallML Pipeline fitted successfully!")
        print(f"{'='*70}\n")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        sme_id: Optional[str] = None,
        return_uncertainty: bool = True,
        n_posterior_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Make predictions with Bayesian + conformal uncertainty.

        Parameters
        ----------
        X : pd.DataFrame
            New data with same features as training data.

        sme_id : str, optional
            Entity ID for predictions. If None, uses first entity.

        return_uncertainty : bool, default=True
            If True, returns Bayesian std and conformal prediction sets.

        n_posterior_samples : int, default=1000
            Number of posterior samples for predictions.

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame with columns:
            - prediction: Point estimate (probability)
            - bayesian_std: Posterior standard deviation (if return_uncertainty)
            - bayesian_lower_90: Lower bound of 90% credible interval
            - bayesian_upper_90: Upper bound of 90% credible interval
            - conformal_set: Prediction set {0}, {1}, or {0,1}
            - conformal_set_size: 1 (certain) or 2 (uncertain)
        """
        if self.hierarchical_model is None:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")

        # Get entity ID
        if sme_id is None:
            sme_id = self.sme_names[0]
            warnings.warn(f"No sme_id specified. Using '{sme_id}'.")

        sme_idx = self.sme_names.index(sme_id)

        # Posterior predictive from hierarchical model
        posterior = self.hierarchical_model.posterior_predictive(
            X[self.feature_names].values,
            sme_id=sme_idx,
            n_samples=n_posterior_samples
        )

        predictions = pd.DataFrame({
            'prediction': posterior['mean']
        })

        if return_uncertainty:
            predictions['bayesian_std'] = posterior['std']
            predictions['bayesian_lower_90'] = posterior['lower_90']
            predictions['bayesian_upper_90'] = posterior['upper_90']

            # Conformal prediction sets
            sets = self.conformal_predictor.predict_set(posterior['mean'])
            predictions['conformal_set'] = [str(s) for s in sets]
            predictions['conformal_set_size'] = [len(s) for s in sets]

        return predictions

    def get_convergence_diagnostics(self) -> pd.DataFrame:
        """
        Return MCMC convergence diagnostics (R̂, ESS).

        Returns
        -------
        diagnostics : pd.DataFrame
            DataFrame with columns: parameter, r_hat, ess_bulk, ess_tail
        """
        if self.hierarchical_model is None:
            raise RuntimeError("Model not fitted yet.")

        return self.hierarchical_model.check_convergence()

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sme_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate on test data.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.

        y_test : pd.Series
            Test labels.

        sme_id : str, optional
            Entity ID for predictions. If None, uses first entity.

        Returns
        -------
        metrics : dict
            Dictionary with keys:
            - auc: Area under ROC curve
            - accuracy: Classification accuracy at 0.5 threshold
            - f1_score: F1 score
            - conformal_coverage: Empirical coverage rate (should be ~90%)
            - mean_set_size: Average prediction set size (1.0-2.0)
        """
        # Get predictions
        preds = self.predict(X_test, sme_id=sme_id, return_uncertainty=True)

        # Compute prediction metrics
        y_pred = (preds['prediction'] > 0.5).astype(int)

        metrics = {
            'auc': roc_auc_score(y_test, preds['prediction']),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
        }

        # Compute conformal coverage
        coverage = self.conformal_predictor.validate_coverage(
            y_test.values,
            preds['prediction'].values
        )
        metrics['conformal_coverage'] = coverage['empirical_coverage']
        metrics['mean_set_size'] = preds['conformal_set_size'].mean()

        return metrics

    def save(self, filepath: str):
        """Save fitted pipeline to disk (pickle)."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SmallMLPipeline':
        """Load fitted pipeline from disk."""
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"✓ Pipeline loaded from {filepath}")
        return pipeline

    # ---- Private methods ----

    def _validate_sme_data(self, sme_data: Dict, target_col: str):
        """Validate entity data structure and content."""
        if not isinstance(sme_data, dict):
            raise TypeError("sme_data must be dict of {entity_name: dataframe}")

        if len(sme_data) < 3:
            warnings.warn(
                f"Only {len(sme_data)} entities provided. "
                "Recommend at least 3 for stable pooling, 5+ ideal."
            )

        # Check all dataframes
        for sme_name, df in sme_data.items():
            if target_col not in df.columns:
                raise ValueError(
                    f"Entity '{sme_name}' missing target column '{target_col}'"
                )

            if len(df) < 30:
                warnings.warn(
                    f"Entity '{sme_name}' has only {len(df)} observations. "
                    "Minimum 30, recommend 50+ for reliable inference."
                )

    def _split_train_calibration(self, sme_data, cal_frac):
        """Split each entity's data into training and calibration sets."""
        train_data = {}
        cal_data = {}

        for sme_name, df in sme_data.items():
            n_cal = int(len(df) * cal_frac)

            # Random split
            df_shuffled = df.sample(frac=1, random_state=self.random_seed)
            cal_data[sme_name] = df_shuffled.iloc[:n_cal]
            train_data[sme_name] = df_shuffled.iloc[n_cal:]

        n_train = sum(len(d) for d in train_data.values())
        n_cal = sum(len(d) for d in cal_data.values())

        print(f"✓ Split data: {len(train_data)} entities, "
              f"{n_train} train, {n_cal} calibration observations")

        return train_data, cal_data

    def _fit_hierarchical(self, train_data):
        """Fit hierarchical Bayesian model (Layer 2)."""

        # Convert to format expected by HierarchicalBayesianModel
        sme_datasets = {}
        for i, (sme_name, df) in enumerate(train_data.items()):
            sme_datasets[i] = {
                'X': df[self.feature_names],
                'y': df[self.target_col]
            }

        # MCMC settings
        n_chains = 2 if self.quick_mode else 4
        n_draws = 500 if self.quick_mode else 2000
        n_tune = 500 if self.quick_mode else 1000

        # Initialize and fit
        self.hierarchical_model = HierarchicalBayesianModel(
            beta_0=self.priors['beta_0'] if self.priors else None,
            Sigma_0=self.priors['Sigma_0'] if self.priors else None,
            tau=2.0,
            random_seed=self.random_seed
        )

        self.hierarchical_model.fit(
            sme_datasets,
            chains=n_chains,
            draws=n_draws,
            tune=n_tune
        )

        print("  ✓ Hierarchical model fitted")

    def _validate_convergence(self):
        """Check MCMC convergence and raise error if failed."""
        diagnostics = self.hierarchical_model.check_convergence()

        # Check R̂ < 1.01 for all parameters
        max_rhat = diagnostics['r_hat'].max()
        if max_rhat >= 1.01:
            failed_params = diagnostics[diagnostics['r_hat'] >= 1.01]
            raise RuntimeError(
                f"MCMC convergence failed! R̂ ≥ 1.01 for {len(failed_params)} parameters.\n"
                f"Max R̂ = {max_rhat:.4f}\n"
                f"Failed parameters:\n{failed_params}\n\n"
                "Try increasing MCMC draws: pipeline = SmallMLPipeline(quick_mode=False)"
            )

        # Check ESS > 400 for all parameters
        min_ess = diagnostics['ess_bulk'].min()
        if min_ess < 400:
            warnings.warn(
                f"Low effective sample size detected (ESS = {min_ess:.0f}). "
                "Consider increasing MCMC draws for more reliable inference."
            )

        print(f"  ✓ Convergence validated (max R̂ = {max_rhat:.4f}, "
              f"min ESS = {min_ess:.0f})")

    def _fit_conformal(self, cal_data):
        """Calibrate conformal predictor (Layer 3)."""

        # Get predictions on calibration data
        cal_predictions = []
        cal_labels = []

        for i, (sme_name, df) in enumerate(cal_data.items()):
            posterior = self.hierarchical_model.posterior_predictive(
                df[self.feature_names].values,
                sme_id=i,
                n_samples=1000
            )
            cal_predictions.extend(posterior['mean'])
            cal_labels.extend(df[self.target_col].values)

        # Calibrate
        self.conformal_predictor = ConformalPredictor(
            alpha=0.10,  # 90% coverage
            random_seed=self.random_seed
        )

        q_hat = self.conformal_predictor.calibrate(
            np.array(cal_labels),
            np.array(cal_predictions)
        )

        print(f"  ✓ Conformal predictor calibrated (q̂ = {q_hat:.4f})")
