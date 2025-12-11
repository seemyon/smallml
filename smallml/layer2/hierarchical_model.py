"""
Hierarchical Bayesian Model for Multi-SME Inference

This module implements Algorithm 4.3 from the SmallML framework, using PyMC
to specify and fit hierarchical logistic regression models via MCMC (NUTS).

"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.linear_model import LogisticRegression


class HierarchicalBayesianModel:
    """
    Hierarchical Bayesian model for pooling information across multiple SMEs.

    Implements Algorithm 4.3: Hierarchical Bayesian Inference via NUTS.
    Uses transfer learning priors from Layer 1 to inform population-level
    hyperparameters, then estimates SME-specific coefficients via partial pooling.

    Parameters
    ----------
    beta_0 : np.ndarray, shape (p,)
        Prior mean vector from transfer learning
    Sigma_0 : np.ndarray, shape (p, p)
        Prior covariance matrix from transfer learning
    tau : float, optional (default=2.0)
        Hyperparameter for industry-level variance prior
        Controls expected heterogeneity across SMEs
    random_seed : int, optional (default=42)
        Random seed for MCMC reproducibility

    Attributes
    ----------
    model_ : pm.Model
        PyMC model object
    trace_ : az.InferenceData
        Posterior samples from MCMC
    convergence_ : Dict
        Convergence diagnostics (R̂, ESS, etc.)
    J_ : int
        Number of SMEs
    p_ : int
        Number of features

    Examples
    --------
    >>> # Load priors from previous steps
    >>> with open('models/transfer_learning/priors.pkl', 'rb') as f:
    ...     priors = pickle.load(f)
    >>>
    >>> # Initialize model
    >>> model = HierarchicalBayesianModel(
    ...     beta_0=priors['beta_0'],
    ...     Sigma_0=priors['Sigma_0']
    ... )
    >>>
    >>> # Fit to SME data
    >>> model.fit(sme_datasets, chains=4, draws=2000, tune=1000)
    >>>
    >>> # Check convergence
    >>> model.check_convergence()
    >>>
    >>> # Make predictions
    >>> predictions = model.posterior_predictive(X_new, sme_id=0)
    """

    def __init__(
        self,
        beta_0: np.ndarray,
        Sigma_0: np.ndarray,
        tau: float = 2.0,
        random_seed: int = 42
    ):
        """Initialize hierarchical Bayesian model."""
        self.beta_0 = beta_0
        self.Sigma_0 = Sigma_0
        self.tau = tau
        self.random_seed = random_seed

        # Extract dimensions
        self.p_ = len(beta_0)

        # Extract prior standard deviations (diagonal of Sigma_0)
        self.sigma_0 = np.sqrt(np.diag(Sigma_0))

        # Placeholders
        self.model_ = None
        self.trace_ = None
        self.convergence_ = None
        self.J_ = None
        self.feature_names_ = None
        self.sme_datasets_ = None

        # Validate inputs
        if Sigma_0.shape != (self.p_, self.p_):
            raise ValueError(
                f"Sigma_0 shape mismatch. Expected ({self.p_}, {self.p_}), "
                f"got {Sigma_0.shape}"
            )

        if tau <= 0:
            raise ValueError(f"tau must be positive. Got: {tau}")

    def fit(
        self,
        sme_datasets: Dict[int, Dict[str, pd.DataFrame]],
        chains: int = 4,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.90,
        cores: Optional[int] = None,
        verbose: bool = True
    ) -> "HierarchicalBayesianModel":
        """
        Fit hierarchical Bayesian model via MCMC (NUTS).

        Implements Algorithm 4.3 from Section 4.3.2.

        Parameters
        ----------
        sme_datasets : Dict[int, Dict[str, pd.DataFrame]]
            Dictionary mapping SME index j to {'X': features, 'y': target}
        chains : int, optional (default=4)
            Number of MCMC chains to run in parallel
        draws : int, optional (default=2000)
            Number of samples per chain (N_samples in Algorithm 4.3)
        tune : int, optional (default=1000)
            Number of warmup iterations (N_warmup in Algorithm 4.3)
        target_accept : float, optional (default=0.90)
            Target acceptance rate for NUTS (higher = more accurate but slower)
        cores : int, optional (default=None)
            Number of CPU cores to use. If None, uses all available cores
        verbose : bool, optional (default=True)
            If True, print MCMC progress

        Returns
        -------
        self : HierarchicalBayesianModel
            Fitted model with populated trace_

        The NUTS sampler automatically tunes step size and mass matrix during
        the warmup phase, requiring minimal manual intervention.
        """
        self.sme_datasets_ = sme_datasets
        self.J_ = len(sme_datasets)

        if verbose:
            print(f"\n{'=' * 80}")
            print("HIERARCHICAL BAYESIAN MODEL: MCMC SAMPLING")
            print(f"{'=' * 80}")
            print(f"SMEs (J): {self.J_}")
            print(f"Features (p): {self.p_}")
            print(f"Customers per SME: {len(sme_datasets[0]['X'])}")
            print(f"Total observations: {self.J_ * len(sme_datasets[0]['X'])}")
            print(f"\nMCMC Configuration:")
            print(f"  Chains: {chains}")
            print(f"  Draws per chain: {draws}")
            print(f"  Warmup iterations: {tune}")
            print(f"  Target acceptance: {target_accept}")
            print(f"  Total samples: {chains * draws}")

        # Prepare data
        if verbose:
            print(f"\n{'=' * 80}")
            print("[Step 1/3] Preparing data...")

        X_all, y_all, sme_idx, feature_names = self._prepare_data(
            sme_datasets, verbose=verbose
        )
        self.feature_names_ = feature_names

        # Specify model
        if verbose:
            print(f"\n{'=' * 80}")
            print("[Step 2/3] Specifying PyMC hierarchical model...")

        self.model_ = self._specify_model(X_all, y_all, sme_idx, verbose=verbose)

        # Run MCMC
        if verbose:
            print(f"\n{'=' * 80}")
            print("[Step 3/3] Running MCMC sampling...")
            print(f"Start time: {pd.Timestamp.now().strftime('%H:%M:%S')}")

        with self.model_:
            self.trace_ = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                return_inferencedata=True,
                random_seed=self.random_seed
            )

        if verbose:
            print(f"End time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
            print(f"\n✓ MCMC sampling completed")
            print(f"  Total samples: {chains * draws}")
            print(f"  Warmup samples (discarded): {chains * tune}")

        return self

    def _prepare_data(
        self,
        sme_datasets: Dict[int, Dict[str, pd.DataFrame]],
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for hierarchical model.

        Stacks all SME datasets and creates index mapping each observation
        to its SME.

        Parameters
        ----------
        sme_datasets : Dict[int, Dict[str, pd.DataFrame]]
            SME datasets
        verbose : bool
            Print progress

        Returns
        -------
        X_all : np.ndarray, shape (n_total, p)
            Stacked feature matrix
        y_all : np.ndarray, shape (n_total,)
            Stacked target vector
        sme_idx : np.ndarray, shape (n_total,)
            SME index for each observation
        feature_names : List[str]
            Feature names
        """
        X_list = []
        y_list = []
        sme_idx_list = []

        J = len(sme_datasets)

        for j in range(J):
            X_j = sme_datasets[j]['X'].values
            y_j = sme_datasets[j]['y'].values
            n_j = len(X_j)

            X_list.append(X_j)
            y_list.append(y_j)
            sme_idx_list.append(np.repeat(j, n_j))

            if verbose and (j + 1) % 5 == 0:
                print(f"  Processed SME {j + 1}/{J}...")

        # Stack
        X_all = np.vstack(X_list)
        y_all = np.hstack(y_list)
        sme_idx = np.hstack(sme_idx_list)

        # Get feature names
        feature_names = list(sme_datasets[0]['X'].columns)

        if verbose:
            print(f"\n✓ Data prepared")
            print(f"  X_all shape: {X_all.shape}")
            print(f"  y_all shape: {y_all.shape}")
            print(f"  Churn rate: {y_all.mean():.3f}")

        return X_all, y_all, sme_idx, feature_names

    def _specify_model(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        sme_idx: np.ndarray,
        verbose: bool = True
    ) -> pm.Model:
        """
        Specify hierarchical model in PyMC (Algorithm 4.3, Step 1).

        Three-level hierarchy:
        Level 1 (Population): μ_industry, σ_industry
        Level 2 (SME-specific): β_j for j = 1,...,J
        Level 3 (Observations): y_ij ~ Bernoulli(logit^{-1}(β_j^T x_ij))

        Parameters
        ----------
        X_all : np.ndarray, shape (n_total, p)
            Stacked features
        y_all : np.ndarray, shape (n_total,)
            Stacked targets
        sme_idx : np.ndarray, shape (n_total,)
            SME index for each observation
        verbose : bool
            Print model structure

        Returns
        -------
        model : pm.Model
            PyMC model object
        """
        import pytensor.tensor as pt

        # Ensure data is in correct format and check for NaNs
        X_all = X_all.astype(np.float64)
        y_all = y_all.astype(np.int32)
        sme_idx = sme_idx.astype(np.int32)

        # Check for NaNs and handle them
        if np.any(np.isnan(X_all)):
            if verbose:
                print(f"  Warning: Found NaN values in X_all, filling with 0")
            X_all = np.nan_to_num(X_all, nan=0.0)

        if np.any(np.isnan(y_all)):
            raise ValueError("NaN values found in target variable y_all")

        with pm.Model() as model:
            # Convert data to PyTensor constants (faster than pm.Data for fixed data)
            # We use constants since we're not doing mini-batch sampling
            X_const = pt.as_tensor_variable(X_all)
            sme_idx_const = pt.as_tensor_variable(sme_idx)

            # Level 1: Population hyperpriors (informed by transfer learning)
            # μ_industry ~ Normal(β₀, √Σ₀)
            mu_industry = pm.Normal(
                'mu_industry',
                mu=self.beta_0,
                sigma=self.sigma_0,
                shape=self.p_,
                initval=self.beta_0  # Initialize at prior mean
            )

            # σ_industry ~ HalfNormal(τ)
            sigma_industry = pm.HalfNormal(
                'sigma_industry',
                sigma=self.tau,
                initval=1.0  # Initialize at reasonable value
            )

            # Level 2: SME-specific parameters (non-centered parameterization)
            # Non-centered to avoid Neal's funnel and improve MCMC convergence
            # β_j_raw ~ Normal(0, 1), then β_j = μ_industry + σ_industry * β_j_raw
            beta_j_raw = pm.Normal(
                'beta_j_raw',
                mu=0,
                sigma=1,
                shape=(self.J_, self.p_),
                initval=np.zeros((self.J_, self.p_))
            )

            # Deterministic transformation (decorrelates parameters)
            beta_j = pm.Deterministic(
                'beta_j',
                mu_industry + sigma_industry * beta_j_raw
            )

            # Level 3: Observations
            # For each observation, select corresponding SME's coefficients
            beta_obs = beta_j[sme_idx_const]  # Shape: (n_total, p)

            # Compute linear predictor: X^T β
            eta = pt.sum(X_const * beta_obs, axis=1)

            # Bernoulli likelihood with logit link
            y_obs = pm.Bernoulli(
                'y_obs',
                logit_p=eta,
                observed=y_all
            )

        if verbose:
            print(f"\n✓ PyMC model specified")
            print(f"  Parameters:")
            print(f"    mu_industry: {self.p_} coefficients (population mean)")
            print(f"    sigma_industry: 1 scalar (between-SME variance)")
            print(f"    beta_j_raw: {self.J_} × {self.p_} = {self.J_ * self.p_} " +
                  f"coefficients (non-centered)")
            print(f"    beta_j: {self.J_} × {self.p_} deterministic (transformed)")
            print(f"  Total parameters: {self.p_ + 1 + self.J_ * self.p_}")
            print(f"  Total observations: {len(y_all)}")

        return model

    def check_convergence(
        self,
        verbose: bool = True
    ) -> Dict[str, bool]:
        """
        Check MCMC convergence using R̂ and ESS diagnostics.

        Implements Table 4.8 criteria from Section 4.3.2.

        Parameters
        ----------
        verbose : bool, optional (default=True)
            If True, print convergence summary

        Returns
        -------
        convergence : Dict[str, bool]
            Dictionary with convergence checks:
            - 'rhat_ok': All R̂ < 1.01
            - 'ess_ok': All ESS > 400
            - 'all_ok': Both criteria met

        Raises
        ------
        ValueError
            If model hasn't been fitted yet
        """
        if self.trace_ is None:
            raise ValueError(
                "Model not fitted. Call fit() before checking convergence."
            )

        if verbose:
            print(f"\n{'=' * 80}")
            print("CONVERGENCE DIAGNOSTICS")
            print(f"{'=' * 80}")

        # Compute R̂ (Gelman-Rubin statistic)
        rhat = az.rhat(self.trace_)

        # Extract R̂ values for all parameters
        rhat_values = []
        for var in rhat.data_vars:
            rhat_var = rhat[var].values.flatten()
            rhat_values.extend(rhat_var)

        rhat_max = np.max(rhat_values)
        rhat_ok = rhat_max < 1.01

        if verbose:
            print(f"\n1. R̂ (Gelman-Rubin Statistic)")
            print(f"   Criterion: R̂ < 1.01 for all parameters")
            print(f"   Max R̂: {rhat_max:.6f}")
            print(f"   Status: {'✓ PASS' if rhat_ok else '✗ FAIL'}")

        # Compute ESS (Effective Sample Size)
        ess = az.ess(self.trace_)

        # Extract ESS values for all parameters
        ess_values = []
        for var in ess.data_vars:
            ess_var = ess[var].values.flatten()
            ess_values.extend(ess_var)

        ess_min = np.min(ess_values)
        ess_ok = ess_min > 400

        if verbose:
            print(f"\n2. ESS (Effective Sample Size)")
            print(f"   Criterion: ESS > 400 for all parameters")
            print(f"   Min ESS: {ess_min:.0f}")
            print(f"   Status: {'✓ PASS' if ess_ok else '✗ FAIL'}")

        # Overall convergence
        all_ok = rhat_ok and ess_ok

        if verbose:
            print(f"\n{'=' * 80}")
            if all_ok:
                print("✓ ALL CONVERGENCE CRITERIA MET")
            else:
                print("✗ CONVERGENCE ISSUES DETECTED")
                print("\nTroubleshooting:")
                if not rhat_ok:
                    print("  - R̂ > 1.01: Chains haven't converged")
                    print("    → Increase draws or tune iterations")
                if not ess_ok:
                    print("  - ESS < 400: High autocorrelation")
                    print("    → Increase draws or check posterior geometry")
            print(f"{'=' * 80}")

        # Store convergence info
        self.convergence_ = {
            'rhat_ok': bool(rhat_ok),
            'rhat_max': float(rhat_max),
            'ess_ok': bool(ess_ok),
            'ess_min': float(ess_min),
            'all_ok': bool(all_ok)
        }

        return self.convergence_

    def extract_posterior_means(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract posterior mean estimates for all parameters.

        Returns
        -------
        mu_industry_mean : np.ndarray, shape (p,)
            Posterior mean of population-level coefficients
        sigma_industry_mean : float
            Posterior mean of between-SME standard deviation
        beta_j_mean : np.ndarray, shape (J, p)
            Posterior means of SME-specific coefficients

        Raises
        ------
        ValueError
            If model hasn't been fitted yet
        """
        if self.trace_ is None:
            raise ValueError(
                "Model not fitted. Call fit() before extracting means."
            )

        # Extract posterior samples
        mu_samples = self.trace_.posterior['mu_industry'].values  # (chains, draws, p)
        sigma_samples = self.trace_.posterior['sigma_industry'].values  # (chains, draws)
        beta_samples = self.trace_.posterior['beta_j'].values  # (chains, draws, J, p)

        # Compute means across chains and draws
        mu_industry_mean = mu_samples.mean(axis=(0, 1))  # Shape: (p,)
        sigma_industry_mean = sigma_samples.mean()  # Scalar
        beta_j_mean = beta_samples.mean(axis=(0, 1))  # Shape: (J, p)

        return mu_industry_mean, sigma_industry_mean, beta_j_mean

    def compute_mle_estimates(
        self,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute MLE estimates for comparison with hierarchical estimates.

        Fits independent logistic regressions for each SME (no pooling).

        Parameters
        ----------
        verbose : bool, optional (default=True)
            If True, print progress

        Returns
        -------
        beta_j_mle : np.ndarray, shape (J, p)
            MLE coefficient estimates for each SME

        Notes
        -----
        Uses scikit-learn's LogisticRegression with no regularization
        (penalty=None) to obtain true maximum likelihood estimates.
        """
        if self.sme_datasets_ is None:
            raise ValueError(
                "No SME datasets available. Call fit() first."
            )

        if verbose:
            print(f"\nComputing MLE estimates (independent models)...")

        beta_j_mle = np.zeros((self.J_, self.p_))

        for j in range(self.J_):
            X_j = self.sme_datasets_[j]['X'].values
            y_j = self.sme_datasets_[j]['y'].values

            # Handle NaN values (same as PyMC model)
            if np.any(np.isnan(X_j)):
                X_j = np.nan_to_num(X_j, nan=0.0)

            # Fit logistic regression (no regularization)
            lr = LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_seed
            )
            lr.fit(X_j, y_j)

            beta_j_mle[j] = lr.coef_[0]

            if verbose and (j + 1) % 5 == 0:
                print(f"  Fitted SME {j + 1}/{self.J_}...")

        if verbose:
            print(f"✓ MLE estimates computed for {self.J_} SMEs")

        return beta_j_mle

    def posterior_predictive(
        self,
        X_new: np.ndarray,
        sme_id: int,
        n_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Generate posterior predictive distribution for new customers.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, p)
            Feature matrix for new customers
        sme_id : int
            SME identifier (0 to J-1)
        n_samples : int, optional (default=1000)
            Number of posterior samples to use

        Returns
        -------
        predictions : Dict[str, np.ndarray]
            Dictionary containing:
            - 'mean': Mean predicted probabilities, shape (n_new,)
            - 'std': Standard deviation, shape (n_new,)
            - 'lower_90': 5th percentile, shape (n_new,)
            - 'upper_90': 95th percentile, shape (n_new,)
            - 'samples': Full posterior predictive samples, shape (n_samples, n_new)

        Raises
        ------
        ValueError
            If model hasn't been fitted or sme_id is invalid
        """
        if self.trace_ is None:
            raise ValueError(
                "Model not fitted. Call fit() before making predictions."
            )

        if not (0 <= sme_id < self.J_):
            raise ValueError(
                f"Invalid sme_id: {sme_id}. Must be in [0, {self.J_-1}]"
            )

        # Handle NaN values in input (same as during training)
        X_new = X_new.astype(np.float64)
        if np.any(np.isnan(X_new)):
            X_new = np.nan_to_num(X_new, nan=0.0)

        # Get posterior samples for beta_j
        beta_samples = self.trace_.posterior['beta_j'].values  # (chains, draws, J, p)

        # Flatten chains and draws, select SME
        beta_samples = beta_samples.reshape(-1, self.J_, self.p_)  # (n_samples_total, J, p)
        beta_j_samples = beta_samples[:, sme_id, :]  # (n_samples_total, p)

        # Subsample if requested
        if n_samples < len(beta_j_samples):
            idx = np.random.choice(len(beta_j_samples), size=n_samples, replace=False)
            beta_j_samples = beta_j_samples[idx]

        # Compute predictions for each posterior sample
        logit_p = X_new @ beta_j_samples.T  # (n_new, n_samples)
        p_samples = 1 / (1 + np.exp(-logit_p))  # (n_new, n_samples)

        # Compute summaries
        predictions = {
            'mean': p_samples.mean(axis=1),
            'std': p_samples.std(axis=1),
            'lower_90': np.percentile(p_samples, 5, axis=1),
            'upper_90': np.percentile(p_samples, 95, axis=1),
            'samples': p_samples
        }

        return predictions

    def save_trace(
        self,
        filepath: str,
        verbose: bool = True
    ) -> None:
        """
        Save MCMC trace to NetCDF format.

        Parameters
        ----------
        filepath : str
            Path to save trace (should end with .nc)
        verbose : bool, optional (default=True)
            If True, print confirmation

        Notes
        -----
        Uses arviz's to_netcdf() method, which is the standard format
        for PyMC traces. Can be loaded with arviz.from_netcdf().
        """
        if self.trace_ is None:
            raise ValueError(
                "No trace to save. Call fit() first."
            )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.trace_.to_netcdf(filepath)

        if verbose:
            print(f"\n✓ Trace saved to {filepath}")
            print(f"  File size: {filepath.stat().st_size / 1e6:.1f} MB")

    @staticmethod
    def load_trace(
        filepath: str,
        verbose: bool = True
    ) -> az.InferenceData:
        """
        Load MCMC trace from NetCDF format.

        Parameters
        ----------
        filepath : str
            Path to trace file
        verbose : bool, optional (default=True)
            If True, print confirmation

        Returns
        -------
        trace : az.InferenceData
            Loaded trace

        Raises
        ------
        FileNotFoundError
            If filepath doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Trace file not found: {filepath}")

        trace = az.from_netcdf(filepath)

        if verbose:
            print(f"✓ Trace loaded from {filepath}")
            # Extract some basic info
            chains = trace.posterior.dims['chain']
            draws = trace.posterior.dims['draw']
            print(f"  Chains: {chains}")
            print(f"  Draws per chain: {draws}")
            print(f"  Total samples: {chains * draws}")

        return trace
