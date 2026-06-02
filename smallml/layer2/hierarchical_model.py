"""
Hierarchical Bayesian Model for Multi-Entity Inference

Implements hierarchical (partial-pooling) regression via PyMC and the NUTS
sampler. Information is pooled across multiple *entities* (the user-defined unit
of grouping) so that data-poor entities borrow statistical strength from the
population.

The hierarchy is the same for every supported task; only the observation model at
the lowest level changes with the chosen ``likelihood``:

    Level 1 (population)  : mu_population, sigma_population  (informed by Layer 1)
    Level 2 (per-entity)  : beta_j = mu_population + sigma_population * beta_j_raw
    Level 3 (observations):
        binary     : y ~ Bernoulli(logit_p = xᵀβ_j)
        regression : y ~ Normal(mu = xᵀβ_j, sigma = sigma_y)
        count      : y ~ Poisson(mu = exp(xᵀβ_j))
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
from typing import Dict, Tuple, Optional, List

VALID_LIKELIHOODS = ("binary", "regression", "count")


class HierarchicalBayesianModel:
    """
    Hierarchical Bayesian model that pools information across multiple entities.

    Uses transfer-learning priors from Layer 1 to inform the population-level
    hyperparameters, then estimates entity-specific coefficients via partial
    pooling and MCMC (NUTS).

    Parameters
    ----------
    beta_0 : np.ndarray, shape (p,)
        Prior mean vector (transfer-learning prior).
    Sigma_0 : np.ndarray, shape (p, p)
        Prior covariance matrix (transfer-learning prior).
    tau : float, default=2.0
        Scale of the half-normal prior on the between-entity standard deviation
        ``sigma_population``. Controls expected heterogeneity across entities.
    likelihood : {'binary', 'regression', 'count'}, default='binary'
        Observation model at the lowest level of the hierarchy.
    random_seed : int, default=42
        Random seed for MCMC reproducibility.

    Attributes
    ----------
    model_ : pm.Model
    trace_ : az.InferenceData
    convergence_ : dict
    J_ : int
        Number of entity groups.
    p_ : int
        Number of features.
    """

    def __init__(
        self,
        beta_0: np.ndarray,
        Sigma_0: np.ndarray,
        tau: float = 2.0,
        likelihood: str = "binary",
        random_seed: int = 42,
    ):
        if likelihood not in VALID_LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {VALID_LIKELIHOODS}, got '{likelihood}'"
            )

        self.beta_0 = beta_0
        self.Sigma_0 = Sigma_0
        self.tau = tau
        self.likelihood = likelihood
        self.random_seed = random_seed

        self.p_ = len(beta_0)
        self.sigma_0 = np.sqrt(np.diag(Sigma_0))

        # Placeholders
        self.model_ = None
        self.trace_ = None
        self.convergence_ = None
        self.J_ = None
        self.feature_names_ = None
        self.entity_datasets_ = None
        self._y_std = 1.0  # observation-noise scale hint (regression)

        if Sigma_0.shape != (self.p_, self.p_):
            raise ValueError(
                f"Sigma_0 shape mismatch. Expected ({self.p_}, {self.p_}), "
                f"got {Sigma_0.shape}"
            )
        if tau <= 0:
            raise ValueError(f"tau must be positive. Got: {tau}")

    def fit(
        self,
        entity_datasets: Dict[int, Dict[str, pd.DataFrame]],
        chains: int = 4,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.90,
        cores: Optional[int] = None,
        verbose: bool = True,
    ) -> "HierarchicalBayesianModel":
        """
        Fit the hierarchical model via MCMC (NUTS).

        Parameters
        ----------
        entity_datasets : dict of {int: {'X': DataFrame, 'y': Series}}
            Mapping from entity index ``j`` to its feature matrix and target.
        chains, draws, tune : int
            MCMC configuration.
        target_accept : float, default=0.90
            NUTS target acceptance rate.
        cores : int, optional
            CPU cores (defaults to all available).
        verbose : bool, default=True

        Returns
        -------
        self
        """
        self.entity_datasets_ = entity_datasets
        self.J_ = len(entity_datasets)

        if verbose:
            print(f"\n{'=' * 80}")
            print("HIERARCHICAL BAYESIAN MODEL: MCMC SAMPLING")
            print(f"{'=' * 80}")
            print(f"Entity groups (J): {self.J_}")
            print(f"Features (p): {self.p_}")
            print(f"Likelihood: {self.likelihood}")
            print(f"Observations in first entity: {len(entity_datasets[0]['X'])}")
            print("\nMCMC Configuration:")
            print(f"  Chains: {chains}")
            print(f"  Draws per chain: {draws}")
            print(f"  Warmup iterations: {tune}")
            print(f"  Target acceptance: {target_accept}")

        X_all, y_all, entity_idx, feature_names = self._prepare_data(
            entity_datasets, verbose=verbose
        )
        self.feature_names_ = feature_names

        if self.likelihood == "regression":
            y_std = float(np.nanstd(y_all))
            self._y_std = y_std if y_std > 1e-8 else 1.0

        self.model_ = self._specify_model(X_all, y_all, entity_idx, verbose=verbose)

        if verbose:
            print(f"\n{'=' * 80}")
            print("Running MCMC sampling...")
            print(f"Start time: {pd.Timestamp.now().strftime('%H:%M:%S')}")

        with self.model_:
            self.trace_ = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                return_inferencedata=True,
                random_seed=self.random_seed,
            )

        if verbose:
            print(f"End time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
            print("\n✓ MCMC sampling completed")

        return self

    def _prepare_data(
        self,
        entity_datasets: Dict[int, Dict[str, pd.DataFrame]],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Stack all entity datasets and build a per-observation entity index."""
        X_list, y_list, entity_idx_list = [], [], []
        J = len(entity_datasets)

        for j in range(J):
            X_j = entity_datasets[j]["X"].values
            y_j = entity_datasets[j]["y"].values
            X_list.append(X_j)
            y_list.append(y_j)
            entity_idx_list.append(np.repeat(j, len(X_j)))

        X_all = np.vstack(X_list)
        y_all = np.hstack(y_list)
        entity_idx = np.hstack(entity_idx_list)
        feature_names = list(entity_datasets[0]["X"].columns)

        if verbose:
            print("\n✓ Data prepared")
            print(f"  X_all shape: {X_all.shape}")
            print(f"  Target mean: {np.nanmean(y_all):.3f}")

        return X_all, y_all, entity_idx, feature_names

    def _specify_model(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        entity_idx: np.ndarray,
        verbose: bool = True,
    ) -> pm.Model:
        """Specify the PyMC hierarchical model (observation model per likelihood)."""
        import pytensor.tensor as pt

        X_all = X_all.astype(np.float64)
        entity_idx = entity_idx.astype(np.int32)

        if np.any(np.isnan(X_all)):
            if verbose:
                print("  Warning: NaN values in features, filling with 0")
            X_all = np.nan_to_num(X_all, nan=0.0)
        if np.any(pd.isnull(y_all)):
            raise ValueError("NaN values found in target variable")

        # Cast target appropriately for the likelihood.
        if self.likelihood == "binary":
            y_all = y_all.astype(np.int32)
        elif self.likelihood == "count":
            y_all = y_all.astype(np.int64)
        else:  # regression
            y_all = y_all.astype(np.float64)

        with pm.Model() as model:
            X_const = pt.as_tensor_variable(X_all)
            entity_idx_const = pt.as_tensor_variable(entity_idx)

            # Level 1: population hyperpriors (informed by transfer learning)
            mu_population = pm.Normal(
                "mu_population",
                mu=self.beta_0,
                sigma=self.sigma_0,
                shape=self.p_,
                initval=self.beta_0,
            )
            sigma_population = pm.HalfNormal(
                "sigma_population", sigma=self.tau, initval=1.0
            )

            # Level 2: entity-specific coefficients (non-centered)
            beta_j_raw = pm.Normal(
                "beta_j_raw",
                mu=0,
                sigma=1,
                shape=(self.J_, self.p_),
                initval=np.zeros((self.J_, self.p_)),
            )
            beta_j = pm.Deterministic(
                "beta_j", mu_population + sigma_population * beta_j_raw
            )

            # Level 3: observation model
            beta_obs = beta_j[entity_idx_const]  # (n_total, p)
            eta = pt.sum(X_const * beta_obs, axis=1)  # linear predictor

            if self.likelihood == "binary":
                pm.Bernoulli("y_obs", logit_p=eta, observed=y_all)
            elif self.likelihood == "regression":
                sigma_y = pm.HalfNormal("sigma_y", sigma=self._y_std)
                pm.Normal("y_obs", mu=eta, sigma=sigma_y, observed=y_all)
            else:  # count
                pm.Poisson("y_obs", mu=pt.exp(eta), observed=y_all)

        if verbose:
            print("\n✓ PyMC model specified")
            print(f"  Likelihood: {self.likelihood}")
            print(f"  Total parameters: {self.p_ + 1 + self.J_ * self.p_}")
            print(f"  Total observations: {len(y_all)}")

        return model

    def check_convergence(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Check MCMC convergence via R̂ and ESS.

        Returns a dict with ``rhat_ok``/``rhat_max``/``ess_ok``/``ess_min``/``all_ok``.
        Criteria: R̂ < 1.01 and ESS > 400 for all parameters.
        """
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        rhat = az.rhat(self.trace_)
        rhat_values = []
        for var in rhat.data_vars:
            rhat_values.extend(rhat[var].values.flatten())
        rhat_max = float(np.max(rhat_values))
        rhat_ok = rhat_max < 1.01

        ess = az.ess(self.trace_)
        ess_values = []
        for var in ess.data_vars:
            ess_values.extend(ess[var].values.flatten())
        ess_min = float(np.min(ess_values))
        ess_ok = ess_min > 400

        all_ok = rhat_ok and ess_ok

        if verbose:
            print(f"\n{'=' * 80}")
            print("CONVERGENCE DIAGNOSTICS")
            print(f"{'=' * 80}")
            print(f"  Max R̂:  {rhat_max:.6f}  ({'PASS' if rhat_ok else 'FAIL'})")
            print(f"  Min ESS: {ess_min:.0f}  ({'PASS' if ess_ok else 'FAIL'})")

        self.convergence_ = {
            "rhat_ok": bool(rhat_ok),
            "rhat_max": rhat_max,
            "ess_ok": bool(ess_ok),
            "ess_min": ess_min,
            "all_ok": bool(all_ok),
        }
        return self.convergence_

    def extract_posterior_means(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """Return posterior means of (mu_population, sigma_population, beta_j)."""
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        mu_samples = self.trace_.posterior["mu_population"].values
        sigma_samples = self.trace_.posterior["sigma_population"].values
        beta_samples = self.trace_.posterior["beta_j"].values

        return (
            mu_samples.mean(axis=(0, 1)),
            float(sigma_samples.mean()),
            beta_samples.mean(axis=(0, 1)),
        )

    def posterior_predictive(
        self,
        X_new: np.ndarray,
        entity_id: int,
        n_samples: int = 1000,
        credible_level: float = 0.90,
    ) -> Dict[str, np.ndarray]:
        """
        Posterior predictive distribution of the expected outcome for new rows.

        The returned ``mean`` is the expected outcome in the task's natural space:
        probability (binary), value (regression), or expected count (count). The
        credible interval reflects epistemic (parameter) uncertainty.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, p)
        entity_id : int
            Entity index (0 .. J-1).
        n_samples : int, default=1000
            Number of posterior draws to use.
        credible_level : float, default=0.90
            Central credible-interval mass (e.g. 0.90 -> 5th/95th percentiles).

        Returns
        -------
        dict with keys 'mean', 'std', 'lower', 'upper', 'samples'
            ``samples`` has shape (n_new, n_used_samples).
        """
        if self.trace_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if not (0 <= entity_id < self.J_):
            raise ValueError(
                f"Invalid entity_id: {entity_id}. Must be in [0, {self.J_ - 1}]"
            )

        X_new = X_new.astype(np.float64)
        if np.any(np.isnan(X_new)):
            X_new = np.nan_to_num(X_new, nan=0.0)

        beta_samples = self.trace_.posterior["beta_j"].values
        beta_samples = beta_samples.reshape(-1, self.J_, self.p_)
        beta_entity = beta_samples[:, entity_id, :]  # (n_total_samples, p)

        if n_samples < len(beta_entity):
            rng = np.random.RandomState(self.random_seed)
            idx = rng.choice(len(beta_entity), size=n_samples, replace=False)
            beta_entity = beta_entity[idx]

        eta = X_new @ beta_entity.T  # (n_new, n_samples)

        if self.likelihood == "binary":
            preds = 1.0 / (1.0 + np.exp(-eta))
        elif self.likelihood == "regression":
            preds = eta
        else:  # count
            preds = np.exp(np.clip(eta, -30, 30))

        lower_q = 100 * (1 - credible_level) / 2
        upper_q = 100 * (1 + credible_level) / 2

        return {
            "mean": preds.mean(axis=1),
            "std": preds.std(axis=1),
            "lower": np.percentile(preds, lower_q, axis=1),
            "upper": np.percentile(preds, upper_q, axis=1),
            "samples": preds,
        }

    def save_trace(self, filepath: str, verbose: bool = True) -> None:
        """Save the MCMC trace to NetCDF (loadable with arviz.from_netcdf)."""
        if self.trace_ is None:
            raise ValueError("No trace to save. Call fit() first.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.trace_.to_netcdf(filepath)
        if verbose:
            print(f"\n✓ Trace saved to {filepath}")

    @staticmethod
    def load_trace(filepath: str, verbose: bool = True) -> az.InferenceData:
        """Load an MCMC trace from NetCDF."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Trace file not found: {filepath}")
        trace = az.from_netcdf(filepath)
        if verbose:
            print(f"✓ Trace loaded from {filepath}")
        return trace
