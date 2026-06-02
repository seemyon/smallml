"""SmallML Pipeline - Domain-Agnostic User Interface"""

import pickle
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from .layer1 import PriorExtractor
from .layer2.hierarchical_model import HierarchicalBayesianModel
from .layer3.conformal_predictor import ConformalPredictor
from .layer3.cqr import ConformalQuantileRegressor
from .layer3.prediction_sets import (
    confidence_flag_from_set,
    confidence_flag_from_width,
)
from .preprocessing import FeaturePreprocessor, align_features
from .feature_matcher import FeatureMatcher

VALID_TASKS = ("binary", "regression", "count")

# Standard output contract (identical for every task / domain).
CONTRACT_COLUMNS = [
    "point_prediction",
    "posterior_distribution",
    "credible_lower",
    "credible_upper",
    "conformal_set",
    "confidence_flag",
]

# Minimum-data thresholds (see requirements §6).
MIN_REFERENCE_RATIO = 5  # reference must be >= 5x the combined target size
MIN_ENTITIES = 5  # absolute minimum number of entity groups
RECOMMENDED_ENTITIES = 10  # warn below this
MIN_OBS_PER_ENTITY = 20  # below this, even pooling cannot compensate


class Pipeline:
    """
    End-to-end SmallML pipeline for small-data predictive analytics.

    Domain-agnostic Bayesian transfer learning: bring a large *reference*
    dataset, a small *target* dataset with labeled entities, and a task type;
    receive a standardized five-field output contract regardless of domain.

    Parameters
    ----------
    task : {'binary', 'regression', 'count'}, default='binary'
        Prediction task / likelihood:
        - 'binary'     : logistic likelihood, probability output, conformal sets.
        - 'regression' : Gaussian likelihood, value output, conformal intervals.
        - 'count'      : Poisson likelihood, expected-count output, intervals.
    confidence_level : float, default=0.90
        Central mass for Bayesian credible intervals and the conformal target
        coverage (conformal miscoverage alpha = 1 - confidence_level).
    quick_mode : bool, default=False
        Faster MCMC (2 chains / 500 draws) for prototyping.
    random_seed : int, default=42
    lambda_scale : float, default=1.0
        Prior-variance scaling used when extracting priors from a reference
        dataset (see :class:`smallml.PriorExtractor`).
    allow_single_entity : bool, default=False
        Bypass the ``J >= 5`` minimum (single-/few-entity mode). Relies heavily
        on transfer-learning priors; accuracy is typically lower without pooling.
    preprocess : bool, default=True
        Standardize numeric features during preprocessing (categorical encoding
        and imputation always apply).
    cqr_min_train : int, default=150
        For regression/count, the minimum pooled training rows required to fit
        Conformalized Quantile Regression (CQR). Below this the pipeline falls
        back to locally-adaptive normalized split conformal.

    Examples
    --------
    >>> from smallml import Pipeline
    >>> # Multi-entity binary task with a user-supplied reference dataset
    >>> pipe = Pipeline(task='binary')
    >>> pipe.fit(target_df, target_col='outcome', entity_col='group',
    ...          reference_data=reference_df)
    >>> preds = pipe.predict(new_rows, entity_id='group_A')
    >>> preds[['point_prediction', 'conformal_set', 'confidence_flag']]
    """

    def __init__(
        self,
        task: str = "binary",
        confidence_level: float = 0.90,
        quick_mode: bool = False,
        random_seed: int = 42,
        lambda_scale: float = 1.0,
        allow_single_entity: bool = False,
        preprocess: bool = True,
        cqr_min_train: int = 150,
    ):
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")

        self.task = task
        self.confidence_level = confidence_level
        self.quick_mode = quick_mode
        self.random_seed = random_seed
        self.lambda_scale = lambda_scale
        self.allow_single_entity = allow_single_entity
        self.preprocess = preprocess
        self.cqr_min_train = cqr_min_train

        # Set during fit()
        self.hierarchical_model = None
        self.conformal_predictor = None  # binary sets / normalized-split fallback
        self.conformal_regressor = None  # CQR (regression/count default)
        self.conformal_method_: Optional[str] = None
        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_input_cols_: Optional[List[str]] = None
        self.target_col: Optional[str] = None
        self.entity_names: Optional[List[str]] = None
        self.single_entity_mode = False
        self.width_threshold_: Optional[float] = None
        self._priors: Optional[Dict] = None

    # ================================================================== #
    # fit
    # ================================================================== #

    def fit(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        target_col: str,
        entity_col: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        calibration_fraction: float = 0.25,
        validate_convergence: bool = True,
    ) -> "Pipeline":
        """
        Fit the pipeline on entity-grouped target data.

        Parameters
        ----------
        data : pd.DataFrame or dict of {str: pd.DataFrame}
            Either a single DataFrame (use ``entity_col`` to identify entities)
            or a dict mapping entity name -> DataFrame. Each row is one
            observation; each frame holds features plus ``target_col``.
        target_col : str
            Name of the target column (required, no default).
        entity_col : str, optional
            Column identifying the entity each observation belongs to. Required
            when ``data`` is a single DataFrame with more than one entity.
        reference_data : pd.DataFrame, optional
            Large reference dataset for Layer-1 prior extraction (must contain
            ``target_col``). If omitted (and ``priors`` is not given), weakly
            informative priors are used.
        priors : dict, optional
            Precomputed priors ``{'beta_0', 'Sigma_0', 'feature_names'}`` to use
            instead of extracting from ``reference_data``.
        calibration_fraction : float, default=0.25
            Fraction of each entity's data reserved for conformal calibration.
        validate_convergence : bool, default=True
            Raise if MCMC fails the R̂ < 1.01 criterion.

        Returns
        -------
        self
        """
        self.target_col = target_col

        # --- Normalize input into {entity_name: DataFrame(features + target)} ---
        data_dict = self._normalize_input(data, target_col, entity_col)
        self.entity_names = list(data_dict.keys())
        J = len(self.entity_names)
        self.single_entity_mode = J == 1

        first_df = next(iter(data_dict.values()))
        raw_features = [c for c in first_df.columns if c != target_col]
        total_target_rows = sum(len(df) for df in data_dict.values())

        # --- Feature alignment (intersection with reference) -----------------
        if reference_data is not None:
            ref_features = [c for c in reference_data.columns if c != target_col]
            feature_input_cols = align_features(ref_features, raw_features)
        else:
            feature_input_cols = raw_features
        self.feature_input_cols_ = feature_input_cols

        # --- Enforce minimum-data requirements -------------------------------
        self._enforce_minimums(reference_data, total_target_rows, data_dict, J)

        print(f"\n{'=' * 70}")
        print(f"SmallML Pipeline ({self.task}): {J} entity group(s)")
        print(f"{'=' * 70}\n")

        # --- Preprocessing ---------------------------------------------------
        self.preprocessor = FeaturePreprocessor(scale=self.preprocess)
        if reference_data is not None:
            self.preprocessor.fit(reference_data, feature_names=feature_input_cols)
        else:
            all_X = pd.concat(
                [df[feature_input_cols] for df in data_dict.values()],
                ignore_index=True,
            )
            self.preprocessor.fit(all_X, feature_names=feature_input_cols)
        self.feature_names = self.preprocessor.get_feature_names_out()

        # --- Priors ----------------------------------------------------------
        self._priors = self._resolve_priors(priors, reference_data, target_col)

        # --- Transform entities + split into train / calibration -------------
        train_data, cal_data = self._build_entity_datasets(
            data_dict, target_col, calibration_fraction
        )

        # --- Layer 2: hierarchical Bayesian inference ------------------------
        print("[Layer 2] Fitting hierarchical Bayesian model...")
        self._fit_hierarchical(train_data)
        if validate_convergence:
            self._validate_convergence()

        # --- Layer 3: conformal calibration ----------------------------------
        print("\n[Layer 3] Calibrating conformal predictor...")
        self._fit_conformal(train_data, cal_data)

        print(f"\n{'=' * 70}")
        print("✓ SmallML Pipeline fitted successfully!")
        print(f"{'=' * 70}\n")
        return self

    # ================================================================== #
    # predict
    # ================================================================== #

    def predict(
        self,
        X: pd.DataFrame,
        entity_id: Optional[str] = None,
        n_posterior_samples: int = 1000,
    ) -> pd.DataFrame:
        """
        Predict with the standard five-field output contract.

        Returns
        -------
        pd.DataFrame with columns:
            - ``point_prediction``      : most likely outcome (prob / value / count)
            - ``posterior_distribution``: per-row array of posterior samples
            - ``credible_lower``/``credible_upper`` : Bayesian credible interval
            - ``conformal_set``         : set ``"{0}"/"{1}"/"{0,1}"`` (binary) or
                                          interval ``"[lo, hi]"`` (regression/count)
            - ``confidence_flag``       : ``"HIGH"`` or ``"UNCERTAIN"``
        """
        if self.hierarchical_model is None:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")

        entity_idx = self._resolve_entity_idx(entity_id)

        X_t = self.preprocessor.transform(X)
        post = self.hierarchical_model.posterior_predictive(
            X_t.values,
            entity_id=entity_idx,
            n_samples=n_posterior_samples,
            credible_level=self.confidence_level,
        )

        if self.task == "binary":
            point = np.clip(post["mean"], 0.0, 1.0)
            lower = np.clip(post["lower"], 0.0, 1.0)
            upper = np.clip(post["upper"], 0.0, 1.0)
        else:
            point = post["mean"]
            lower = post["lower"]
            upper = post["upper"]

        out = pd.DataFrame(index=X.index)
        out["point_prediction"] = point
        out["posterior_distribution"] = list(post["samples"])
        out["credible_lower"] = lower
        out["credible_upper"] = upper

        if self.task == "binary":
            sets = self.conformal_predictor.predict_set(point, return_sets=True)
            out["conformal_set"] = [self._format_set(s) for s in sets]
            out["confidence_flag"] = [confidence_flag_from_set(s) for s in sets]
        else:
            intervals = self._regression_intervals(X_t, point, post["std"])
            out["conformal_set"] = [f"[{lo:.4g}, {hi:.4g}]" for lo, hi in intervals]
            widths = intervals[:, 1] - intervals[:, 0]
            out["confidence_flag"] = [
                confidence_flag_from_width(w, self.width_threshold_) for w in widths
            ]

        return out[CONTRACT_COLUMNS]

    # ================================================================== #
    # evaluate / diagnostics / persistence
    # ================================================================== #

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        entity_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate on labeled test data with task-appropriate metrics."""
        if self.hierarchical_model is None:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")
        y_true = np.asarray(y_test)

        if self.task == "binary":
            preds = self.predict(X_test, entity_id=entity_id)
            point = preds["point_prediction"].values
            y_pred = (point > 0.5).astype(int)
            cov = self.conformal_predictor.validate_coverage(
                y_true, point, verbose=False
            )
            return {
                "auc": roc_auc_score(y_true, point),
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "conformal_coverage": cov["coverage"],
                "mean_set_size": cov["avg_set_size"],
            }

        # regression / count
        entity_idx = self._resolve_entity_idx(entity_id)
        X_t = self.preprocessor.transform(X_test)
        post = self.hierarchical_model.posterior_predictive(
            X_t.values,
            entity_id=entity_idx,
            n_samples=1000,
            credible_level=self.confidence_level,
        )
        point = post["mean"]
        intervals = self._regression_intervals(X_t, point, post["std"])
        inside = (y_true >= intervals[:, 0]) & (y_true <= intervals[:, 1])
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, point))),
            "mae": float(mean_absolute_error(y_true, point)),
            "conformal_coverage": float(np.mean(inside)),
            "avg_interval_width": float(np.mean(intervals[:, 1] - intervals[:, 0])),
            "conformal_method": self.conformal_method_,
        }

    def get_convergence_diagnostics(self) -> pd.DataFrame:
        """Return per-parameter MCMC diagnostics (r_hat, ess_bulk, ess_tail)."""
        if self.hierarchical_model is None:
            raise RuntimeError("Model not fitted yet.")
        import arviz as az

        summary = az.summary(self.hierarchical_model.trace_).reset_index()
        summary = summary.rename(columns={"index": "parameter"})
        return summary[["parameter", "r_hat", "ess_bulk", "ess_tail"]]

    def save(self, filepath: str):
        """Save the fitted pipeline (drops the unpicklable PyMC model object)."""
        model_backup = None
        if self.hierarchical_model is not None:
            model_backup = self.hierarchical_model.model_
            self.hierarchical_model.model_ = None
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            print(f"✓ Pipeline saved to {filepath}")
        finally:
            if model_backup is not None:
                self.hierarchical_model.model_ = model_backup

    @classmethod
    def load(cls, filepath: str) -> "Pipeline":
        """Load a fitted pipeline from disk."""
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        print(f"✓ Pipeline loaded from {filepath}")
        return pipeline

    # ================================================================== #
    # internal helpers
    # ================================================================== #

    def _normalize_input(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        target_col: str,
        entity_col: Optional[str],
    ) -> Dict[str, pd.DataFrame]:
        """Coerce supported input shapes into {entity_name: DataFrame}."""
        if isinstance(data, pd.DataFrame):
            if entity_col is not None:
                if entity_col not in data.columns:
                    raise ValueError(
                        f"entity_col '{entity_col}' not found in data columns"
                    )
                data_dict = {
                    str(name): grp.drop(columns=[entity_col]).reset_index(drop=True)
                    for name, grp in data.groupby(entity_col)
                }
            else:
                data_dict = {"entity_1": data.reset_index(drop=True)}
        elif isinstance(data, dict):
            if len(data) == 0:
                raise ValueError("data dict is empty")
            data_dict = {str(k): v.reset_index(drop=True) for k, v in data.items()}
        else:
            raise TypeError(
                "data must be a DataFrame or a dict of {entity_name: DataFrame}"
            )

        for name, df in data_dict.items():
            if target_col not in df.columns:
                raise ValueError(
                    f"Entity '{name}' is missing target column '{target_col}'"
                )
        return data_dict

    def _enforce_minimums(
        self,
        reference_data: Optional[pd.DataFrame],
        total_target_rows: int,
        data_dict: Dict[str, pd.DataFrame],
        J: int,
    ):
        """Enforce the minimum-data requirements (requirements §6)."""
        if reference_data is not None:
            if len(reference_data) < MIN_REFERENCE_RATIO * total_target_rows:
                raise ValueError(
                    f"Reference dataset too small for transfer learning: "
                    f"{len(reference_data)} rows < {MIN_REFERENCE_RATIO}x the "
                    f"combined target size ({total_target_rows} rows). Provide a "
                    f"reference dataset at least {MIN_REFERENCE_RATIO}x larger."
                )

        if J < MIN_ENTITIES:
            if self.allow_single_entity:
                warnings.warn(
                    f"Only {J} entity group(s) provided (minimum {MIN_ENTITIES}). "
                    "Proceeding because allow_single_entity=True; hierarchical "
                    "pooling benefits are reduced or absent."
                )
            else:
                raise ValueError(
                    f"Only {J} entity group(s) provided; hierarchical inference "
                    f"requires at least {MIN_ENTITIES}. Provide more entities, or "
                    "set Pipeline(allow_single_entity=True) to override."
                )
        elif J < RECOMMENDED_ENTITIES:
            warnings.warn(
                f"{J} entity groups provided; {RECOMMENDED_ENTITIES}+ is "
                "recommended for stable hierarchical pooling."
            )

        for name, df in data_dict.items():
            if len(df) < MIN_OBS_PER_ENTITY:
                raise ValueError(
                    f"Entity '{name}' has only {len(df)} observations; a minimum "
                    f"of {MIN_OBS_PER_ENTITY} per entity is required. Provide more "
                    "data for this entity or remove it."
                )

    def _resolve_priors(
        self,
        priors: Optional[Dict],
        reference_data: Optional[pd.DataFrame],
        target_col: str,
    ) -> Optional[Dict]:
        """Choose priors: explicit dict, Layer-1 extraction, or weak (None)."""
        if priors is not None:
            return priors
        if reference_data is not None:
            ref_X = self.preprocessor.transform(reference_data)
            ref_df = ref_X.copy()
            ref_df[target_col] = reference_data[target_col].values
            extractor = PriorExtractor(
                task=self.task,
                lambda_scale=self.lambda_scale,
                random_seed=self.random_seed,
            )
            extractor.fit(ref_df, target_col, feature_names=self.feature_names)
            return extractor.get_priors()
        return None

    def _build_entity_datasets(self, data_dict, target_col, cal_frac):
        """Transform features and split each entity into train / calibration."""
        train_data, cal_data = {}, {}
        for i, (name, df) in enumerate(data_dict.items()):
            X_t = self.preprocessor.transform(df[self.feature_input_cols_])
            X_t = X_t.reset_index(drop=True)
            y = df[target_col].reset_index(drop=True)

            n_cal = max(1, int(len(df) * cal_frac))
            shuffled = np.random.RandomState(self.random_seed).permutation(len(df))
            cal_idx = shuffled[:n_cal]
            train_idx = shuffled[n_cal:]

            train_data[i] = {
                "X": X_t.iloc[train_idx].reset_index(drop=True),
                "y": y.iloc[train_idx].reset_index(drop=True),
            }
            cal_data[i] = {
                "X": X_t.iloc[cal_idx].reset_index(drop=True),
                "y": y.iloc[cal_idx].reset_index(drop=True),
            }

        n_train = sum(len(d["X"]) for d in train_data.values())
        n_cal = sum(len(d["X"]) for d in cal_data.values())
        print(
            f"✓ Split data: {len(train_data)} entities, "
            f"{n_train} train / {n_cal} calibration observations"
        )
        return train_data, cal_data

    def _hybrid_priors(self, feature_names: List[str]):
        """Build (beta_0, Sigma_0, tau) aligned to ``feature_names``."""
        N = len(feature_names)
        tau = 2.0

        if self._priors is None or "feature_names" not in self._priors:
            return np.zeros(N), np.eye(N) * 25.0, tau

        pretrained = self._priors["feature_names"]
        matcher = FeatureMatcher(pretrained)
        matches, _ = matcher.match_all(feature_names)
        name_to_idx = {f.lower(): i for i, f in enumerate(pretrained)}

        beta_0 = np.zeros(N)
        Sigma_0 = np.eye(N) * 25.0
        for i, feat in enumerate(feature_names):
            matched = matches.get(feat)
            if matched is not None and matched in name_to_idx:
                j = name_to_idx[matched]
                beta_0[i] = self._priors["beta_0"][j]
                Sigma_0[i, i] = self._priors["Sigma_0"][j, j]
        return beta_0, Sigma_0, tau

    def _fit_hierarchical(self, train_data):
        beta_0, Sigma_0, tau = self._hybrid_priors(self.feature_names)

        n_chains = 2 if self.quick_mode else 4
        n_draws = 500 if self.quick_mode else 2000
        n_tune = 500 if self.quick_mode else 1000

        self.hierarchical_model = HierarchicalBayesianModel(
            beta_0=beta_0,
            Sigma_0=Sigma_0,
            tau=tau,
            likelihood=self.task,
            random_seed=self.random_seed,
        )
        self.hierarchical_model.fit(
            train_data, chains=n_chains, draws=n_draws, tune=n_tune
        )
        print("  ✓ Hierarchical model fitted")

    def _validate_convergence(self):
        diag = self.hierarchical_model.check_convergence(verbose=False)
        if not diag["rhat_ok"]:
            raise RuntimeError(
                f"MCMC convergence failed (max R̂ = {diag['rhat_max']:.4f} ≥ 1.01).\n"
                "Try Pipeline(quick_mode=False) for more draws, or provide more data."
            )
        if not diag["ess_ok"]:
            warnings.warn(
                f"Low effective sample size (min ESS = {diag['ess_min']:.0f}). "
                "Consider increasing MCMC draws."
            )
        print(
            f"  ✓ Convergence validated (max R̂ = {diag['rhat_max']:.4f}, "
            f"min ESS = {diag['ess_min']:.0f})"
        )

    def _fit_conformal(self, train_data, cal_data):
        alpha = 1.0 - self.confidence_level

        # --- Binary: classification prediction sets (unchanged) --------------
        if self.task == "binary":
            cal_preds, cal_labels = [], []
            for i, d in cal_data.items():
                post = self.hierarchical_model.posterior_predictive(
                    d["X"].values,
                    entity_id=i,
                    n_samples=1000,
                    credible_level=self.confidence_level,
                )
                cal_preds.extend(post["mean"])
                cal_labels.extend(d["y"].values)
            cal_preds = np.clip(np.asarray(cal_preds), 0.0, 1.0)
            cal_labels = np.asarray(cal_labels)
            self.conformal_predictor = ConformalPredictor(
                alpha=alpha, task="binary", random_seed=self.random_seed
            )
            q_hat = self.conformal_predictor.calibrate(
                cal_labels, cal_preds, verbose=False
            )
            self.conformal_method_ = "binary_set"
            print(f"  ✓ Conformal calibrated (sets, q̂ = {q_hat:.4f})")
            return

        # --- Regression / count: CQR by default, normalized split fallback ----
        # Pool calibration Bayesian predictions + posterior std (fallback scale).
        cal_preds, cal_labels, cal_std = [], [], []
        for i, d in cal_data.items():
            post = self.hierarchical_model.posterior_predictive(
                d["X"].values,
                entity_id=i,
                n_samples=1000,
                credible_level=self.confidence_level,
            )
            cal_preds.extend(post["mean"])
            cal_std.extend(post["std"])
            cal_labels.extend(d["y"].values)
        cal_preds = np.asarray(cal_preds)
        cal_std = np.asarray(cal_std)
        cal_labels = np.asarray(cal_labels)

        X_cal = np.vstack([d["X"].values for d in cal_data.values()])
        y_cal = np.concatenate([d["y"].values for d in cal_data.values()])
        X_train = np.vstack([d["X"].values for d in train_data.values()])
        y_train = np.concatenate([d["y"].values for d in train_data.values()])

        n_train, n_cal = len(X_train), len(X_cal)
        min_cal = max(10, int(np.ceil(1.0 / alpha)))

        if n_train >= self.cqr_min_train and n_cal >= min_cal:
            try:
                cqr = ConformalQuantileRegressor(
                    alpha=alpha, task=self.task, random_seed=self.random_seed
                )
                cqr.fit(X_train, y_train, X_cal, y_cal)
                self.conformal_regressor = cqr
                self.conformal_predictor = None
                self.conformal_method_ = "cqr"
                iv = cqr.predict_interval(X_cal)
                self.width_threshold_ = float(np.median(iv[:, 1] - iv[:, 0]))
                print(f"  ✓ Conformal calibrated (CQR, q̂ = {cqr.q_hat_:.4f})")
                return
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(
                    f"CQR fitting failed ({exc}); falling back to normalized "
                    "split conformal."
                )

        # Fallback: locally-adaptive normalized split conformal (posterior std).
        self.conformal_predictor = ConformalPredictor(
            alpha=alpha, task=self.task, random_seed=self.random_seed
        )
        q_hat = self.conformal_predictor.calibrate(
            cal_labels, cal_preds, sigma_cal=cal_std, verbose=False
        )
        self.conformal_regressor = None
        self.conformal_method_ = "split_normalized"
        iv = self.conformal_predictor.predict_interval(cal_preds, sigma=cal_std)
        self.width_threshold_ = float(np.median(iv[:, 1] - iv[:, 0]))
        reason = (
            "too few training rows"
            if n_train < self.cqr_min_train
            else "calibration set too small"
        )
        print(
            f"  ✓ Conformal calibrated (normalized split, fallback: {reason}; "
            f"q̂ = {q_hat:.4f})"
        )

    def _regression_intervals(self, X_t, point, std) -> np.ndarray:
        """Conformal intervals for regression/count via the active method."""
        if self.conformal_method_ == "cqr":
            return self.conformal_regressor.predict_interval(X_t.values)
        return self.conformal_predictor.predict_interval(point, sigma=std)

    def _resolve_entity_idx(self, entity_id: Optional[str]) -> int:
        """Resolve an entity name to its index (defaults to the first entity)."""
        if entity_id is None:
            entity_id = self.entity_names[0]
            if not self.single_entity_mode:
                warnings.warn(f"No entity_id specified. Using '{entity_id}'.")
        if entity_id not in self.entity_names:
            raise ValueError(
                f"Unknown entity_id '{entity_id}'. "
                f"Known entities: {self.entity_names}"
            )
        return self.entity_names.index(entity_id)

    @staticmethod
    def _format_set(s: List[int]) -> str:
        if len(s) == 0:
            return "{}"
        return "{" + ", ".join(str(v) for v in sorted(s)) + "}"
