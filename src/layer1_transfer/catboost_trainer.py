"""
CatBoost Base Model Trainer for Transfer Learning (Layer 1)

This module implements the base model pre-training phase of the SmallML framework.
It trains a gradient boosting classifier on harmonized public datasets to learn
universal customer churn patterns that will inform Bayesian priors in Layer 2.

Key Components:
- CatBoostBaseModel: Main class for training and evaluation
- Training uses hyperparameters from Table 4.2
- Evaluation computes metrics for Table 4.3
- Feature importance extraction for Table 4.4

References:
- Section 4.2.2: Base Model Pre-training
- Algorithm 4.1: Base Model Pre-training procedure
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    roc_curve,
)


class CatBoostBaseModel:
    """
    CatBoost-based transfer learning model for customer churn prediction.

    This class implements the base model pre-training phase (Layer 1) of the
    SmallML framework. It trains on large public datasets (N~20K-100K+) to
    learn generalizable churn patterns, which are later extracted as Bayesian
    priors for SME-specific models.

    Parameters
    ----------
    iterations : int, default=1000
        Number of boosting iterations (trees to build)
    learning_rate : float, default=0.03
        Step size for gradient descent (smaller = more conservative)
    depth : int, default=6
        Maximum tree depth (controls interaction complexity)
    min_data_in_leaf : int, default=20
        Minimum samples required in each leaf node
    l2_leaf_reg : float, default=3.0
        L2 regularization coefficient for leaf weights
    subsample : float, default=0.8
        Fraction of data to sample per iteration (row sampling)
    rsm : float, default=0.8
        Fraction of features to sample per tree (column sampling)
    early_stopping_rounds : int, default=50
        Stop training if validation metric doesn't improve for N rounds
    random_seed : int, default=42
        Random seed for reproducibility
    verbose : int, default=100
        Print progress every N iterations

    Attributes
    ----------
    model_ : CatBoostClassifier
        Fitted CatBoost model (available after calling fit())
    feature_names_ : List[str]
        Feature names used during training
    training_history_ : Dict
        Training and validation metrics over iterations
    """

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.03,
        depth: int = 6,
        min_data_in_leaf: int = 20,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        rsm: float = 0.8,
        early_stopping_rounds: int = 50,
        random_seed: int = 42,
        verbose: int = 100,
    ):
        """Initialize CatBoost base model with hyperparameters from Table 4.2."""
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.min_data_in_leaf = min_data_in_leaf
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.rsm = rsm
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.verbose = verbose

        # Initialize CatBoost classifier
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            min_data_in_leaf=self.min_data_in_leaf,
            l2_leaf_reg=self.l2_leaf_reg,
            subsample=self.subsample,
            rsm=self.rsm,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=self.early_stopping_rounds,
            random_seed=self.random_seed,
            verbose=self.verbose,
        )

        self.feature_names_: Optional[List[str]] = None
        self.training_history_: Optional[Dict] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "CatBoostBaseModel":
        """
        Train CatBoost model on training data with optional validation.

        Implements Algorithm 4.1 (Base Model Pre-training) from section 4.2.2.
        Trains gradient boosting ensemble with early stopping based on
        validation performance.

        Parameters
        ----------
        X_train : pd.DataFrame, shape (n_train, p)
            Training feature matrix
        y_train : pd.Series, shape (n_train,)
            Training target (binary: 0=retained, 1=churned)
        X_val : pd.DataFrame, shape (n_val, p), optional
            Validation feature matrix for early stopping
        y_val : pd.Series, shape (n_val,), optional
            Validation target

        Returns
        -------
        self : CatBoostBaseModel
            Fitted model instance

        Raises
        ------
        ValueError
            If X_train and y_train have mismatched lengths
        """
        # Validate inputs
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train ({len(X_train)}) and y_train ({len(y_train)}) "
                f"must have the same length"
            )

        if X_val is not None and y_val is not None:
            if len(X_val) != len(y_val):
                raise ValueError(
                    f"X_val ({len(X_val)}) and y_val ({len(y_val)}) "
                    f"must have the same length"
                )

        # Store feature names
        self.feature_names_ = list(X_train.columns)

        # Prepare evaluation set (both train and validation for tracking)
        eval_set = None
        if X_val is not None and y_val is not None:
            # CatBoost expects a list of tuples (X, y, label)
            eval_set = [(X_train, y_train), (X_val, y_val)]

        print(f"\n{'='*70}")
        print(f"Training CatBoost Base Model (Algorithm 4.1)")
        print(f"{'='*70}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {len(self.feature_names_)}")
        print(f"Churn rate (train): {y_train.mean():.3f}")
        if y_val is not None:
            print(f"Validation samples: {len(X_val):,}")
            print(f"Churn rate (val): {y_val.mean():.3f}")
        print(f"{'='*70}\n")

        # Train model
        self.model_.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else False,
            verbose=self.verbose,
        )

        # Store training history
        if eval_set is not None:
            evals_result = self.model_.get_evals_result()

            # CatBoost eval_set keys: 'validation_0' for train, 'validation_1' for val
            # when eval_set is a list of tuples
            train_key = "validation_0"  # First eval set (training)
            val_key = "validation_1"    # Second eval set (validation)

            # Metric name is exactly as specified in eval_metric parameter
            metric_name = self.model_.get_params()["eval_metric"]

            self.training_history_ = {
                "train_auc": evals_result[train_key][metric_name],
                "val_auc": evals_result[val_key][metric_name],
                "best_iteration": self.model_.get_best_iteration(),
                "total_iterations": self.model_.tree_count_,
            }

        print(f"\n{'='*70}")
        print(f"Training Complete!")
        if self.training_history_:
            print(
                f"Best iteration: {self.training_history_['best_iteration']} "
                f"/ {self.training_history_['total_iterations']}"
            )
            print(
                f"Best validation AUC: "
                f"{self.training_history_['val_auc'][self.training_history_['best_iteration']]:.4f}"
            )
        print(f"{'='*70}\n")

        return self

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Validation"
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.

        Computes comprehensive metrics for Table 4.3 (Base Model Performance).

        Parameters
        ----------
        X : pd.DataFrame, shape (n, p)
            Feature matrix
        y : pd.Series, shape (n,)
            True labels
        dataset_name : str, default='Validation'
            Name of dataset being evaluated (for logging)

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary containing:
            - auc_roc: Area Under ROC Curve
            - accuracy: Classification accuracy at 0.5 threshold
            - precision: Positive predictive value
            - recall: True positive rate (sensitivity)
            - f1_score: Harmonic mean of precision and recall
            - log_loss: Cross-entropy loss (lower is better)

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if self.feature_names_ is None:
            raise ValueError("Model must be fitted before evaluation. Call fit() first.")

        print(f"\n{'='*70}")
        print(f"Evaluating on {dataset_name} Set")
        print(f"{'='*70}")

        # Get predictions
        y_pred_proba = self.model_.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Compute metrics
        metrics = {
            "auc_roc": roc_auc_score(y, y_pred_proba),
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "log_loss": log_loss(y, y_pred_proba),
        }

        # Print metrics
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"Log Loss:  {metrics['log_loss']:.4f}")
        print(f"{'='*70}\n")

        return metrics

    def get_feature_importances(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Extract feature importances from trained model.

        Returns feature importance scores for Table 4.4 (Top 10 Features).
        Importance is measured by total gain contribution across all trees.

        Parameters
        ----------
        top_k : int, optional
            If specified, return only top K most important features

        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with columns ['feature', 'importance', 'rank']
            sorted by importance in descending order

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if self.feature_names_ is None:
            raise ValueError(
                "Model must be fitted before extracting importances. Call fit() first."
            )

        # Get feature importances from CatBoost
        importances = self.model_.get_feature_importance()

        # Create DataFrame
        importance_df = pd.DataFrame(
            {"feature": self.feature_names_, "importance": importances}
        )

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)

        # Add rank
        importance_df["rank"] = range(1, len(importance_df) + 1)

        # Reset index
        importance_df = importance_df.reset_index(drop=True)

        # Return top K if specified
        if top_k is not None:
            importance_df = importance_df.head(top_k)

        return importance_df

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Saves model in CatBoost native .cbm format for efficient loading.

        Parameters
        ----------
        filepath : str
            Path where model will be saved (e.g., 'models/catboost_base.cbm')

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if self.feature_names_ is None:
            raise ValueError("Model must be fitted before saving. Call fit() first.")

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model_.save_model(filepath)
        print(f"✓ Model saved to: {filepath}")

    def save_metadata(
        self, filepath: str, training_metrics: Dict, validation_metrics: Dict
    ) -> None:
        """
        Save training metadata and metrics to JSON.

        Creates a comprehensive metadata file documenting the training run,
        including hyperparameters, dataset statistics, and performance metrics.

        Parameters
        ----------
        filepath : str
            Path where metadata JSON will be saved
        training_metrics : Dict
            Metrics computed on training set
        validation_metrics : Dict
            Metrics computed on validation set
        """
        metadata = {
            "created_at": datetime.now().isoformat(),
            "model_type": "CatBoost",
            "framework_layer": "Layer 1: Transfer Learning",
            "algorithm": "Algorithm 4.1 (Base Model Pre-training)",
            "hyperparameters": {
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "depth": self.depth,
                "min_data_in_leaf": self.min_data_in_leaf,
                "l2_leaf_reg": self.l2_leaf_reg,
                "subsample": self.subsample,
                "rsm": self.rsm,
                "early_stopping_rounds": self.early_stopping_rounds,
                "random_seed": self.random_seed,
            },
            "training_history": self.training_history_,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "feature_count": len(self.feature_names_),
            "feature_names": self.feature_names_,
        }

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "CatBoostBaseModel":
        """
        Load a trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved model file (.cbm)

        Returns
        -------
        model : CatBoostBaseModel
            Loaded model instance

        Raises
        ------
        FileNotFoundError
            If model file doesn't exist
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Create instance with default parameters
        # (actual parameters are stored in the model file)
        instance = cls()

        # Load model
        instance.model_ = CatBoostClassifier()
        instance.model_.load_model(filepath)

        # Extract feature names from model
        instance.feature_names_ = instance.model_.feature_names_

        print(f"✓ Model loaded from: {filepath}")
        print(f"  Features: {len(instance.feature_names_)}")
        print(f"  Trees: {instance.model_.tree_count_}")

        return instance
