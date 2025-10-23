"""
SME Data Generator for Hierarchical Bayesian Model

This module generates synthetic SME datasets by sampling from harmonized public data
and adding business-specific noise to simulate realistic small-data scenarios.

"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split


class SMEDataGenerator:
    """
    Generate synthetic SME datasets for hierarchical Bayesian model validation.

    This class samples customers from harmonized public datasets and adds
    SME-specific noise to create realistic business-level variation.

    Parameters
    ----------
    X : pd.DataFrame, shape (n_samples, p)
        Feature matrix from harmonized public data
    y : pd.Series, shape (n_samples,)
        Target variable (binary churn: 0=retained, 1=churned)
    random_seed : int, optional (default=42)
        Random seed for reproducibility

    Attributes
    ----------
    X_ : pd.DataFrame
        Stored feature matrix
    y_ : pd.Series
        Stored target vector
    feature_names_ : List[str]
        Names of features
    n_features_ : int
        Number of features
    sme_datasets_ : Dict[int, Dict]
        Generated SME datasets (populated after calling create_synthetic_smes)

    Examples
    --------
    >>> generator = SMEDataGenerator(X_train, y_train, random_seed=42)
    >>> sme_datasets = generator.create_synthetic_smes(J=10, n_per_sme=50)
    >>> generator.save_sme_datasets('data/sme_datasets/')
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_seed: int = 42
    ):
        """Initialize SME data generator."""
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.feature_names_ = list(X.columns)
        self.n_features_ = len(self.feature_names_)
        self.random_seed = random_seed
        self.sme_datasets_ = {}

        # Set random seed
        np.random.seed(random_seed)

        # Validate inputs
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )

        if y.nunique() != 2:
            raise ValueError(
                f"y must be binary (0/1). Got unique values: {y.unique()}"
            )

    def create_synthetic_smes(
        self,
        J: int = 10,
        n_per_sme: int = 50,
        noise_scale: float = 0.1,
        stratify: bool = True,
        verbose: bool = True
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Create synthetic SME datasets by sampling from public data.

        Parameters
        ----------
        J : int, optional (default=10)
            Number of SMEs to generate
        n_per_sme : int, optional (default=50)
            Number of customers per SME (small-data regime)
        noise_scale : float, optional (default=0.1)
            Scale of Gaussian noise to add (relative to feature std)
            noise ~ N(0, noise_scale * std(feature))
        stratify : bool, optional (default=True)
            If True, preserve churn rate in each SME sample
        verbose : bool, optional (default=True)
            If True, print progress messages

        Returns
        -------
        sme_datasets : Dict[int, Dict[str, pd.DataFrame]]
            Dictionary mapping SME index j to {'X': features, 'y': target}

        Notes
        -----
        Sampling strategy:
        1. Sample n_per_sme customers with replacement (allows variation)
        2. Add Gaussian noise: X_noise = X + N(0, noise_scale * std(X))
        3. Stratify by churn if requested (preserves ~21% churn rate)
        4. Reset indices for clean DataFrames

        The noise addition simulates business-specific characteristics that
        cause SMEs to deviate from population patterns.
        """
        if verbose:
            print(f"\nGenerating {J} synthetic SME datasets...")
            print(f"  Customers per SME: {n_per_sme}")
            print(f"  Noise scale: {noise_scale}")
            print(f"  Stratified sampling: {stratify}")

        # Validate parameters
        if J < 2:
            raise ValueError(f"J must be >= 2. Got: {J}")
        if n_per_sme < 10:
            raise ValueError(f"n_per_sme must be >= 10. Got: {n_per_sme}")
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0. Got: {noise_scale}")

        # Precompute feature standard deviations for noise
        feature_stds = self.X_.std()

        # Generate datasets for each SME
        self.sme_datasets_ = {}
        churn_rates = []

        for j in range(J):
            if verbose and (j + 1) % 5 == 0:
                print(f"  Generated SME {j + 1}/{J}...")

            # Sample customers (with replacement to create variation)
            if stratify:
                # Stratified sampling preserves churn rate
                indices = self._stratified_sample(
                    self.y_.values,
                    n_samples=n_per_sme,
                    random_state=self.random_seed + j
                )
            else:
                # Simple random sampling
                indices = np.random.choice(
                    len(self.X_),
                    size=n_per_sme,
                    replace=True
                )

            # Extract sampled data
            X_sme = self.X_.iloc[indices].copy()
            y_sme = self.y_.iloc[indices].copy()

            # Add SME-specific noise to features
            if noise_scale > 0:
                noise = np.random.normal(
                    loc=0,
                    scale=noise_scale * feature_stds.values,
                    size=X_sme.shape
                )
                X_sme = X_sme + noise

            # Reset indices for clean DataFrames
            X_sme = X_sme.reset_index(drop=True)
            y_sme = y_sme.reset_index(drop=True)

            # Store
            self.sme_datasets_[j] = {
                'X': X_sme,
                'y': y_sme
            }

            # Track churn rate
            churn_rates.append(y_sme.mean())

        if verbose:
            print(f"\n✓ Generated {J} SME datasets")
            print(f"  Total customers: {J * n_per_sme}")
            print(f"  Mean churn rate: {np.mean(churn_rates):.3f} " +
                  f"(std: {np.std(churn_rates):.3f})")
            print(f"  Churn rate range: [{np.min(churn_rates):.3f}, " +
                  f"{np.max(churn_rates):.3f}]")

        return self.sme_datasets_

    def _stratified_sample(
        self,
        y: np.ndarray,
        n_samples: int,
        random_state: int
    ) -> np.ndarray:
        """
        Perform stratified sampling to preserve class distribution.

        Parameters
        ----------
        y : np.ndarray, shape (n_total,)
            Binary target array
        n_samples : int
            Number of samples to draw
        random_state : int
            Random seed

        Returns
        -------
        indices : np.ndarray, shape (n_samples,)
            Indices of sampled observations
        """
        # Separate churned and retained indices
        churned_idx = np.where(y == 1)[0]
        retained_idx = np.where(y == 0)[0]

        # Calculate number to sample from each class
        churn_rate = y.mean()
        n_churned = int(np.round(n_samples * churn_rate))
        n_retained = n_samples - n_churned

        # Sample with replacement from each class
        np.random.seed(random_state)
        sampled_churned = np.random.choice(
            churned_idx,
            size=n_churned,
            replace=True
        )
        sampled_retained = np.random.choice(
            retained_idx,
            size=n_retained,
            replace=True
        )

        # Combine and shuffle
        indices = np.concatenate([sampled_churned, sampled_retained])
        np.random.shuffle(indices)

        return indices

    def save_sme_datasets(
        self,
        output_dir: str,
        verbose: bool = True
    ) -> None:
        """
        Save SME datasets to CSV files.

        Parameters
        ----------
        output_dir : str
            Directory to save datasets (will be created if doesn't exist)
        verbose : bool, optional (default=True)
            If True, print progress messages

        Notes
        -----
        Creates one CSV file per SME:
        - output_dir/sme_0.csv (X and y combined)
        - output_dir/sme_1.csv
        - ...
        - output_dir/metadata.json (dataset information)

        The 'churned' column contains the target variable.
        """
        if not self.sme_datasets_:
            raise ValueError(
                "No SME datasets to save. Call create_synthetic_smes() first."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nSaving SME datasets to {output_dir}...")

        # Save each SME's data
        J = len(self.sme_datasets_)
        for j in range(J):
            # Combine X and y
            sme_data = self.sme_datasets_[j]['X'].copy()
            sme_data['churned'] = self.sme_datasets_[j]['y'].values

            # Save to CSV
            filepath = output_path / f"sme_{j}.csv"
            sme_data.to_csv(filepath, index=False)

            if verbose and (j + 1) % 5 == 0:
                print(f"  Saved SME {j + 1}/{J}...")

        # Save metadata
        metadata = {
            'n_smes': J,
            'n_per_sme': len(self.sme_datasets_[0]['X']),
            'n_features': self.n_features_,
            'feature_names': self.feature_names_,
            'random_seed': self.random_seed,
            'churn_rates': {
                j: float(self.sme_datasets_[j]['y'].mean())
                for j in range(J)
            },
            'mean_churn_rate': float(np.mean([
                self.sme_datasets_[j]['y'].mean() for j in range(J)
            ])),
            'generation_date': pd.Timestamp.now().isoformat()
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            print(f"\n✓ Saved {J} SME datasets")
            print(f"  Directory: {output_dir}")
            print(f"  Files: sme_0.csv through sme_{J-1}.csv")
            print(f"  Metadata: metadata.json")

    @staticmethod
    def load_sme_datasets(
        input_dir: str,
        verbose: bool = True
    ) -> Tuple[Dict[int, Dict[str, pd.DataFrame]], Dict]:
        """
        Load SME datasets from CSV files.

        Parameters
        ----------
        input_dir : str
            Directory containing SME dataset CSV files
        verbose : bool, optional (default=True)
            If True, print progress messages

        Returns
        -------
        sme_datasets : Dict[int, Dict[str, pd.DataFrame]]
            Dictionary mapping SME index j to {'X': features, 'y': target}
        metadata : Dict
            Metadata about the datasets

        Raises
        ------
        FileNotFoundError
            If metadata.json not found in input_dir
        """
        input_path = Path(input_dir)

        # Load metadata first
        metadata_path = input_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Make sure you're loading from the correct directory."
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        J = metadata['n_smes']

        if verbose:
            print(f"\nLoading {J} SME datasets from {input_dir}...")

        # Load each SME's data
        sme_datasets = {}
        for j in range(J):
            filepath = input_path / f"sme_{j}.csv"

            if not filepath.exists():
                raise FileNotFoundError(
                    f"SME dataset file not found: {filepath}"
                )

            # Load CSV
            sme_data = pd.read_csv(filepath)

            # Split into X and y
            y_sme = sme_data['churned']
            X_sme = sme_data.drop('churned', axis=1)

            sme_datasets[j] = {
                'X': X_sme,
                'y': y_sme
            }

            if verbose and (j + 1) % 5 == 0:
                print(f"  Loaded SME {j + 1}/{J}...")

        if verbose:
            print(f"\n✓ Loaded {J} SME datasets")
            print(f"  Total customers: {J * metadata['n_per_sme']}")
            print(f"  Features: {metadata['n_features']}")
            print(f"  Mean churn rate: {metadata['mean_churn_rate']:.3f}")

        return sme_datasets, metadata

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all SME datasets.

        Returns
        -------
        summary : pd.DataFrame
            Summary statistics with columns:
            - sme_id: SME identifier
            - n_customers: Number of customers
            - churn_rate: Proportion of churned customers
            - mean_X: Mean feature values (one column per feature)

        Raises
        ------
        ValueError
            If no SME datasets have been generated yet
        """
        if not self.sme_datasets_:
            raise ValueError(
                "No SME datasets to summarize. Call create_synthetic_smes() first."
            )

        J = len(self.sme_datasets_)

        summary_data = []
        for j in range(J):
            X_sme = self.sme_datasets_[j]['X']
            y_sme = self.sme_datasets_[j]['y']

            row = {
                'sme_id': j,
                'n_customers': len(X_sme),
                'churn_rate': y_sme.mean(),
            }

            # Add mean of each feature
            for feat in self.feature_names_:
                row[f'mean_{feat}'] = X_sme[feat].mean()

            summary_data.append(row)

        summary = pd.DataFrame(summary_data)
        return summary

    def split_train_calibration(
        self,
        calibration_fraction: float = 0.25,
        random_state: Optional[int] = None
    ) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """
        Split each SME's data into training and calibration sets.

        This split is used for conformal prediction calibration.

        Parameters
        ----------
        calibration_fraction : float, optional (default=0.25)
            Fraction of data to reserve for calibration
        random_state : int, optional (default=None)
            Random seed for reproducibility. If None, uses self.random_seed

        Returns
        -------
        train_datasets : Dict[int, Dict[str, pd.DataFrame]]
            Training datasets for each SME
        cal_datasets : Dict[int, Dict[str, pd.DataFrame]]
            Calibration datasets for each SME

        Examples
        --------
        >>> train_data, cal_data = generator.split_train_calibration(
        ...     calibration_fraction=0.25
        ... )
        >>> # SME 0: 50 customers → 37 train, 13 calibration
        """
        if not self.sme_datasets_:
            raise ValueError(
                "No SME datasets to split. Call create_synthetic_smes() first."
            )

        if random_state is None:
            random_state = self.random_seed

        J = len(self.sme_datasets_)
        train_datasets = {}
        cal_datasets = {}

        for j in range(J):
            X_sme = self.sme_datasets_[j]['X']
            y_sme = self.sme_datasets_[j]['y']

            # Split with stratification to preserve churn rate
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_sme,
                y_sme,
                test_size=calibration_fraction,
                stratify=y_sme,
                random_state=random_state + j
            )

            train_datasets[j] = {'X': X_train, 'y': y_train}
            cal_datasets[j] = {'X': X_cal, 'y': y_cal}

        return train_datasets, cal_datasets
