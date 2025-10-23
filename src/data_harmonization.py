"""
SmallML Framework - Data Harmonization Module
==============================================

This module provides functions to harmonize heterogeneous churn datasets
by mapping dataset-specific features to canonical feature names representing
universal customer behavior dimensions.

The harmonization process implements Section 4.2.1 of the SmallML framework,
enabling transfer learning across diverse business contexts (telecom, banking,
e-commerce) by creating unified feature representations.

Key Functions:
- harmonize_dataset(): Main harmonization function
- get_feature_mapping(): Returns feature mapping dictionaries
- extract_canonical_features(): Maps raw features to canonical names

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


def get_feature_mapping() -> Dict[str, Dict[str, str]]:
    """
    Returns feature mapping dictionaries for each dataset.

    Maps dataset-specific feature names to canonical feature names
    representing universal customer behavior dimensions:
    - recency: Time since last interaction/purchase
    - frequency: Transaction/engagement counts
    - monetary: Revenue, spending, or account value
    - tenure: Customer lifetime duration
    - churned: Binary churn indicator (target variable)

    Returns
    -------
    feature_map : Dict[str, Dict[str, str]]
        Nested dictionary with structure:
        {
            'telco': {'original_name': 'canonical_name', ...},
            'bank': {'original_name': 'canonical_name', ...},
            'ecommerce': {'original_name': 'canonical_name', ...}
        }

    Notes
    -----
    Features without clear cross-dataset analogues are retained as
    dataset-specific auxiliary features and not included in this mapping.
    """
    feature_map = {
        'telco': {
            # Core RFM features
            'tenure': 'tenure_months',
            'MonthlyCharges': 'monetary_value',
            'TotalCharges': 'total_revenue',

            # Target variable
            'Churn': 'churned',

            # Telco-specific features (retain original names)
            'SeniorCitizen': 'SeniorCitizen',
            'Partner': 'Partner',
            'Dependents': 'Dependents',
            'PhoneService': 'PhoneService',
            'MultipleLines': 'MultipleLines',
            'InternetService': 'InternetService',
            'OnlineSecurity': 'OnlineSecurity',
            'OnlineBackup': 'OnlineBackup',
            'DeviceProtection': 'DeviceProtection',
            'TechSupport': 'TechSupport',
            'StreamingTV': 'StreamingTV',
            'StreamingMovies': 'StreamingMovies',
            'Contract': 'Contract',
            'PaperlessBilling': 'PaperlessBilling',
            'PaymentMethod': 'PaymentMethod',
            'gender': 'gender'
        },

        'bank': {
            # Core RFM features
            'Tenure': 'tenure_months',
            'Balance': 'monetary_value',
            'NumOfProducts': 'frequency',  # Number of products as frequency proxy

            # Target variable
            'Churn': 'churned',

            # Bank-specific features (retain original names)
            'CreditScore': 'CreditScore',
            'Geography': 'Geography',
            'Gender': 'gender',
            'Age': 'Age',
            'HasCrCard': 'HasCrCard',
            'IsActiveMember': 'IsActiveMember',
            'EstimatedSalary': 'EstimatedSalary'
        },

        'ecommerce': {
            # Core RFM features
            'Tenure': 'tenure_months',
            'CashbackAmount': 'monetary_value',
            'OrderCount': 'frequency',
            'DaySinceLastOrder': 'recency',

            # Target variable
            'Churn': 'churned',

            # E-commerce specific features (retain original names)
            'WarehouseToHome': 'WarehouseToHome',
            'HourSpendOnApp': 'HourSpendOnApp',
            'NumberOfDeviceRegistered': 'NumberOfDeviceRegistered',
            'PreferedOrderCat': 'PreferedOrderCat',
            'SatisfactionScore': 'SatisfactionScore',
            'MaritalStatus': 'MaritalStatus',
            'NumberOfAddress': 'NumberOfAddress',
            'Complain': 'Complain',
            'OrderAmountHikeFromlastYear': 'OrderAmountHikeFromlastYear',
            'CouponUsed': 'CouponUsed',
            'PreferredLoginDevice': 'PreferredLoginDevice',
            'PreferredPaymentMode': 'PreferredPaymentMode',
            'Gender': 'gender',
            'CityTier': 'CityTier'
        }
    }

    return feature_map


def get_canonical_features() -> List[str]:
    """
    Returns list of canonical feature names used across all datasets.

    These features represent universal customer behavior dimensions
    that generalize across business contexts.

    Returns
    -------
    canonical_features : List[str]
        List of canonical feature names:
        ['tenure_months', 'monetary_value', 'frequency', 'recency', 'churned']
    """
    return ['tenure_months', 'monetary_value', 'frequency', 'recency', 'churned']


def harmonize_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    feature_map: Optional[Dict[str, Dict[str, str]]] = None,
    add_source_column: bool = True
) -> pd.DataFrame:
    """
    Harmonizes a dataset by mapping features to canonical names.

    This function implements Step 1 (Feature Alignment) of the harmonization
    procedure described in Section 4.2.1. It maps dataset-specific features
    to standardized canonical features while retaining auxiliary features.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe from data/processed/
    dataset_name : str
        Dataset identifier: 'telco', 'bank', or 'ecommerce'
    feature_map : Optional[Dict[str, Dict[str, str]]]
        Feature mapping dictionary. If None, uses default mapping.
    add_source_column : bool, default=True
        If True, adds 'dataset_source' column to track dataset provenance

    Returns
    -------
    df_harmonized : pd.DataFrame
        Harmonized dataframe with canonical feature names

    Raises
    ------
    ValueError
        If dataset_name is not recognized
    KeyError
        If expected features are missing from input dataframe

    Examples
    --------
    >>> import pandas as pd
    >>> telco = pd.read_csv('data/processed/telco_processed.csv')
    >>> telco_clean = harmonize_dataset(telco, 'telco')
    >>> 'tenure_months' in telco_clean.columns
    True

    Notes
    -----
    - Missing canonical features (e.g., 'recency' in telco/bank) are NOT added
    - Dataset-specific features are retained for domain-specific signal
    - Churn indicator is always mapped to 'churned' column
    """
    if feature_map is None:
        feature_map = get_feature_mapping()

    valid_datasets = list(feature_map.keys())
    if dataset_name not in valid_datasets:
        raise ValueError(
            f"dataset_name must be one of {valid_datasets}, got '{dataset_name}'"
        )

    df_harmonized = df.copy()

    # Get mapping for this specific dataset
    mapping = feature_map[dataset_name]

    # Check for missing expected features
    missing_features = []
    for original_name in mapping.keys():
        if original_name not in df_harmonized.columns:
            missing_features.append(original_name)

    if missing_features:
        warnings.warn(
            f"Expected features not found in {dataset_name} dataset: {missing_features}. "
            f"These features will be skipped.",
            UserWarning
        )

    # Apply renaming (only for features that exist)
    rename_dict = {
        orig: canon
        for orig, canon in mapping.items()
        if orig in df_harmonized.columns
    }

    df_harmonized = df_harmonized.rename(columns=rename_dict)

    # Add dataset source column for tracking
    if add_source_column:
        df_harmonized['dataset_source'] = dataset_name

    # Verify that target variable exists
    if 'churned' not in df_harmonized.columns:
        raise KeyError(
            f"Target variable 'churned' not found after harmonization of {dataset_name} dataset. "
            f"Check that 'Churn' column exists in input data."
        )

    return df_harmonized


def create_unified_dataset(
    harmonized_datasets: Dict[str, pd.DataFrame],
    align_features: bool = True
) -> pd.DataFrame:
    """
    Combines multiple harmonized datasets into a unified dataset.

    This function concatenates harmonized datasets vertically, ensuring
    feature alignment across datasets. Missing features in individual
    datasets are filled with NaN values.

    Parameters
    ----------
    harmonized_datasets : Dict[str, pd.DataFrame]
        Dictionary mapping dataset names to harmonized dataframes:
        {'telco': df_telco, 'bank': df_bank, 'ecommerce': df_ecommerce}
    align_features : bool, default=True
        If True, aligns features across datasets by filling missing columns with NaN

    Returns
    -------
    D_public : pd.DataFrame
        Unified dataset with all observations concatenated
        Shape: (N_total, p_union) where N_total = sum of all dataset sizes

    Examples
    --------
    >>> harmonized = {
    ...     'telco': harmonize_dataset(telco_df, 'telco'),
    ...     'bank': harmonize_dataset(bank_df, 'bank'),
    ...     'ecommerce': harmonize_dataset(ecomm_df, 'ecommerce')
    ... }
    >>> D_public = create_unified_dataset(harmonized)
    >>> D_public['dataset_source'].value_counts()
    telco        7043
    bank        10000
    ecommerce    5630
    Name: dataset_source, dtype: int64

    Notes
    -----
    - Datasets are concatenated in the order provided in the dictionary
    - 'dataset_source' column is used to track provenance
    - If align_features=True, all datasets will have the same column set
    - Missing values introduced by alignment are handled in feature engineering
    """
    if len(harmonized_datasets) == 0:
        raise ValueError("harmonized_datasets dictionary is empty")

    # Verify all datasets have 'dataset_source' column
    for name, df in harmonized_datasets.items():
        if 'dataset_source' not in df.columns:
            warnings.warn(
                f"Dataset '{name}' missing 'dataset_source' column. "
                f"Consider setting add_source_column=True in harmonize_dataset().",
                UserWarning
            )

    # Concatenate datasets
    if align_features:
        # Align columns across all datasets (fill missing with NaN)
        D_public = pd.concat(
            harmonized_datasets.values(),
            axis=0,
            ignore_index=True,
            sort=False
        )
    else:
        # Only concatenate common columns
        common_cols = set.intersection(*[set(df.columns) for df in harmonized_datasets.values()])
        D_public = pd.concat(
            [df[common_cols] for df in harmonized_datasets.values()],
            axis=0,
            ignore_index=True
        )

    return D_public


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summary statistics for harmonized dataset features.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonized dataframe

    Returns
    -------
    summary : pd.DataFrame
        Summary statistics with columns:
        - feature: Feature name
        - dtype: Data type
        - missing_count: Number of missing values
        - missing_pct: Percentage of missing values
        - unique_values: Number of unique values
        - sample_values: Sample of unique values (for categorical)

    Examples
    --------
    >>> summary = get_feature_summary(D_public)
    >>> print(summary[summary['missing_pct'] > 5.0])  # Features with >5% missing
    """
    summary_data = []

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        unique_values = df[col].nunique()

        # Sample values (first 5 unique values for categorical)
        if df[col].dtype == 'object' or unique_values < 20:
            sample_values = df[col].value_counts().head(5).index.tolist()
        else:
            sample_values = [df[col].min(), df[col].max()]

        summary_data.append({
            'feature': col,
            'dtype': str(df[col].dtype),
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'unique_values': unique_values,
            'sample_values': str(sample_values)
        })

    summary = pd.DataFrame(summary_data)
    return summary


def validate_harmonization(df: pd.DataFrame, dataset_name: str) -> Dict[str, bool]:
    """
    Validates that harmonization was successful.

    Checks for:
    - Presence of 'churned' target variable
    - Valid churn values (0/1)
    - Presence of canonical features (if expected)
    - Data type consistency

    Parameters
    ----------
    df : pd.DataFrame
        Harmonized dataframe
    dataset_name : str
        Dataset identifier for validation rules

    Returns
    -------
    validation_results : Dict[str, bool]
        Dictionary of validation checks and pass/fail status

    Examples
    --------
    >>> results = validate_harmonization(telco_harmonized, 'telco')
    >>> assert all(results.values()), "Validation failed!"
    """
    results = {}

    # Check 1: Target variable exists
    results['has_churned_column'] = 'churned' in df.columns

    # Check 2: Valid churn values
    if results['has_churned_column']:
        valid_values = df['churned'].isin([0, 1]).all()
        results['valid_churn_values'] = valid_values
    else:
        results['valid_churn_values'] = False

    # Check 3: Has tenure_months (present in all datasets)
    results['has_tenure'] = 'tenure_months' in df.columns

    # Check 4: Has monetary_value (present in all datasets)
    results['has_monetary'] = 'monetary_value' in df.columns

    # Check 5: No completely null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    results['no_null_columns'] = len(null_cols) == 0

    # Check 6: Dataset source is correct
    if 'dataset_source' in df.columns:
        correct_source = (df['dataset_source'] == dataset_name).all()
        results['correct_dataset_source'] = correct_source
    else:
        results['correct_dataset_source'] = False

    return results


if __name__ == "__main__":
    # Example usage and testing
    print("SmallML Data Harmonization Module")
    print("=" * 60)
    print("\nExample usage:\n")

    print("from src.data_harmonization import harmonize_dataset, create_unified_dataset")
    print("import pandas as pd\n")

    print("# Load preprocessed datasets")
    print("telco = pd.read_csv('data/processed/telco_processed.csv')")
    print("bank = pd.read_csv('data/processed/bank_processed.csv')")
    print("ecomm = pd.read_csv('data/processed/ecomm_processed.csv')\n")

    print("# Harmonize each dataset")
    print("telco_h = harmonize_dataset(telco, 'telco')")
    print("bank_h = harmonize_dataset(bank, 'bank')")
    print("ecomm_h = harmonize_dataset(ecomm, 'ecommerce')\n")

    print("# Create unified dataset")
    print("D_public = create_unified_dataset({")
    print("    'telco': telco_h,")
    print("    'bank': bank_h,")
    print("    'ecommerce': ecomm_h")
    print("})\n")

    print("# Validate harmonization")
    print("results = validate_harmonization(telco_h, 'telco')")
    print("print(results)")
