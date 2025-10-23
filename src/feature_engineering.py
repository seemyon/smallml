"""
SmallML Framework - Feature Engineering Module
==============================================

This module provides functions for feature engineering, normalization, and
encoding of harmonized datasets.

Implements Steps 2-4 of the harmonization procedure (Section 4.2.1):
- Step 2: Missing value handling (already done in preprocessing)
- Step 3: Target variable standardization (handled in harmonization)
- Step 4: Feature normalization and categorical encoding

Key Functions:
- normalize_numerical_features(): Standardize numerical features
- encode_categorical_features(): One-hot encode categorical variables
- consolidate_rare_categories(): Merge infrequent categories
- prepare_for_modeling(): Complete feature engineering pipeline

"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
import warnings


def identify_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically identifies numerical and categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    feature_types : Dict[str, List[str]]
        Dictionary with keys 'numerical', 'categorical', 'binary', 'target'

    Notes
    -----
    - Binary features (only 0/1 values) are kept separate from categorical
    - Target variable 'churned' is identified separately
    - Features ending with '_missing' are treated as binary indicators
    """
    feature_types = {
        'numerical': [],
        'categorical': [],
        'binary': [],
        'target': [],
        'identifier': []
    }

    for col in df.columns:
        # Skip target variable
        if col == 'churned':
            feature_types['target'].append(col)
            continue

        # Skip dataset source identifier
        if col == 'dataset_source':
            feature_types['identifier'].append(col)
            continue

        # Check if binary (0/1 only)
        if df[col].dtype in ['int64', 'float64']:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                feature_types['binary'].append(col)
            else:
                feature_types['numerical'].append(col)
        elif df[col].dtype == 'object':
            feature_types['categorical'].append(col)
        else:
            # Default to numerical for other dtypes
            feature_types['numerical'].append(col)

    return feature_types


def consolidate_rare_categories(
    df: pd.DataFrame,
    categorical_cols: List[str],
    threshold: float = 0.01,
    other_label: str = 'Other'
) -> pd.DataFrame:
    """
    Consolidates rare categories (< threshold frequency) into 'Other' category.

    This prevents high-dimensional sparsity after one-hot encoding and
    improves model generalization by grouping infrequent categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : List[str]
        List of categorical column names to process
    threshold : float, default=0.01
        Minimum frequency threshold (as proportion). Categories below this
        are consolidated into 'Other'
    other_label : str, default='Other'
        Label for consolidated rare categories

    Returns
    -------
    df_consolidated : pd.DataFrame
        Dataframe with rare categories consolidated

    Examples
    --------
    >>> df_clean = consolidate_rare_categories(df, ['PaymentMethod'], threshold=0.01)
    >>> df_clean['PaymentMethod'].value_counts()
    Electronic check    2365
    Mailed check        1612
    Other               1200
    dtype: int64
    """
    df_consolidated = df.copy()

    for col in categorical_cols:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in dataframe. Skipping.", UserWarning)
            continue

        # Calculate value counts and frequencies
        value_counts = df_consolidated[col].value_counts()
        frequencies = value_counts / len(df_consolidated)

        # Identify rare categories
        rare_categories = frequencies[frequencies < threshold].index.tolist()

        if len(rare_categories) > 0:
            # Replace rare categories with 'Other'
            df_consolidated[col] = df_consolidated[col].replace(rare_categories, other_label)

            print(f"  {col}: Consolidated {len(rare_categories)} rare categories → '{other_label}'")

    return df_consolidated


def normalize_numerical_features(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
    exclude_binary: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalizes numerical features to zero mean and unit variance.

    Implements Step 4 (Feature Normalization) from Section 4.2.1:
    x̃_j = (x_j - μ_j) / σ_j

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    numerical_cols : Optional[List[str]]
        List of numerical columns to normalize. If None, auto-detected.
    scaler : Optional[StandardScaler]
        Pre-fitted scaler (for validation/test sets). If None, fits new scaler.
    exclude_binary : bool, default=True
        If True, excludes binary (0/1) features from normalization

    Returns
    -------
    df_normalized : pd.DataFrame
        Dataframe with normalized numerical features
    scaler : StandardScaler
        Fitted scaler object (for applying to validation/test sets)

    Examples
    --------
    >>> # Fit on training set
    >>> df_train_norm, scaler = normalize_numerical_features(df_train)
    >>> # Apply to validation set
    >>> df_val_norm, _ = normalize_numerical_features(df_val, scaler=scaler)

    Notes
    -----
    - Binary features (0/1) are typically NOT normalized
    - Missing values should be imputed before normalization
    - Scaler should be saved for consistent test set transformation
    """
    df_normalized = df.copy()

    # Auto-detect numerical columns if not provided
    if numerical_cols is None:
        feature_types = identify_feature_types(df)
        numerical_cols = feature_types['numerical']

        if exclude_binary:
            # Remove binary features from normalization
            binary_cols = feature_types['binary']
            numerical_cols = [col for col in numerical_cols if col not in binary_cols]

    # Filter to only existing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if len(numerical_cols) == 0:
        warnings.warn("No numerical columns found for normalization.", UserWarning)
        return df_normalized, StandardScaler()

    # Fit or use existing scaler
    if scaler is None:
        scaler = StandardScaler()
        df_normalized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print(f"Normalized {len(numerical_cols)} numerical features")
    else:
        # Use pre-fitted scaler (for validation/test sets)
        df_normalized[numerical_cols] = scaler.transform(df[numerical_cols])
        print(f"Applied scaler to {len(numerical_cols)} numerical features")

    return df_normalized, scaler


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    drop_first: bool = False,
    handle_unknown: str = 'ignore'
) -> pd.DataFrame:
    """
    One-hot encodes categorical features.

    Implements Step 4 (Categorical Encoding) from Section 4.2.1.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : Optional[List[str]]
        List of categorical columns to encode. If None, auto-detected.
    drop_first : bool, default=False
        If True, drops first category to avoid multicollinearity
    handle_unknown : str, default='ignore'
        How to handle unknown categories in test set:
        - 'ignore': Unknown categories become all zeros
        - 'error': Raise error for unknown categories

    Returns
    -------
    df_encoded : pd.DataFrame
        Dataframe with one-hot encoded categorical features

    Examples
    --------
    >>> df_encoded = encode_categorical_features(df, ['gender', 'Contract'])
    >>> df_encoded.columns
    Index(['tenure_months', 'monetary_value', ..., 'gender_Female',
           'gender_Male', 'Contract_Month-to-month', ...])

    Notes
    -----
    - Original categorical columns are dropped after encoding
    - Encoded columns follow naming: {original}_{category}
    - Binary features are NOT encoded (already 0/1)
    """
    df_encoded = df.copy()

    # Auto-detect categorical columns if not provided
    if categorical_cols is None:
        feature_types = identify_feature_types(df)
        categorical_cols = feature_types['categorical']

    # Filter to only existing columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    if len(categorical_cols) == 0:
        warnings.warn("No categorical columns found for encoding.", UserWarning)
        return df_encoded

    # One-hot encode
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=categorical_cols,
        drop_first=drop_first,
        dtype=int
    )

    n_new_cols = len(df_encoded.columns) - len(df.columns) + len(categorical_cols)
    print(f"One-hot encoded {len(categorical_cols)} categorical features → {n_new_cols} binary columns")

    return df_encoded


def prepare_for_modeling(
    df: pd.DataFrame,
    normalize: bool = True,
    encode_categoricals: bool = True,
    consolidate_rare: bool = True,
    rare_threshold: float = 0.01,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Complete feature engineering pipeline.

    Applies all feature engineering steps in sequence:
    1. Consolidate rare categories (optional)
    2. Normalize numerical features (optional)
    3. One-hot encode categorical features (optional)

    Parameters
    ----------
    df : pd.DataFrame
        Harmonized input dataframe
    normalize : bool, default=True
        If True, normalizes numerical features
    encode_categoricals : bool, default=True
        If True, one-hot encodes categorical features
    consolidate_rare : bool, default=True
        If True, consolidates rare categories before encoding
    rare_threshold : float, default=0.01
        Minimum frequency for rare category consolidation
    scaler : Optional[StandardScaler]
        Pre-fitted scaler (for validation/test sets)

    Returns
    -------
    df_processed : pd.DataFrame
        Fully processed dataframe ready for modeling
    scaler : Optional[StandardScaler]
        Fitted scaler (None if normalize=False)

    Examples
    --------
    >>> # Training set: fit and transform
    >>> df_train_ready, scaler = prepare_for_modeling(df_train_harmonized)
    >>>
    >>> # Validation set: transform only
    >>> df_val_ready, _ = prepare_for_modeling(
    ...     df_val_harmonized,
    ...     scaler=scaler  # Use fitted scaler
    ... )

    Notes
    -----
    - This is the main function to use for complete preprocessing
    - Always fit on training set, then apply to validation/test sets
    - Order matters: consolidate → normalize → encode
    """
    print("\n" + "="*60)
    print("Feature Engineering Pipeline")
    print("="*60)

    df_processed = df.copy()

    # Identify feature types
    feature_types = identify_feature_types(df_processed)
    print(f"\nFeature inventory:")
    print(f"  Numerical: {len(feature_types['numerical'])} features")
    print(f"  Categorical: {len(feature_types['categorical'])} features")
    print(f"  Binary: {len(feature_types['binary'])} features")
    print(f"  Target: {feature_types['target']}")

    # Step 1: Consolidate rare categories (only if fitting, not transforming)
    if consolidate_rare and encode_categoricals and scaler is None:
        print(f"\nStep 1: Consolidating rare categories (threshold={rare_threshold})...")
        df_processed = consolidate_rare_categories(
            df_processed,
            feature_types['categorical'],
            threshold=rare_threshold
        )

    # Step 2: Normalize numerical features
    fitted_scaler = None
    if normalize:
        print("\nStep 2: Normalizing numerical features...")
        df_processed, fitted_scaler = normalize_numerical_features(
            df_processed,
            numerical_cols=feature_types['numerical'],
            scaler=scaler,
            exclude_binary=True
        )

    # Step 3: Encode categorical features
    if encode_categoricals:
        print("\nStep 3: One-hot encoding categorical features...")
        df_processed = encode_categorical_features(
            df_processed,
            categorical_cols=feature_types['categorical'],
            drop_first=False  # Keep all categories for interpretability
        )

    print("\n" + "="*60)
    print(f"Feature engineering complete!")
    print(f"Final shape: {df_processed.shape}")
    print("="*60)

    return df_processed, fitted_scaler


def get_feature_importance_compatible_names(df: pd.DataFrame) -> List[str]:
    """
    Returns feature names safe for modeling (no special characters).

    Some ML libraries (CatBoost, XGBoost) have issues with special characters
    in column names. This function cleans feature names.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with encoded features

    Returns
    -------
    clean_names : List[str]
        List of cleaned feature names

    Examples
    --------
    >>> clean_names = get_feature_importance_compatible_names(df_encoded)
    >>> df_encoded.columns = clean_names
    """
    clean_names = []

    for col in df.columns:
        # Replace special characters
        clean_name = col.replace('[', '_').replace(']', '_')
        clean_name = clean_name.replace('<', 'lt').replace('>', 'gt')
        clean_name = clean_name.replace(' ', '_').replace('-', '_')
        clean_name = clean_name.replace('(', '').replace(')', '')

        clean_names.append(clean_name)

    return clean_names


if __name__ == "__main__":
    # Example usage
    print("SmallML Feature Engineering Module")
    print("=" * 60)
    print("\nExample usage:\n")

    print("from src.feature_engineering import prepare_for_modeling")
    print("import pandas as pd\n")

    print("# Load harmonized dataset")
    print("df_harmonized = pd.read_csv('data/harmonized/D_public_harmonized.csv')\n")

    print("# Apply feature engineering (training set)")
    print("df_processed, scaler = prepare_for_modeling(df_harmonized)\n")

    print("# Separate features and target")
    print("X = df_processed.drop(['churned', 'dataset_source'], axis=1)")
    print("y = df_processed['churned']\n")

    print("# Split into train/test")
    print("from sklearn.model_selection import train_test_split")
    print("X_train, X_val, y_train, y_val = train_test_split(")
    print("    X, y, test_size=0.2, stratify=y, random_state=42")
    print(")")
