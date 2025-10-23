"""
This script preprocesses the three churn datasets (telco, bank, ecommerce). The script performs:
1. Data cleaning (handling missing values, type conversions)
2. Feature standardization
3. Target variable harmonization
4. Removal of unnecessary columns
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path


def preprocess_bank_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess bank_churn.csv dataset.

    Transformations:
    - Drop 'RowNumber' (just an index)
    - Drop 'Surname' (not predictive, privacy concern)
    - Rename 'Exited' → 'Churn' (standardization)
    - Drop 'CustomerId' (identifier only)

    Args:
        df: Raw bank churn dataframe

    Returns:
        Preprocessed dataframe
    """
    print("\n" + "="*60)
    print("PREPROCESSING: bank_churn.csv")
    print("="*60)

    df_clean = df.copy()

    # Initial shape
    print(f"Initial shape: {df_clean.shape}")
    print(f"Initial columns: {df_clean.columns.tolist()}")

    # Drop unnecessary columns
    cols_to_drop = ['RowNumber', 'Surname', 'CustomerId']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]

    print(f"\nDropping columns: {existing_cols_to_drop}")
    df_clean = df_clean.drop(existing_cols_to_drop, axis=1)

    # Rename target column
    if 'Exited' in df_clean.columns:
        print("Renaming 'Exited' → 'Churn'")
        df_clean = df_clean.rename(columns={'Exited': 'Churn'})

    # Check for missing values
    missing = df_clean.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0]}")
    if missing.sum() == 0:
        print("No missing values - dataset is pristine! ✓")

    # Verify churn distribution
    churn_dist = df_clean['Churn'].value_counts()
    churn_rate = df_clean['Churn'].mean()
    print(f"\nChurn distribution:\n{churn_dist}")
    print(f"Churn rate: {churn_rate:.3f} ({churn_rate*100:.1f}%)")

    print(f"\nFinal shape: {df_clean.shape}")
    print(f"Final columns: {df_clean.columns.tolist()}")

    return df_clean


def preprocess_telco_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess telco_churn.csv dataset.

    Transformations:
    - Convert 'TotalCharges' from object to float
    - Impute missing TotalCharges (11 empty strings) with tenure × MonthlyCharges
    - Convert 'Churn' from Yes/No to 1/0
    - Drop 'customerID' (identifier only)
    - Optional: Lowercase column names for consistency

    Args:
        df: Raw telco churn dataframe

    Returns:
        Preprocessed dataframe
    """
    print("\n" + "="*60)
    print("PREPROCESSING: telco_churn.csv")
    print("="*60)

    df_clean = df.copy()

    # Initial shape
    print(f"Initial shape: {df_clean.shape}")
    print(f"Initial columns: {df_clean.columns.tolist()}")

    # Drop customer ID
    if 'customerID' in df_clean.columns:
        print("\nDropping 'customerID' column")
        df_clean = df_clean.drop('customerID', axis=1)

    # Fix TotalCharges type issue
    print("\nFixing 'TotalCharges' data type...")
    if df_clean['TotalCharges'].dtype == 'object':
        # Count empty strings
        empty_count = (df_clean['TotalCharges'] == ' ').sum() + (df_clean['TotalCharges'] == '').sum()
        print(f"Found {empty_count} empty string values in TotalCharges")

        # Convert to numeric (empty strings become NaN)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

        # Impute missing values with tenure × MonthlyCharges
        missing_mask = df_clean['TotalCharges'].isnull()
        if missing_mask.sum() > 0:
            print(f"Imputing {missing_mask.sum()} missing TotalCharges values using tenure × MonthlyCharges")
            df_clean.loc[missing_mask, 'TotalCharges'] = \
                df_clean.loc[missing_mask, 'tenure'] * df_clean.loc[missing_mask, 'MonthlyCharges']

            # Verify average tenure for imputed customers
            avg_tenure_imputed = df_clean.loc[missing_mask, 'tenure'].mean()
            print(f"Average tenure for imputed customers: {avg_tenure_imputed:.1f} months")

    # Convert Churn from Yes/No to 1/0
    print("\nConverting 'Churn' from Yes/No to 1/0...")
    if df_clean['Churn'].dtype == 'object':
        original_dist = df_clean['Churn'].value_counts()
        print(f"Original distribution:\n{original_dist}")

        df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)

        churn_rate = df_clean['Churn'].mean()
        print(f"New churn rate: {churn_rate:.3f} ({churn_rate*100:.1f}%)")

    # Optional: Lowercase column names for consistency
    # print("\nLowercasing column names for consistency...")
    # df_clean.columns = df_clean.columns.str.lower()

    # Check for any remaining missing values
    missing = df_clean.isnull().sum()
    print(f"\nMissing values after preprocessing:\n{missing[missing > 0]}")
    if missing.sum() == 0:
        print("No missing values remaining ✓")

    print(f"\nFinal shape: {df_clean.shape}")
    print(f"Final columns: {df_clean.columns.tolist()}")

    return df_clean


def preprocess_ecommerce_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess ecommerce_churn.csv dataset.

    Transformations:
    - Median imputation for numerical columns with missing values
    - Add missingness indicator flags (optional but recommended)
    - Drop 'CustomerID' (identifier only)

    Missing values in:
    - Tenure (264 missing, 4.7%)
    - DaySinceLastOrder (307 missing, 5.5%)
    - WarehouseToHome (251 missing, 4.5%)
    - HourSpendOnApp (255 missing, 4.5%)
    - OrderAmountHikeFromlastYear (265 missing, 4.7%)
    - CouponUsed (256 missing, 4.5%)
    - OrderCount (258 missing, 4.6%)

    Args:
        df: Raw ecommerce churn dataframe

    Returns:
        Preprocessed dataframe
    """
    print("\n" + "="*60)
    print("PREPROCESSING: ecommerce_churn.csv")
    print("="*60)

    df_clean = df.copy()

    # Initial shape
    print(f"Initial shape: {df_clean.shape}")
    print(f"Initial columns: {df_clean.columns.tolist()}")

    # Drop customer ID
    if 'CustomerID' in df_clean.columns:
        print("\nDropping 'CustomerID' column")
        df_clean = df_clean.drop('CustomerID', axis=1)

    # Identify columns with missing values
    missing_before = df_clean.isnull().sum()
    missing_cols = missing_before[missing_before > 0]

    print(f"\nMissing values before imputation:")
    for col, count in missing_cols.items():
        pct = (count / len(df_clean)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")

    # List of numerical columns with missing values
    numerical_cols_with_missing = [
        'Tenure', 'WarehouseToHome', 'HourSpendOnApp',
        'OrderAmountHikeFromlastYear', 'CouponUsed',
        'OrderCount', 'DaySinceLastOrder'
    ]

    # Filter to only columns that actually exist and have missing values
    numerical_cols_with_missing = [
        col for col in numerical_cols_with_missing
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0
    ]

    if len(numerical_cols_with_missing) > 0:
        print(f"\nApplying median imputation to {len(numerical_cols_with_missing)} columns...")

        # Add missingness indicator flags (before imputation)
        print("Creating missingness indicator flags...")
        for col in numerical_cols_with_missing:
            flag_name = f'{col}_missing'
            df_clean[flag_name] = df_clean[col].isnull().astype(int)
            n_missing = df_clean[flag_name].sum()
            print(f"  {flag_name}: {n_missing} observations marked")

        # Perform median imputation
        imputer = SimpleImputer(strategy='median')
        df_clean[numerical_cols_with_missing] = imputer.fit_transform(
            df_clean[numerical_cols_with_missing]
        )

        print("\nMedian imputation complete ✓")

    # Verify no missing values remain in numerical columns
    missing_after = df_clean[numerical_cols_with_missing].isnull().sum() if numerical_cols_with_missing else pd.Series()
    if len(missing_after) > 0 and missing_after.sum() == 0:
        print("All numerical missing values imputed successfully ✓")

    # Check for any remaining missing values in the entire dataset
    total_missing = df_clean.isnull().sum().sum()
    if total_missing > 0:
        print(f"\nRemaining missing values: {total_missing}")
        print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
    else:
        print("\nNo missing values remaining ✓")

    # Verify churn distribution
    churn_dist = df_clean['Churn'].value_counts()
    churn_rate = df_clean['Churn'].mean()
    print(f"\nChurn distribution:\n{churn_dist}")
    print(f"Churn rate: {churn_rate:.3f} ({churn_rate*100:.1f}%)")

    print(f"\nFinal shape: {df_clean.shape}")
    print(f"Added {len([c for c in df_clean.columns if '_missing' in c])} missingness indicator flags")

    return df_clean


def main():
    """
    Main preprocessing pipeline.

    Loads raw datasets, applies preprocessing functions, and saves
    processed datasets to data/processed/ folder.
    """
    print("="*60)
    print("SmallML Framework - Data Preprocessing Pipeline")
    print("="*60)
    print("\nThis script will preprocess three churn datasets:")
    print("1. bank_churn.csv")
    print("2. telco_churn.csv")
    print("3. ecommerce_churn.csv")
    print("\nProcessed files will be saved to: data/processed/")

    # Define file paths
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')

    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load raw datasets
        print("\n" + "="*60)
        print("LOADING RAW DATASETS")
        print("="*60)

        print("\nLoading bank_churn.csv...")
        bank_raw = pd.read_csv(raw_dir / 'bank_churn.csv')
        print(f"Loaded: {bank_raw.shape}")

        print("\nLoading telco_churn.csv...")
        telco_raw = pd.read_csv(raw_dir / 'telco_churn.csv')
        print(f"Loaded: {telco_raw.shape}")

        print("\nLoading ecommerce_churn.csv...")
        ecomm_raw = pd.read_csv(raw_dir / 'ecommerce_churn.csv')
        print(f"Loaded: {ecomm_raw.shape}")

        # Preprocess datasets
        bank_processed = preprocess_bank_churn(bank_raw)
        telco_processed = preprocess_telco_churn(telco_raw)
        ecomm_processed = preprocess_ecommerce_churn(ecomm_raw)

        # Save processed datasets
        print("\n" + "="*60)
        print("SAVING PROCESSED DATASETS")
        print("="*60)

        bank_output = processed_dir / 'bank_processed.csv'
        telco_output = processed_dir / 'telco_processed.csv'
        ecomm_output = processed_dir / 'ecomm_processed.csv'

        print(f"\nSaving to {bank_output}...")
        bank_processed.to_csv(bank_output, index=False)
        print(f"Saved: {bank_processed.shape}")

        print(f"\nSaving to {telco_output}...")
        telco_processed.to_csv(telco_output, index=False)
        print(f"Saved: {telco_processed.shape}")

        print(f"\nSaving to {ecomm_output}...")
        ecomm_processed.to_csv(ecomm_output, index=False)
        print(f"Saved: {ecomm_processed.shape}")

        # Summary
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE ✓")
        print("="*60)

        print("\nProcessed Dataset Summary:")
        print(f"  Bank:      {bank_processed.shape[0]:>6} rows × {bank_processed.shape[1]:>2} features")
        print(f"  Telco:     {telco_processed.shape[0]:>6} rows × {telco_processed.shape[1]:>2} features")
        print(f"  E-commerce:{ecomm_processed.shape[0]:>6} rows × {ecomm_processed.shape[1]:>2} features")
        print(f"  TOTAL:     {bank_processed.shape[0] + telco_processed.shape[0] + ecomm_processed.shape[0]:>6} rows")

        print("\nChurn Rates:")
        print(f"  Bank:       {bank_processed['Churn'].mean():.3f} ({bank_processed['Churn'].mean()*100:.1f}%)")
        print(f"  Telco:      {telco_processed['Churn'].mean():.3f} ({telco_processed['Churn'].mean()*100:.1f}%)")
        print(f"  E-commerce: {ecomm_processed['Churn'].mean():.3f} ({ecomm_processed['Churn'].mean()*100:.1f}%)")

        print("\nFiles saved to:")
        print(f"  - {bank_output}")
        print(f"  - {telco_output}")
        print(f"  - {ecomm_output}")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find file: {e}")
        print("Please ensure all raw CSV files are in data/raw/ directory:")
        print("  - data/raw/bank_churn.csv")
        print("  - data/raw/telco_churn.csv")
        print("  - data/raw/ecommerce_churn.csv")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
