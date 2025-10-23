"""
SmallML Framework - Train/Test Split Script
============================================

This script creates stratified train/test splits from the processed unified
dataset for transfer learning model training.

Implements Step 5 (Train-Test Stratification) from Section 4.2.1:
- 80% training set (for CatBoost base model)
- 20% validation set (for evaluating transfer learning quality)
- Stratified by churn label to preserve class balance

Outputs:
- X_train.csv, y_train.csv (training features and labels)
- X_val.csv, y_val.csv (validation features and labels)
- dataset_metadata.json (metadata about splits)

Usage:
    python scripts/create_train_test_split.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from datetime import datetime


def create_train_test_split(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: str = 'churned'
) -> None:
    """
    Creates stratified train/test split from processed dataset.

    Parameters
    ----------
    input_path : Path
        Path to processed unified dataset (D_public_processed.csv)
    output_dir : Path
        Directory to save train/test splits
    test_size : float, default=0.2
        Proportion of data for validation set (0.2 = 20%)
    random_state : int, default=42
        Random seed for reproducibility
    stratify_by : str, default='churned'
        Column name to stratify by (preserves class balance)
    """
    print("="*70)
    print("SmallML Framework - Train/Test Split")
    print("="*70)

    # Load processed dataset
    print(f"\nLoading processed dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")
    print(f"  Size: {input_path.stat().st_size / (1024**2):.2f} MB")

    # Verify required columns exist
    if stratify_by not in df.columns:
        raise ValueError(f"Stratification column '{stratify_by}' not found in dataset")

    if 'dataset_source' not in df.columns:
        print("  Warning: 'dataset_source' column not found. Cannot track source distribution.")

    # Separate features and target
    print(f"\nSeparating features and target...")

    # Columns to exclude from features
    exclude_cols = [stratify_by, 'dataset_source']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[stratify_by]

    print(f"  Features (X): {X.shape}")
    print(f"  Target (y):   {y.shape}")
    print(f"  Churn rate:   {y.mean():.3f} ({y.mean()*100:.1f}%)")

    # Create stratified split
    print(f"\nCreating stratified split (test_size={test_size})...")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"  Train set: {X_train.shape[0]:>6,} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Val set:   {X_val.shape[0]:>6,} samples ({test_size*100:.0f}%)")

    # Verify stratification worked
    train_churn_rate = y_train.mean()
    val_churn_rate = y_val.mean()
    print(f"\n  Train churn rate: {train_churn_rate:.3f} ({train_churn_rate*100:.1f}%)")
    print(f"  Val churn rate:   {val_churn_rate:.3f} ({val_churn_rate*100:.1f}%)")

    diff = abs(train_churn_rate - val_churn_rate)
    if diff < 0.01:
        print(f"  ✓ Stratification successful (difference: {diff:.4f})")
    else:
        print(f"  ⚠ Stratification may have issues (difference: {diff:.4f})")

    # Check dataset source distribution (if available)
    if 'dataset_source' in df.columns:
        print(f"\nDataset source distribution:")

        # Get source for train/val splits
        train_indices = X_train.index
        val_indices = X_val.index

        train_sources = df.loc[train_indices, 'dataset_source'].value_counts()
        val_sources = df.loc[val_indices, 'dataset_source'].value_counts()

        print(f"\n  Training set:")
        for source, count in train_sources.items():
            pct = (count / len(train_indices)) * 100
            print(f"    {source:12} {count:>6,} ({pct:>5.1f}%)")

        print(f"\n  Validation set:")
        for source, count in val_sources.items():
            pct = (count / len(val_indices)) * 100
            print(f"    {source:>12} {count:>6,} ({pct:>5.1f}%)")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train/test splits
    print(f"\nSaving splits to: {output_dir}")

    X_train_path = output_dir / 'X_train.csv'
    y_train_path = output_dir / 'y_train.csv'
    X_val_path = output_dir / 'X_val.csv'
    y_val_path = output_dir / 'y_val.csv'

    print(f"  Writing {X_train_path.name}...")
    X_train.to_csv(X_train_path, index=False)

    print(f"  Writing {y_train_path.name}...")
    y_train.to_csv(y_train_path, index=False, header=True)

    print(f"  Writing {X_val_path.name}...")
    X_val.to_csv(X_val_path, index=False)

    print(f"  Writing {y_val_path.name}...")
    y_val.to_csv(y_val_path, index=False, header=True)

    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'input_file': str(input_path),
        'random_state': random_state,
        'test_size': test_size,
        'stratify_by': stratify_by,
        'train': {
            'n_samples': int(len(X_train)),
            'n_features': int(X_train.shape[1]),
            'churn_rate': float(y_train.mean()),
            'churn_count': int(y_train.sum()),
            'file_X': str(X_train_path.name),
            'file_y': str(y_train_path.name)
        },
        'validation': {
            'n_samples': int(len(X_val)),
            'n_features': int(X_val.shape[1]),
            'churn_rate': float(y_val.mean()),
            'churn_count': int(y_val.sum()),
            'file_X': str(X_val_path.name),
            'file_y': str(y_val_path.name)
        },
        'feature_names': X_train.columns.tolist()
    }

    # Add dataset source distribution if available
    if 'dataset_source' in df.columns:
        metadata['train']['source_distribution'] = train_sources.to_dict()
        metadata['validation']['source_distribution'] = val_sources.to_dict()

    metadata_path = output_dir / 'dataset_metadata.json'
    print(f"  Writing {metadata_path.name}...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("Train/Test Split Complete!")
    print("="*70)

    print(f"\nOutput Files:")
    print(f"  - {X_train_path} ({X_train_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"  - {y_train_path} ({y_train_path.stat().st_size / 1024:.2f} KB)")
    print(f"  - {X_val_path} ({X_val_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"  - {y_val_path} ({y_val_path.stat().st_size / 1024:.2f} KB)")
    print(f"  - {metadata_path}")

    print(f"\nSummary Statistics:")
    print(f"  Total samples:        {len(df):>6,}")
    print(f"  Training samples:     {len(X_train):>6,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Validation samples:   {len(X_val):>6,} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Number of features:   {X_train.shape[1]:>6}")
    print(f"  Overall churn rate:   {y.mean():>6.3f}")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("  1. Review the splits in data/harmonized/")
    print("  2. Run scripts/generate_table_4_1.py to create Table 4.1")
    print("  3. Proceed to: Transfer Learning (CatBoost training)")
    print("  4. Load data for training:")
    print("     X_train = pd.read_csv('data/harmonized/X_train.csv')")
    print("     y_train = pd.read_csv('data/harmonized/y_train.csv')['churned']")


def main():
    """Main execution function."""
    # Define paths
    input_path = Path('data/harmonized/D_public_processed.csv')
    output_dir = Path('data/harmonized')

    # Check if input file exists
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("\nPlease run the following first:")
        print("  1. python preprocess_datasets.py")
        print("  2. Run notebooks/02_harmonization_and_encoding.ipynb")
        print("\nThis will create the required D_public_processed.csv file.")
        return

    # Create train/test split
    try:
        create_train_test_split(
            input_path=input_path,
            output_dir=output_dir,
            test_size=0.2,
            random_state=42,
            stratify_by='churned'
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
