"""
SmallML Framework - Table 4.1 Generation Script
================================================

This script generates **Table 4.1: Public Datasets for Transfer Learning**
from the SmallML white paper (Section 4.2.1).

Table 4.1 presents summary statistics for the public datasets used for
transfer learning pre-training, including:
- Dataset name
- Source (Kaggle, UCI, etc.)
- Number of samples
- Number of features
- Churn rate
- Business type

Outputs:
- results/tables/table_4_1.csv (machine-readable format)
- results/tables/table_4_1.md (markdown format for documentation)
- Printed summary to console

Usage:
    python scripts/generate_table_4_1.py

Author: SmallML Framework
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def generate_table_4_1(
    processed_dir: Path,
    output_dir: Path
) -> pd.DataFrame:
    """
    Generates Table 4.1 from processed datasets.

    Parameters
    ----------
    processed_dir : Path
        Directory containing processed datasets
    output_dir : Path
        Directory to save generated table

    Returns
    -------
    table_4_1 : pd.DataFrame
        Table 4.1 dataframe
    """
    print("="*70)
    print("Generating Table 4.1: Public Datasets for Transfer Learning")
    print("="*70)

    # Dataset metadata
    dataset_metadata = [
        {
            'dataset_name': 'Telco Customer Churn',
            'source': 'Kaggle',
            'business_type': 'Telecommunications',
            'file': 'telco_processed.csv',
            'features_original': 19  # After preprocessing (removed customerID)
        },
        {
            'dataset_name': 'Bank Customer Churn',
            'source': 'Kaggle',
            'business_type': 'Financial Services',
            'file': 'bank_processed.csv',
            'features_original': 11  # After preprocessing
        },
        {
            'dataset_name': 'E-commerce Customer Churn',
            'source': 'Kaggle',
            'business_type': 'Online Retail',
            'file': 'ecomm_processed.csv',
            'features_original': 19  # After preprocessing + missingness flags
        }
    ]

    # Load datasets and compute statistics
    print("\nLoading preprocessed datasets...\n")

    table_data = []

    for meta in dataset_metadata:
        file_path = processed_dir / meta['file']

        if not file_path.exists():
            print(f"  ⚠ Warning: {meta['file']} not found. Skipping.")
            continue

        # Load dataset
        df = pd.read_csv(file_path)

        # Compute statistics
        n_samples = len(df)

        # Identify target column (should be 'Churn' or 'churned')
        if 'Churn' in df.columns:
            target_col = 'Churn'
        elif 'churned' in df.columns:
            target_col = 'churned'
        else:
            print(f"  ⚠ Warning: No churn column found in {meta['file']}")
            churn_rate = np.nan

        churn_rate = df[target_col].mean()

        # Count features (exclude target)
        n_features = len(df.columns) - 1  # Exclude target variable

        # Create row
        row = {
            'Dataset Name': meta['dataset_name'],
            'Source': meta['source'],
            '# Samples': n_samples,
            '# Features': n_features,
            'Churn Rate': churn_rate,
            'Business Type': meta['business_type']
        }

        table_data.append(row)

        print(f"  ✓ {meta['dataset_name']:30} {n_samples:>6,} samples, {n_features:>2} features, churn={churn_rate:.1%}")

    # Create DataFrame
    table_4_1 = pd.DataFrame(table_data)

    # Add combined row
    combined_row = {
        'Dataset Name': '**Combined**',
        'Source': '**Multiple**',
        '# Samples': table_4_1['# Samples'].sum(),
        '# Features': 23,  # As per Section 4.2.1 (after harmonization)
        'Churn Rate': np.average(
            table_4_1['Churn Rate'],
            weights=table_4_1['# Samples']
        ),
        'Business Type': '**Mixed**'
    }

    # Append combined row
    table_4_1 = pd.concat([table_4_1, pd.DataFrame([combined_row])], ignore_index=True)

    print(f"\n  ✓ Combined dataset:              {combined_row['# Samples']:>6,} samples, {combined_row['# Features']:>2} features, churn={combined_row['Churn Rate']:.1%}")

    return table_4_1


def save_table(table: pd.DataFrame, output_dir: Path) -> None:
    """
    Saves table in multiple formats.

    Parameters
    ----------
    table : pd.DataFrame
        Table 4.1 dataframe
    output_dir : Path
        Output directory
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV (machine-readable)
    csv_path = output_dir / 'table_4_1.csv'
    print(f"\nSaving CSV to: {csv_path}")
    table.to_csv(csv_path, index=False)
    print(f"  ✓ Saved ({csv_path.stat().st_size / 1024:.2f} KB)")

    # Save as Markdown (documentation-friendly)
    md_path = output_dir / 'table_4_1.md'
    print(f"\nSaving Markdown to: {md_path}")

    # Format table for markdown
    table_formatted = table.copy()
    table_formatted['# Samples'] = table_formatted['# Samples'].apply(lambda x: f"{x:,}")
    table_formatted['Churn Rate'] = table_formatted['Churn Rate'].apply(lambda x: f"{x:.1%}")

    with open(md_path, 'w') as f:
        f.write("# Table 4.1: Public Datasets for Transfer Learning Pre-training\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(table_formatted.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("**Notes:**\n")
        f.write("- # Features reported after preprocessing (ID columns removed, missing values handled)\n")
        f.write("- Combined dataset has 23 features after harmonization (Section 4.2.1)\n")
        f.write("- Churn Rate for Combined is weighted average by dataset size\n")
        f.write("- These datasets form the foundation for Layer 1 transfer learning (Section 4.2)\n")

    print(f"  ✓ Saved ({md_path.stat().st_size / 1024:.2f} KB)")


def print_table_summary(table: pd.DataFrame) -> None:
    """
    Prints formatted table to console.

    Parameters
    ----------
    table : pd.DataFrame
        Table 4.1 dataframe
    """
    print("\n" + "="*70)
    print("Table 4.1: Public Datasets for Transfer Learning Pre-training")
    print("="*70)
    print()

    # Format for display
    table_display = table.copy()
    table_display['# Samples'] = table_display['# Samples'].apply(lambda x: f"{x:>8,}")
    table_display['# Features'] = table_display['# Features'].apply(lambda x: f"{x:>3}")
    table_display['Churn Rate'] = table_display['Churn Rate'].apply(lambda x: f"{x:>6.1%}")

    # Print formatted table
    print(table_display.to_string(index=False))
    print()


def main():
    """Main execution function."""
    # Define paths
    processed_dir = Path('data/processed')
    output_dir = Path('results/tables')

    # Check if processed directory exists
    if not processed_dir.exists():
        print(f"ERROR: Processed data directory not found: {processed_dir}")
        print("\nPlease run the following first:")
        print("  python preprocess_datasets.py")
        return

    # Generate table
    try:
        table_4_1 = generate_table_4_1(processed_dir, output_dir)

        # Save table
        save_table(table_4_1, output_dir)

        # Print summary
        print_table_summary(table_4_1)

        # Success message
        print("="*70)
        print("Table 4.1 Generation Complete!")
        print("="*70)

        print("\nOutput Files:")
        print(f"  - results/tables/table_4_1.csv (machine-readable)")
        print(f"  - results/tables/table_4_1.md (documentation)")

        print("\nKey Statistics:")
        print(f"  Total datasets:       {len(table_4_1) - 1}")  # Exclude combined row
        print(f"  Total samples:        {table_4_1.iloc[-1]['# Samples']:,}")
        print(f"  Features (combined):  {table_4_1.iloc[-1]['# Features']}")
        print(f"  Avg churn rate:       {table_4_1.iloc[-1]['Churn Rate']:.1%}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
