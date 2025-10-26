"""
Create Synthetic SME Datasets for Hierarchical Bayesian Model

This script generates synthetic SME datasets by sampling from harmonized
public data and adding business-specific noise.

"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer2_bayesian import SMEDataGenerator


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Create synthetic SME datasets"
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Quick test mode (J=3, n=30)'
    )
    args = parser.parse_args()

    print(f"{'=' * 80}")
    print("CREATE SYNTHETIC SME DATASETS")
    print(f"{'=' * 80}")
    print(f"Mode: {'TEST' if args.test_mode else 'FULL'}")
    print(f"Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    if args.test_mode:
        J = 3               # 3 SMEs for quick testing
        n_per_sme = 30      # 30 customers each
        noise_scale = 0.1
    else:
        J = 15              # 15 SMEs (updated for publication)
        n_per_sme = 100     # 100 customers each (more robust estimates)
        noise_scale = 0.1   # 10% of feature std

    random_seed = 42
    stratify = True

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "harmonized"
    output_dir = project_root / "data" / "sme_datasets"
    priors_path = project_root / "models" / "transfer_learning" / "priors.pkl"

    print(f"{'=' * 80}")
    print("[Step 1/4] Validating prerequisites...")
    print(f"{'=' * 80}")

    # Check data exists
    required_files = [
        data_dir / "X_train.csv",
        data_dir / "y_train.csv",
        data_dir / "D_public_processed.csv"
    ]

    missing_files = []
    for filepath in required_files:
        if filepath.exists():
            print(f"✓ Found: {filepath.name}")
        else:
            print(f"✗ Missing: {filepath}")
            missing_files.append(filepath)

    if missing_files:
        print(f"\n✗ ERROR: Missing required files")
        print(f"  Please run data harmonization first")
        sys.exit(1)

    # Check priors (for reference, not required)
    if priors_path.exists():
        print(f"✓ Found: priors.pkl")
    else:
        print(f"⚠ Warning: priors.pkl not found")
        print(f"  This is OK, but you'll need it for training")

    print(f"\n{'=' * 80}")
    print("[Step 2/4] Loading harmonized data...")
    print(f"{'=' * 80}")

    try:
        # Load training data
        X_train = pd.read_csv(data_dir / "X_train.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()

        print(f"✓ Loaded training data")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Churn rate: {y_train.mean():.3f}")

    except Exception as e:
        print(f"\n✗ ERROR loading data: {e}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 3/4] Generating SME datasets...")
    print(f"{'=' * 80}")
    print(f"Configuration:")
    print(f"  Number of SMEs (J): {J}")
    print(f"  Customers per SME (n): {n_per_sme}")
    print(f"  Noise scale: {noise_scale}")
    print(f"  Stratified sampling: {stratify}")
    print(f"  Random seed: {random_seed}")
    print(f"  Total customers: {J * n_per_sme}")

    try:
        # Initialize generator
        generator = SMEDataGenerator(
            X=X_train,
            y=y_train,
            random_seed=random_seed
        )

        # Create synthetic SMEs
        sme_datasets = generator.create_synthetic_smes(
            J=J,
            n_per_sme=n_per_sme,
            noise_scale=noise_scale,
            stratify=stratify,
            verbose=True
        )

        # Get summary statistics
        summary = generator.get_summary_statistics()

        print(f"\nSME Dataset Summary:")
        print(f"  Total SMEs: {len(sme_datasets)}")
        print(f"  Customers per SME: {n_per_sme}")
        print(f"  Total customers: {J * n_per_sme}")
        print(f"  Mean churn rate: {summary['churn_rate'].mean():.3f}")
        print(f"  Churn rate std: {summary['churn_rate'].std():.3f}")
        print(f"  Churn rate range: [{summary['churn_rate'].min():.3f}, " +
              f"{summary['churn_rate'].max():.3f}]")

    except Exception as e:
        print(f"\n✗ ERROR generating SME datasets: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 4/4] Saving datasets...")
    print(f"{'=' * 80}")

    try:
        # Save datasets
        generator.save_sme_datasets(
            output_dir=str(output_dir),
            verbose=True
        )

        # Save summary statistics
        summary_path = output_dir / "summary_statistics.csv"
        summary.to_csv(summary_path, index=False)
        print(f"✓ Summary statistics saved: {summary_path.name}")

        # Load and verify
        print(f"\nVerifying saved datasets...")
        loaded_datasets, metadata = SMEDataGenerator.load_sme_datasets(
            input_dir=str(output_dir),
            verbose=False
        )

        # Quick verification
        assert len(loaded_datasets) == J, "Dataset count mismatch"
        assert loaded_datasets[0]['X'].shape[1] == X_train.shape[1], "Feature count mismatch"
        print(f"✓ Verification passed")

    except Exception as e:
        print(f"\n✗ ERROR saving datasets: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"✓ Successfully created {J} SME datasets")
    print(f"\nGenerated Files:")
    print(f"  Directory: {output_dir}")
    print(f"  Datasets: sme_0.csv through sme_{J-1}.csv")
    print(f"  Metadata: metadata.json")
    print(f"  Summary: summary_statistics.csv")
    print(f"\nDataset Statistics:")
    print(f"  Total customers: {J * n_per_sme}")
    print(f"  Features per customer: {X_train.shape[1]}")
    print(f"  Mean churn rate: {summary['churn_rate'].mean():.3f}")
    print(f"  Churn rate range: [{summary['churn_rate'].min():.3f}, " +
          f"{summary['churn_rate'].max():.3f}]")

    if args.test_mode:
        print(f"   python scripts/train_hierarchical_model.py --quick-test")
    else:
        print(f"   python scripts/train_hierarchical_model.py")
    print(f"\n3. After training, analyze shrinkage:")
    print(f"   python scripts/analyze_shrinkage.py")

    print(f"\n{'=' * 80}")
    print(f"End time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
