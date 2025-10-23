"""
Automated Prior Extraction Pipeline

This script implements the complete prior extraction pipeline from Algorithm 4.2.
It loads the trained CatBoost model, computes SHAP values on validation
data, and extracts Bayesian priors (β₀, Σ₀) for the hierarchical model.

Outputs:
- models/transfer_learning/priors.pkl (β₀, Σ₀)
- models/transfer_learning/prior_extraction_metadata.json
- results/tables/table_4_6.csv + .md (Top 5 features with priors)
- results/tables/table_4_7.csv + .md (Prior predictive check)

Usage:
    python scripts/extract_priors.py

Requirements:
- Trained CatBoost model
- Harmonized data
- SHAP library installed
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src.layer1_transfer.shap_extractor import SHAPPriorExtractor

warnings.filterwarnings('ignore')


def load_data():
    """Load training and validation data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load training data
    X_train = pd.read_csv('data/harmonized/X_train.csv')
    y_train = pd.read_csv('data/harmonized/y_train.csv')['churned']
    print(f"✓ Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

    # Load validation data
    X_val = pd.read_csv('data/harmonized/X_val.csv')
    y_val = pd.read_csv('data/harmonized/y_val.csv')['churned']
    print(f"✓ Validation data: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")

    # Load full processed dataset to extract dataset_source labels
    D_public = pd.read_csv('data/harmonized/D_public_processed.csv')
    print(f"✓ Full processed dataset: {D_public.shape[0]:,} samples")

    # Recreate split to get dataset_source labels for validation set
    print("\nRecreating split to extract dataset_source labels...")
    X_full = D_public.drop(columns=['churned', 'dataset_source'])
    y_full = D_public['churned']
    dataset_source_full = D_public['dataset_source']

    # Use same random_state as previously
    _, _, _, _, _, dataset_source_val = train_test_split(
        X_full, y_full, dataset_source_full,
        test_size=0.2,
        stratify=y_full,
        random_state=42
    )

    # Verify split matches
    if len(dataset_source_val) != len(X_val):
        raise ValueError("Split mismatch! Check random_state and stratification.")

    print(f"✓ Dataset source labels extracted: {len(dataset_source_val):,} samples")
    print("\nDataset distribution in validation set:")
    for dataset, count in dataset_source_val.value_counts().items():
        print(f"  {dataset}: {count:,} samples ({count/len(dataset_source_val)*100:.1f}%)")

    return X_train, y_train, X_val, y_val, dataset_source_val


def load_model():
    """Load trained CatBoost model."""
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model_path = 'models/transfer_learning/catboost_base.cbm'

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Please run training first: python scripts/train_catboost_base.py"
        )

    model = CatBoostClassifier()
    model.load_model(model_path)

    print(f"✓ Model loaded: {model.tree_count_} trees")

    # Load metadata
    metadata_path = 'models/transfer_learning/training_metadata.json'
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Validation AUC: {metadata['validation_metrics']['auc_roc']:.4f}")

    return model


def extract_priors_pipeline(model, X_train, X_val, y_val, dataset_source_val):
    """Run complete prior extraction pipeline."""
    print("\n" + "=" * 80)
    print("PRIOR EXTRACTION PIPELINE (Algorithm 4.2)")
    print("=" * 80)

    # Initialize extractor
    extractor = SHAPPriorExtractor(
        model=model,
        X_train=X_train,
        lambda_scale=1.0,  # Table 4.18 default
        random_seed=42
    )

    print(f"\n✓ Extractor initialized")
    print(f"  Features: {len(extractor.feature_names_)}")
    print(f"  Scaling factor: λ = {extractor.lambda_scale}")

    # Extract priors (runs full Algorithm 4.2)
    print("\n" + "-" * 80)
    print("Step 1: Computing SHAP values...")
    print("-" * 80)

    beta_0, Sigma_0 = extractor.extract_priors(
        X_val,
        dataset_source_val,
        verbose=True
    )

    # Prior predictive check
    print("\n" + "-" * 80)
    print("Step 2: Prior Predictive Check (Table 4.7)")
    print("-" * 80)

    results = extractor.prior_predictive_check(
        X_val,
        y_val,
        n_samples=100,
        verbose=True
    )

    return extractor, beta_0, Sigma_0, results


def generate_tables(extractor, results):
    """Generate Tables 4.6 and 4.7."""
    print("\n" + "=" * 80)
    print("GENERATING TABLES")
    print("=" * 80)

    # Load feature importances from Table 4.4
    table_4_4_path = 'results/tables/table_4_4.csv'
    if not Path(table_4_4_path).exists():
        raise FileNotFoundError(
            f"Table 4.4 not found: {table_4_4_path}\n"
            "Please run training first."
        )

    table_4_4 = pd.read_csv(table_4_4_path)

    # Generate Table 4.6
    print("\nGenerating Table 4.6: Extracted Prior Distributions...")
    table_4_6 = extractor.generate_table_4_6(
        top_n=5,
        feature_importances=table_4_4
    )

    # Save Table 4.6
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    table_4_6.to_csv('results/tables/table_4_6.csv', index=False)

    # Save as Markdown
    with open('results/tables/table_4_6.md', 'w') as f:
        f.write("# Table 4.6: Extracted Prior Distributions for Top 5 Features\n\n")
        f.write(table_4_6.to_markdown(index=False))
        f.write("\n\n*Generated from SHAP values using Algorithm 4.2*\n")
        f.write(f"\n*Extraction date: {datetime.now().strftime('%Y-%m-%d')}*\n")

    print("✓ Table 4.6 saved:")
    print("  - results/tables/table_4_6.csv")
    print("  - results/tables/table_4_6.md")
    print("\nTable 4.6 Preview:")
    print(table_4_6.to_string(index=False))

    # Generate Table 4.7
    print("\n" + "-" * 80)
    print("Generating Table 4.7: Prior Predictive Performance...")

    table_4_7 = pd.DataFrame([
        {
            'Model': 'Random coefficients β ~ N(0, 1)',
            'AUC': results['random_coefficients'],
            'Interpretation': 'Barely better than chance'
        },
        {
            'Model': 'Prior-only β ~ N(β₀, Σ₀)',
            'AUC': results['prior_only'],
            'Interpretation': 'Substantial signal from transfer learning'
        },
        {
            'Model': 'Fully-trained CatBoost',
            'AUC': results['trained_catboost'],
            'Interpretation': 'Full model performance'
        }
    ])

    # Save Table 4.7
    table_4_7.to_csv('results/tables/table_4_7.csv', index=False)

    # Save as Markdown
    with open('results/tables/table_4_7.md', 'w') as f:
        f.write("# Table 4.7: Prior Predictive Performance on Validation Data\n\n")
        f.write(table_4_7.to_markdown(index=False))
        f.write("\n\n*Prior predictive check validates that extracted priors encode transferable knowledge.*\n")
        f.write("\n## Interpretation\n\n")
        f.write("- **Random coefficients**: Baseline performance (no information)\n")
        f.write("- **Prior-only**: Uses only transferred knowledge from public datasets\n")
        f.write("- **Trained CatBoost**: Full model with all data\n\n")
        f.write("A well-calibrated prior should:\n")
        f.write("1. Outperform random coefficients (AUC_prior >> AUC_random)\n")
        f.write("2. Underperform full model (AUC_prior < AUC_catboost)\n\n")
        f.write(f"*Extraction date: {datetime.now().strftime('%Y-%m-%d')}*\n")

    print("✓ Table 4.7 saved:")
    print("  - results/tables/table_4_7.csv")
    print("  - results/tables/table_4_7.md")
    print("\nTable 4.7 Preview:")
    print(table_4_7.to_string(index=False))

    return table_4_6, table_4_7


def save_priors(extractor):
    """Save priors and metadata."""
    print("\n" + "=" * 80)
    print("SAVING PRIORS")
    print("=" * 80)

    # Save priors pickle
    priors_path = 'models/transfer_learning/priors.pkl'
    Path(priors_path).parent.mkdir(parents=True, exist_ok=True)

    extractor.save_priors(priors_path, include_metadata=True)

    # Save metadata JSON
    metadata = {
        "extraction_timestamp": datetime.now().isoformat(),
        "algorithm": "Algorithm 4.2: Prior Distribution Extraction",
        "section": "Section 4.2.3: Prior Distribution Extraction",
        "lambda_scale": extractor.lambda_scale,
        "n_features": len(extractor.feature_names_),
        "prior_statistics": {
            "beta_0": {
                "shape": list(extractor.beta_0_.shape),
                "mean_abs": float(np.abs(extractor.beta_0_).mean()),
                "max_abs": float(np.abs(extractor.beta_0_).max()),
                "n_positive": int((extractor.beta_0_ > 0).sum()),
                "n_negative": int((extractor.beta_0_ < 0).sum()),
            },
            "Sigma_0": {
                "shape": list(extractor.Sigma_0_.shape),
                "mean_std": float(np.sqrt(np.diag(extractor.Sigma_0_)).mean()),
                "median_std": float(np.median(np.sqrt(np.diag(extractor.Sigma_0_)))),
                "max_std": float(np.sqrt(np.diag(extractor.Sigma_0_)).max()),
            }
        },
        "validation_metrics": extractor.prior_metadata_,
    }

    metadata_path = 'models/transfer_learning/prior_extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved: {metadata_path}")

    return priors_path, metadata_path


def print_summary(beta_0, Sigma_0, results, priors_path):
    """Print final summary."""
    print("\n" + "=" * 80)
    print("COMPLETE: SHAP-BASED PRIOR EXTRACTION")
    print("=" * 80)

    prior_stds = np.sqrt(np.diag(Sigma_0))

    print("\nPrior Means (β₀):")
    print(f"  Shape: {beta_0.shape}")
    print(f"  Mean |β₀|: {np.abs(beta_0).mean():.4f}")
    print(f"  Max |β₀|: {np.abs(beta_0).max():.4f}")
    print(f"  Positive coefficients: {(beta_0 > 0).sum()}")
    print(f"  Negative coefficients: {(beta_0 < 0).sum()}")

    print("\nPrior Covariance (Σ₀):")
    print(f"  Shape: {Sigma_0.shape}")
    print(f"  Mean σ₀: {prior_stds.mean():.4f}")
    print(f"  Median σ₀: {np.median(prior_stds):.4f}")
    print(f"  Max σ₀: {prior_stds.max():.4f}")

    print("\nPrior Predictive Check (Table 4.7):")
    print(f"  Random coefficients:  AUC = {results['random_coefficients']:.4f}")
    print(f"  Prior-only:           AUC = {results['prior_only']:.4f}")
    print(f"  Trained CatBoost:     AUC = {results['trained_catboost']:.4f}")
    print(f"  Prior improvement:    Δ = {results['prior_only'] - results['random_coefficients']:+.4f}")
    print(f"  Remaining gap:        Δ = {results['trained_catboost'] - results['prior_only']:+.4f}")

    # Validation checks
    print("\n" + "-" * 80)
    print("Validation Checks:")
    print("-" * 80)

    checks_passed = True

    # Check 1: Prior-only > Random
    if results['prior_only'] > results['random_coefficients'] + 0.05:
        print("  ✓ Priors encode transferable knowledge (prior >> random)")
    else:
        print("  ✗ WARNING: Weak priors (prior ≈ random)")
        checks_passed = False

    # Check 2: CatBoost > Prior-only
    if results['trained_catboost'] > results['prior_only'] + 0.05:
        print("  ✓ SME data essential (CatBoost >> prior)")
    else:
        print("  ⚠ Priors may be overconfident (CatBoost ≈ prior)")

    # Check 3: Prior means reasonable
    if np.abs(beta_0).mean() > 0.01 and np.abs(beta_0).mean() < 2.0:
        print("  ✓ Prior means in reasonable range")
    else:
        print("  ⚠ Prior means may be extreme")
        checks_passed = False

    # Check 4: Prior variances reasonable
    if prior_stds.mean() > 0.01 and prior_stds.mean() < 1.0:
        print("  ✓ Prior variances in reasonable range")
    else:
        print("  ⚠ Prior variances may be extreme")
        checks_passed = False

    print("\n" + "=" * 80)
    print("OUTPUTS GENERATED")
    print("=" * 80)
    print("\nTables:")
    print("  ✓ results/tables/table_4_6.csv + .md")
    print("  ✓ results/tables/table_4_7.csv + .md")
    print("\nPriors:")
    print(f"  ✓ {priors_path}")
    print("  ✓ models/transfer_learning/prior_extraction_metadata.json")

    if checks_passed:
        print("\n✓ All validation checks passed!")
        return 0
    else:
        print("\n⚠ Some validation checks failed. Review prior extraction.")
        return 1


def main():
    """Main pipeline."""
    try:
        start_time = datetime.now()

        print("\n" + "=" * 80)
        print("SHAP-BASED PRIOR EXTRACTION")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nThis script implements Algorithm 4.2 from Section 4.2.3")
        print("=" * 80)

        # Step 1: Load data
        X_train, y_train, X_val, y_val, dataset_source_val = load_data()

        # Step 2: Load model
        model = load_model()

        # Step 3: Extract priors
        extractor, beta_0, Sigma_0, results = extract_priors_pipeline(
            model, X_train, X_val, y_val, dataset_source_val
        )

        # Step 4: Generate tables
        table_4_6, table_4_7 = generate_tables(extractor, results)

        # Step 5: Save priors
        priors_path, metadata_path = save_priors(extractor)

        # Step 6: Print summary
        exit_code = print_summary(beta_0, Sigma_0, results, priors_path)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print("\n" + "=" * 80)
        print(f"Total runtime: {duration:.1f} minutes")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        return exit_code

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR")
        print("=" * 80)
        print(f"\n{type(e).__name__}: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Verify: ls models/transfer_learning/catboost_base.cbm")
        print("  2. Verify: ls data/harmonized/X_train.csv")
        print("  3. Check Python environment: conda activate smallml")
        print("  4. Verify SHAP installed: python -c 'import shap'")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
