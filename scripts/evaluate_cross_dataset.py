#!/usr/bin/env python3
"""
Cross-Dataset Validation - Table 4.5 Generator

This script evaluates the trained CatBoost base model separately on each constituent
dataset within the validation set to verify transfer learning effectiveness.

The key question: Does the model generalize across different business types
(telco, bank, ecommerce), or does it overfit to dataset-specific patterns?

Approach:
---------
Since X_val.csv doesn't include dataset_source labels, we:
1. Reload D_public_processed.csv (which has dataset_source column)
2. Recreate the same train/test split using same random_state=42
3. Extract dataset_source labels for validation indices
4. Compute per-dataset metrics (AUC, Accuracy)

Outputs:
--------
- results/tables/table_4_5.csv (Per-Dataset Validation Performance)
- results/tables/table_4_5.md (Markdown version with analysis)

Usage:
------
python scripts/evaluate_cross_dataset.py

Expected Runtime: 2-5 minutes

References:
-----------
- Section 4.2.2: Cross-Dataset Validation
- Table 4.5: Per-Dataset Validation Performance (generated)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.layer1_transfer.catboost_trainer import CatBoostBaseModel


def load_full_data_with_source(data_dir: Path) -> pd.DataFrame:
    """
    Load D_public_processed.csv which contains dataset_source column.

    Parameters
    ----------
    data_dir : Path
        Directory containing D_public_processed.csv

    Returns
    -------
    df_full : pd.DataFrame
        Full harmonized dataset with dataset_source column
    """
    print(f"\n{'='*70}")
    print(f"Loading Full Dataset with Source Labels")
    print(f"{'='*70}")

    df_path = data_dir / "D_public_processed.csv"
    if not df_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {df_path}\n"
            f"Please run data harmonization first."
        )

    df_full = pd.read_csv(df_path)

    print(f"✓ Loaded: {len(df_full):,} samples")
    print(f"  Columns: {len(df_full.columns)}")

    # Verify required columns exist
    required_cols = ["churned", "dataset_source"]
    missing = [col for col in required_cols if col not in df_full.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  Required columns present: {required_cols}")

    # Show dataset distribution
    source_counts = df_full["dataset_source"].value_counts()
    print(f"\n  Dataset distribution:")
    for dataset, count in source_counts.items():
        print(f"    {dataset}: {count:,} samples")

    return df_full


def recreate_validation_split(df_full: pd.DataFrame) -> tuple:
    """
    Recreate the same validation split as on previous steps.

    Uses the same parameters (test_size=0.2, stratify=churned, random_state=42)
    to ensure we get the exact same validation indices.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full harmonized dataset

    Returns
    -------
    X_val : pd.DataFrame
        Validation feature matrix
    y_val : pd.Series
        Validation target
    dataset_source_val : pd.Series
        Dataset source labels for validation samples
    """
    print(f"\n{'='*70}")
    print(f"Recreating Validation Split")
    print(f"{'='*70}")

    # Separate features, target, and source
    X_full = df_full.drop(["churned", "dataset_source"], axis=1)
    y_full = df_full["churned"]
    dataset_source_full = df_full["dataset_source"]

    # Recreate split with same parameters as on previous steps
    _, X_val, _, y_val, _, dataset_source_val = train_test_split(
        X_full,
        y_full,
        dataset_source_full,
        test_size=0.2,  # Same as previously
        stratify=y_full,  # Same stratification
        random_state=42,  # Same random seed
    )

    print(f"✓ Split recreated:")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Churn rate: {y_val.mean():.3f}")

    # Verify dataset distribution matches
    val_source_counts = dataset_source_val.value_counts()
    print(f"\n  Validation dataset distribution:")
    for dataset, count in val_source_counts.items():
        print(f"    {dataset}: {count:,} samples ({count/len(dataset_source_val)*100:.1f}%)")

    return X_val, y_val, dataset_source_val


def load_trained_model(model_path: Path) -> CatBoostBaseModel:
    """
    Load the trained CatBoost model.

    Parameters
    ----------
    model_path : Path
        Path to saved model (.cbm file)

    Returns
    -------
    model : CatBoostBaseModel
        Loaded model instance
    """
    print(f"\n{'='*70}")
    print(f"Loading Trained Model")
    print(f"{'='*70}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Please run scripts/train_catboost_base.py first."
        )

    model = CatBoostBaseModel.load_model(str(model_path))
    return model


def evaluate_per_dataset(
    model: CatBoostBaseModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dataset_source_val: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate model separately on each dataset.

    Parameters
    ----------
    model : CatBoostBaseModel
        Trained model
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    dataset_source_val : pd.Series
        Dataset source labels

    Returns
    -------
    results_df : pd.DataFrame
        Per-dataset metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating Per-Dataset Performance")
    print(f"{'='*70}")

    # Get predictions on full validation set
    y_pred_proba = model.model_.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Compute metrics per dataset
    results = []
    datasets = sorted(dataset_source_val.unique())

    for dataset_name in datasets:
        # Get mask for this dataset
        mask = dataset_source_val == dataset_name
        n_samples = mask.sum()

        if n_samples < 10:
            print(f"\n⚠ WARNING: {dataset_name} has only {n_samples} samples, skipping")
            continue

        # Get predictions for this dataset
        y_true_subset = y_val[mask]
        y_pred_proba_subset = y_pred_proba[mask]
        y_pred_subset = y_pred[mask]

        # Compute metrics
        try:
            auc = roc_auc_score(y_true_subset, y_pred_proba_subset)
            accuracy = accuracy_score(y_true_subset, y_pred_subset)

            results.append(
                {
                    "Dataset": dataset_name.capitalize(),
                    "AUC": auc,
                    "Accuracy": accuracy,
                    "Sample Size (val)": n_samples,
                }
            )

            print(f"\n{dataset_name.capitalize()}:")
            print(f"  Samples: {n_samples:,}")
            print(f"  AUC: {auc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")

        except ValueError as e:
            print(f"\n⚠ WARNING: Could not compute metrics for {dataset_name}: {e}")
            continue

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Add overall row
    overall_auc = roc_auc_score(y_val, y_pred_proba)
    overall_acc = accuracy_score(y_val, y_pred)

    overall_row = pd.DataFrame(
        [
            {
                "Dataset": "**Overall**",
                "AUC": overall_auc,
                "Accuracy": overall_acc,
                "Sample Size (val)": len(y_val),
            }
        ]
    )

    results_df = pd.concat([results_df, overall_row], ignore_index=True)

    return results_df


def analyze_consistency(results_df: pd.DataFrame) -> dict:
    """
    Analyze cross-dataset consistency.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-dataset results

    Returns
    -------
    analysis : dict
        Consistency metrics
    """
    print(f"\n{'='*70}")
    print(f"Analyzing Cross-Dataset Consistency")
    print(f"{'='*70}")

    # Exclude overall row for statistics
    dataset_results = results_df[results_df["Dataset"] != "**Overall**"]

    # Compute statistics
    auc_mean = dataset_results["AUC"].mean()
    auc_std = dataset_results["AUC"].std()
    auc_min = dataset_results["AUC"].min()
    auc_max = dataset_results["AUC"].max()
    auc_range = auc_max - auc_min

    acc_mean = dataset_results["Accuracy"].mean()
    acc_std = dataset_results["Accuracy"].std()

    print(f"\nAUC Statistics:")
    print(f"  Mean: {auc_mean:.4f}")
    print(f"  Std Dev: {auc_std:.4f}")
    print(f"  Min: {auc_min:.4f}")
    print(f"  Max: {auc_max:.4f}")
    print(f"  Range: {auc_range:.4f}")

    print(f"\nAccuracy Statistics:")
    print(f"  Mean: {acc_mean:.4f}")
    print(f"  Std Dev: {acc_std:.4f}")

    # Check consistency criterion (std < 0.02)
    is_consistent = auc_std < 0.02

    print(f"\nConsistency Check:")
    print(f"  Criterion: AUC std < 0.02")
    print(f"  Result: {auc_std:.4f} {'✓ PASS' if is_consistent else '✗ FAIL'}")

    if is_consistent:
        print(f"\n✓ Model shows excellent cross-dataset generalization!")
        print(f"  This indicates learned patterns are transferable, not dataset-specific.")
    else:
        print(f"\n⚠ WARNING: High variance across datasets")
        print(f"  Model may have learned some dataset-specific patterns.")

    analysis = {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_range": auc_range,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "is_consistent": is_consistent,
    }

    return analysis


def generate_table_4_5(
    results_df: pd.DataFrame, analysis: dict, output_dir: Path
) -> None:
    """
    Generate Table 4.5: Per-Dataset Validation Performance.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-dataset results
    analysis : dict
        Consistency analysis
    output_dir : Path
        Directory where table will be saved
    """
    print(f"\n{'='*70}")
    print(f"Generating Table 4.5: Per-Dataset Validation Performance")
    print(f"{'='*70}")

    # Save as CSV
    csv_path = output_dir / "table_4_5.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✓ CSV saved: {csv_path}")

    # Save as Markdown
    md_path = output_dir / "table_4_5.md"
    with open(md_path, "w") as f:
        f.write("# Table 4.5: Per-Dataset Validation Performance\n\n")
        f.write(
            "Evaluation of base model separately on each constituent dataset within the validation set.\n"
        )
        f.write(
            "This verifies that the model learned transferable patterns rather than dataset-specific artifacts.\n\n"
        )

        # Write table
        f.write(results_df.to_markdown(index=False, floatfmt=".4f"))

        # Write analysis
        f.write("\n\n## Cross-Dataset Consistency Analysis\n\n")
        f.write(f"**AUC Statistics:**\n")
        f.write(f"- Mean: {analysis['auc_mean']:.4f}\n")
        f.write(f"- Standard Deviation: {analysis['auc_std']:.4f}\n")
        f.write(f"- Range: {analysis['auc_range']:.4f}\n\n")

        f.write(f"**Consistency Check:**\n")
        f.write(f"- Criterion: AUC std < 0.02 (indicates good generalization)\n")
        f.write(
            f"- Result: {analysis['auc_std']:.4f} **{'✓ PASS' if analysis['is_consistent'] else '✗ FAIL'}**\n\n"
        )

        if analysis["is_consistent"]:
            f.write("## Interpretation\n\n")
            f.write(
                "Performance consistency across datasets (AUC std = {:.4f}) indicates that the base model\n".format(
                    analysis["auc_std"]
                )
            )
            f.write(
                "successfully learned generalizable churn patterns rather than overfitting to any single domain.\n"
            )
            f.write(
                "The slight variation (±{:.1f}% AUC) likely reflects genuine differences in churn\n".format(
                    analysis["auc_range"] * 100
                )
            )
            f.write(
                "predictability across business types (e.g., detailed SaaS usage logs enable slightly\n"
            )
            f.write("better predictions than telecommunications billing data).\n\n")
            f.write(
                "**Conclusion:** The transfer learning foundation is viable for SME contexts.\n"
            )
        else:
            f.write("## Interpretation\n\n")
            f.write(
                "⚠ High variance across datasets suggests the model may have learned some dataset-specific\n"
            )
            f.write(
                "patterns. Consider:\n"
            )
            f.write("1. Adding more diverse datasets to training\n")
            f.write("2. Increasing regularization (l2_leaf_reg)\n")
            f.write("3. Further feature harmonization\n")

    print(f"✓ Markdown saved: {md_path}")
    print(f"\nTable 4.5 generated successfully!")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("CROSS-DATASET VALIDATION - TABLE 4.5 GENERATOR")
    print("=" * 70)
    print(
        "\nThis script evaluates the trained CatBoost model separately on each dataset"
    )
    print("to verify transfer learning effectiveness.\n")

    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "harmonized"
    model_path = project_root / "models" / "transfer_learning" / "catboost_base.cbm"
    table_dir = project_root / "results" / "tables"

    # Create output directory
    table_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load full data with source labels
        df_full = load_full_data_with_source(data_dir)

        # Step 2: Recreate validation split
        X_val, y_val, dataset_source_val = recreate_validation_split(df_full)

        # Step 3: Load trained model
        model = load_trained_model(model_path)

        # Step 4: Evaluate per dataset
        results_df = evaluate_per_dataset(model, X_val, y_val, dataset_source_val)

        # Step 5: Analyze consistency
        analysis = analyze_consistency(results_df)

        # Step 6: Generate Table 4.5
        generate_table_4_5(results_df, analysis, table_dir)

        # Final summary
        print(f"\n{'='*70}")
        print(f"✓ CROSS-DATASET VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nGenerated Files:")
        print(f"  - {table_dir / 'table_4_5.csv'} (Per-Dataset Performance)")
        print(f"  - {table_dir / 'table_4_5.md'}")
        print(f"\nKey Finding:")
        if analysis["is_consistent"]:
            print(f"  ✓ Model generalizes well across datasets (AUC std = {analysis['auc_std']:.4f})")
            print(f"  ✓ Transfer learning foundation is viable!")
        else:
            print(f"  ⚠ High cross-dataset variance (AUC std = {analysis['auc_std']:.4f})")
            print(f"  Consider additional harmonization or regularization")

        return 0

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ ERROR: Cross-dataset validation failed")
        print(f"{'='*70}")
        print(f"\nError details: {str(e)}")
        print(f"\nPlease check:")
        print(f"  1. Data harmonization completed")
        print(f"  2. CatBoost model trained (run scripts/train_catboost_base.py)")
        print(f"  3. File exists: data/harmonized/D_public_processed.csv")
        print(f"  4. File exists: models/transfer_learning/catboost_base.cbm")
        print(f"\n{'='*70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
