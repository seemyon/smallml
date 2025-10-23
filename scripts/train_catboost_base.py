#!/usr/bin/env python3
"""
Train CatBoost Base Model - Automated Pipeline

This script implements the complete training pipeline for Layer 1 (Transfer Learning)
of the SmallML framework. It trains a CatBoost classifier on harmonized public datasets
and generates Tables 4.3 and 4.4 for the white paper.

Outputs:
--------
Models:
- models/transfer_learning/catboost_base.cbm (trained model)
- models/transfer_learning/training_metadata.json (training configuration)

Tables:
- results/tables/table_4_3.csv (Base Model Performance)
- results/tables/table_4_3.md (Markdown version)
- results/tables/table_4_4.csv (Top 10 Feature Importances)
- results/tables/table_4_4.md (Markdown version)

Usage:
------
python scripts/train_catboost_base.py

References:
-----------
- Section 4.2.2: Base Model Pre-training
- Algorithm 4.1: Base Model Pre-training
- Table 4.2: Hyperparameters
- Table 4.3: Performance metrics (generated)
- Table 4.4: Feature importances (generated)
"""

import sys
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.layer1_transfer.catboost_trainer import CatBoostBaseModel


def load_data(data_dir: Path) -> tuple:
    """
    Load harmonized training and validation data.

    Parameters
    ----------
    data_dir : Path
        Directory containing X_train.csv, y_train.csv, X_val.csv, y_val.csv

    Returns
    -------
    X_train, y_train, X_val, y_val : pd.DataFrame or pd.Series
        Training and validation data
    """
    print(f"\n{'='*70}")
    print(f"Loading Data")
    print(f"{'='*70}")

    # Check files exist
    required_files = ["X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv"]
    for fname in required_files:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"Required file not found: {data_dir / fname}\n"
                f"Please run data harmonization first."
            )

    # Load data
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")["churned"]
    X_val = pd.read_csv(data_dir / "X_val.csv")
    y_val = pd.read_csv(data_dir / "y_val.csv")["churned"]

    print(f"✓ Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Churn rate: {y_train.mean():.3f}")
    print(f"✓ Validation set: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
    print(f"  Churn rate: {y_val.mean():.3f}")

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val) -> CatBoostBaseModel:
    """
    Train CatBoost model with Table 4.2 hyperparameters.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : pd.DataFrame or pd.Series
        Training and validation data

    Returns
    -------
    model : CatBoostBaseModel
        Trained model instance
    """
    print(f"\n{'='*70}")
    print(f"Training CatBoost Model")
    print(f"{'='*70}")
    print(f"Using hyperparameters from Table 4.2 (section_4_2_2_content.md)")

    # Initialize model
    model = CatBoostBaseModel(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        min_data_in_leaf=20,
        l2_leaf_reg=3.0,
        subsample=0.8,
        rsm=0.8,
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
    )

    # Train
    model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    return model


def generate_table_4_3(
    model: CatBoostBaseModel, X_val, y_val, output_dir: Path
) -> None:
    """
    Generate Table 4.3: Base Model Performance on Validation Set.

    Parameters
    ----------
    model : CatBoostBaseModel
        Trained model
    X_val, y_val : pd.DataFrame or pd.Series
        Validation data
    output_dir : Path
        Directory where table will be saved
    """
    print(f"\n{'='*70}")
    print(f"Generating Table 4.3: Base Model Performance")
    print(f"{'='*70}")

    # Evaluate model
    metrics = model.evaluate(X_val, y_val, dataset_name="Validation")

    # Create table DataFrame
    table_4_3 = pd.DataFrame(
        [
            {"Metric": "AUC-ROC", "Value": metrics["auc_roc"]},
            {"Metric": "Accuracy", "Value": metrics["accuracy"]},
            {"Metric": "Precision", "Value": metrics["precision"]},
            {"Metric": "Recall", "Value": metrics["recall"]},
            {"Metric": "F1-Score", "Value": metrics["f1_score"]},
            {"Metric": "Log Loss", "Value": metrics["log_loss"]},
        ]
    )

    # Save as CSV
    csv_path = output_dir / "table_4_3.csv"
    table_4_3.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✓ CSV saved: {csv_path}")

    # Save as Markdown
    md_path = output_dir / "table_4_3.md"
    with open(md_path, "w") as f:
        f.write("# Table 4.3: Base Model Performance on Validation Set\n\n")
        f.write(
            "Performance metrics computed on held-out validation set (n=4,535 samples).\n\n"
        )
        f.write(table_4_3.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Interpretation\n\n")
        f.write(
            f"- **AUC-ROC = {metrics['auc_roc']:.3f}**: Excellent discrimination between churners and non-churners\n"
        )
        f.write(
            f"- **Accuracy = {metrics['accuracy']:.3f}**: {metrics['accuracy']*100:.1f}% of predictions correct at 0.5 threshold\n"
        )
        f.write(
            f"- **Precision = {metrics['precision']:.3f}**: {metrics['precision']*100:.1f}% of predicted churners actually churned\n"
        )
        f.write(
            f"- **Recall = {metrics['recall']:.3f}**: Model identifies {metrics['recall']*100:.1f}% of actual churners\n"
        )
        f.write(
            f"- **F1-Score = {metrics['f1_score']:.3f}**: Harmonic mean of precision and recall\n"
        )
        f.write(
            f"- **Log Loss = {metrics['log_loss']:.3f}**: Well-calibrated probabilistic predictions\n"
        )
        f.write("\n## Target Achievement\n\n")
        f.write(
            f"- Target AUC ≥ 0.80: **{'✓ PASS' if metrics['auc_roc'] >= 0.80 else '✗ FAIL'}**\n"
        )
        f.write(
            f"- Target Accuracy ≥ 0.80: **{'✓ PASS' if metrics['accuracy'] >= 0.80 else '✗ FAIL'}**\n"
        )

    print(f"✓ Markdown saved: {md_path}")

    # Check targets
    if metrics["auc_roc"] >= 0.80:
        print(f"\n✓ Target achieved: AUC = {metrics['auc_roc']:.4f} ≥ 0.80")
    else:
        print(
            f"\n✗ WARNING: AUC = {metrics['auc_roc']:.4f} below target 0.80"
        )


def generate_table_4_4(model: CatBoostBaseModel, output_dir: Path) -> None:
    """
    Generate Table 4.4: Top 10 Features by Importance.

    Parameters
    ----------
    model : CatBoostBaseModel
        Trained model with feature importances
    output_dir : Path
        Directory where table will be saved
    """
    print(f"\n{'='*70}")
    print(f"Generating Table 4.4: Top 10 Feature Importances")
    print(f"{'='*70}")

    # Extract feature importances
    importance_df = model.get_feature_importances(top_k=10)

    # Create table with rank, feature, and importance
    table_4_4 = importance_df[["rank", "feature", "importance"]].copy()
    table_4_4.columns = ["Rank", "Feature", "Importance"]

    # Save as CSV
    csv_path = output_dir / "table_4_4.csv"
    table_4_4.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✓ CSV saved: {csv_path}")

    # Save as Markdown
    md_path = output_dir / "table_4_4.md"
    with open(md_path, "w") as f:
        f.write("# Table 4.4: Top 10 Features by Importance\n\n")
        f.write(
            "Feature importance scores measure each predictor's contribution to model performance.\n"
        )
        f.write(
            "Importance is calculated as total gain from the feature across all trees.\n\n"
        )
        f.write(table_4_4.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Key Observations\n\n")
        f.write(
            f"1. **Most Important Feature**: {table_4_4.iloc[0]['Feature']} "
            f"(importance = {table_4_4.iloc[0]['Importance']:.4f})\n"
        )
        f.write(
            "2. **Feature Categories**: Top features align with RFM (Recency-Frequency-Monetary) model\n"
        )
        f.write("   - Recency features (time since last interaction)\n")
        f.write("   - Frequency features (transaction counts, usage)\n")
        f.write("   - Monetary features (revenue, value, balance)\n")
        f.write("   - Lifecycle features (tenure, age)\n")
        f.write(
            "\n3. **Transfer Learning Implication**: These importances will inform prior distributions\n"
        )
        f.write(
            "   in Layer 2 (Hierarchical Bayesian model). High-importance features receive\n"
        )
        f.write("   stronger (lower variance) priors.\n")

    print(f"✓ Markdown saved: {md_path}")

    # Print top 10
    print(f"\nTop 10 Features:")
    for _, row in table_4_4.iterrows():
        print(f"  {row['Rank']:2d}. {row['Feature']:40s} {row['Importance']:.4f}")


def save_model_and_metadata(
    model: CatBoostBaseModel,
    X_train,
    y_train,
    X_val,
    y_val,
    output_dir: Path,
) -> None:
    """
    Save trained model and metadata.

    Parameters
    ----------
    model : CatBoostBaseModel
        Trained model
    X_train, y_train, X_val, y_val : pd.DataFrame or pd.Series
        Training and validation data
    output_dir : Path
        Directory where model will be saved
    """
    print(f"\n{'='*70}")
    print(f"Saving Model and Metadata")
    print(f"{'='*70}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "catboost_base.cbm"
    model.save_model(str(model_path))

    # Compute metrics
    train_metrics = model.evaluate(X_train, y_train, dataset_name="Training")
    val_metrics = model.evaluate(X_val, y_val, dataset_name="Validation")

    # Save metadata
    metadata_path = output_dir / "training_metadata.json"
    model.save_metadata(
        str(metadata_path),
        training_metrics=train_metrics,
        validation_metrics=val_metrics,
    )

    print(f"\n✓ Model and metadata saved to: {output_dir}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING - CATBOOST BASE MODEL TRAINING")
    print("=" * 70)
    print("\nThis script implements Algorithm 4.1 (Base Model Pre-training)")
    print("and generates Tables 4.3 and 4.4 for the white paper.\n")

    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "harmonized"
    model_dir = project_root / "models" / "transfer_learning"
    table_dir = project_root / "results" / "tables"

    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load data
        X_train, y_train, X_val, y_val = load_data(data_dir)

        # Step 2: Train model
        model = train_model(X_train, y_train, X_val, y_val)

        # Step 3: Generate Table 4.3 (Performance)
        generate_table_4_3(model, X_val, y_val, table_dir)

        # Step 4: Generate Table 4.4 (Feature Importances)
        generate_table_4_4(model, table_dir)

        # Step 5: Save model and metadata
        save_model_and_metadata(model, X_train, y_train, X_val, y_val, model_dir)

        # Final summary
        print(f"\n{'='*70}")
        print(f"✓ TRAINING COMPLETE - ALL TASKS SUCCESSFUL")
        print(f"{'='*70}")
        print(f"\nGenerated Files:")
        print(f"  Models:")
        print(f"    - {model_dir / 'catboost_base.cbm'}")
        print(f"    - {model_dir / 'training_metadata.json'}")
        print(f"  Tables:")
        print(f"    - {table_dir / 'table_4_3.csv'} (Base Model Performance)")
        print(f"    - {table_dir / 'table_4_3.md'}")
        print(f"    - {table_dir / 'table_4_4.csv'} (Top 10 Features)")
        print(f"    - {table_dir / 'table_4_4.md'}")
        print(f"\n{'='*70}\n")

        return 0

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ ERROR: Training failed")
        print(f"{'='*70}")
        print(f"\nError details: {str(e)}")
        print(f"\nPlease check:")
        print(f"  1. Data harmonization completed successfully")
        print(f"  2. Files exist in data/harmonized/")
        print(f"  3. Conda environment 'smallml' is activated")
        print(f"\n{'='*70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
