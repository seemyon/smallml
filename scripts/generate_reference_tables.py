"""
Generate Reference Tables for SmallML White Paper

This script creates Tables 4.2, 4.8, 4.15, and 4.18 which are reference/configuration
tables that document the framework's hyperparameters, convergence criteria, software
stack, and tuning guidance.

Tables Generated:
- Table 4.2: CatBoost Hyperparameters (Section 4.2.2)
- Table 4.8: MCMC Convergence Criteria (Section 4.3.2)
- Table 4.15: Software Stack & Dependencies (Section 4.5)
- Table 4.18: Hyperparameter Tuning Guidance (Section 4.5)

Usage:
    python scripts/generate_reference_tables.py

Author: SmallML Framework
Date: October 2025
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List


def ensure_directories():
    """Create output directories if they don't exist."""
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    print("✓ Output directories ready")


def generate_table_4_2() -> pd.DataFrame:
    """
    Generate Table 4.2: Base Model Hyperparameters

    Documents the CatBoost configuration used in Layer 1 (Transfer Learning).
    These hyperparameters were chosen to balance model capacity with overfitting
    prevention on the public dataset (~120K observations).

    Returns:
        pd.DataFrame: Table with parameter names, values, and justifications
    """
    data = {
        "Parameter": [
            "Number of iterations (M)",
            "Learning rate (η)",
            "Tree depth (d)",
            "Min samples per leaf (s)",
            "L2 regularization (λ)",
            "Subsample ratio",
            "Feature subsample ratio",
            "Early stopping rounds",
            "Loss function"
        ],
        "Value": [
            "1000",
            "0.03",
            "6",
            "20",
            "3.0",
            "0.8",
            "0.8",
            "50",
            "Logloss"
        ],
        "Justification": [
            "Sufficient capacity with early stopping",
            "Conservative rate preventing overfitting",
            "Balances model complexity and generalization",
            "Prevents overfitting to rare patterns",
            "Ridge penalty on leaf weights",
            "Random 80% sample per iteration (bagging)",
            "Random 80% features per tree (diversity)",
            "Stop if validation loss doesn't improve",
            "Standard for binary classification"
        ]
    }

    df = pd.DataFrame(data)
    print(f"✓ Generated Table 4.2: {len(df)} hyperparameters documented")
    return df


def generate_table_4_8() -> pd.DataFrame:
    """
    Generate Table 4.8: Target Convergence Criteria

    Documents the MCMC convergence diagnostics used to validate that the
    hierarchical Bayesian model (Layer 2) has properly converged. These
    criteria must be met before using posterior samples for inference.

    Returns:
        pd.DataFrame: Table with diagnostic names, thresholds, and interpretations
    """
    data = {
        "Diagnostic": [
            "R̂ (Gelman-Rubin)",
            "ESS (Effective Sample Size)",
            "Acceptance Rate",
            "Divergences"
        ],
        "Threshold": [
            "< 1.01",
            "> 400",
            "0.80 - 0.95",
            "0"
        ],
        "Interpretation": [
            "Between-chain agreement (convergence)",
            "Sufficient independent samples",
            "Efficient MCMC exploration",
            "No numerical instabilities"
        ]
    }

    df = pd.DataFrame(data)
    print(f"✓ Generated Table 4.8: {len(df)} convergence diagnostics")
    return df


def generate_table_4_15() -> pd.DataFrame:
    """
    Generate Table 4.15: Software Stack

    Documents all required Python libraries and their versions for the
    SmallML framework. This table enables reproducibility and provides
    installation guidance for practitioners.

    Returns:
        pd.DataFrame: Table with library names, versions, and purposes
    """
    data = {
        "Component": [
            "Bayesian Inference",
            "Transfer Learning",
            "Feature Importance",
            "Conformal Prediction",
            "Data Processing",
            "Numerical Computing",
            "Statistical Tools",
            "Visualization",
            "Machine Learning Utils"
        ],
        "Library": [
            "PyMC",
            "CatBoost",
            "SHAP",
            "MAPIE",
            "pandas",
            "NumPy",
            "SciPy",
            "matplotlib",
            "scikit-learn"
        ],
        "Version": [
            "≥5.0",
            "≥1.2",
            "≥0.42",
            "≥0.6",
            "≥2.0",
            "≥1.24",
            "≥1.10",
            "≥3.7",
            "≥1.3"
        ],
        "Purpose": [
            "Hierarchical model specification, MCMC sampling",
            "Gradient boosting base model pre-training",
            "Shapley value computation for prior extraction",
            "Calibration and prediction set construction",
            "Data manipulation, harmonization",
            "Array operations, linear algebra",
            "Probability distributions, statistical tests",
            "Plotting shrinkage diagrams, convergence diagnostics",
            "Preprocessing, train/test splits, metrics"
        ]
    }

    df = pd.DataFrame(data)
    print(f"✓ Generated Table 4.15: {len(df)} dependencies documented")
    return df


def generate_table_4_18() -> pd.DataFrame:
    """
    Generate Table 4.18: Hyperparameter Defaults and Tuning Guidance

    Provides practical guidance for practitioners on hyperparameter selection
    and tuning. Includes default values, reasonable ranges, and signals that
    indicate when tuning is needed.

    Returns:
        pd.DataFrame: Table with parameter names, defaults, ranges, and tuning signals
    """
    data = {
        "Parameter": [
            "τ (industry variance prior)",
            "MCMC warmup iterations",
            "MCMC sampling iterations",
            "Number of MCMC chains",
            "Target acceptance rate",
            "Conformal α",
            "Prior scaling λ"
        ],
        "Default": [
            "2.0",
            "1000",
            "2000",
            "4",
            "0.90",
            "0.10",
            "1.0"
        ],
        "Range": [
            "[1.0, 5.0]",
            "[500, 2000]",
            "[1000, 5000]",
            "[2, 8]",
            "[0.80, 0.95]",
            "[0.05, 0.20]",
            "[0.5, 2.0]"
        ],
        "Tuning Signal": [
            "If R̂ > 1.05, increase τ",
            "If R̂ > 1.01, increase warmup",
            "If ESS < 400, increase sampling",
            "Use 4 for standard, 8 for critical applications",
            "Higher = slower but more accurate",
            "Business risk tolerance (lower = more conservative)",
            "If prior-only AUC << 0.65, increase scaling"
        ]
    }

    df = pd.DataFrame(data)
    print(f"✓ Generated Table 4.18: {len(df)} hyperparameters with tuning guidance")
    return df


def save_table(df: pd.DataFrame, table_num: str, title: str):
    """
    Save table in both CSV and Markdown formats.

    Args:
        df: DataFrame containing the table data
        table_num: Table number (e.g., "4_2")
        title: Descriptive title for the table
    """
    # Save as CSV
    csv_path = f"results/tables/table_{table_num}.csv"
    df.to_csv(csv_path, index=False)

    # Save as Markdown with title
    md_path = f"results/tables/table_{table_num}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Table {table_num.replace('_', '.')}: {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"*Generated by SmallML Framework*\n")
        f.write(f"*Total rows: {len(df)}*\n")

    print(f"  → Saved to {csv_path}")
    print(f"  → Saved to {md_path}")


def generate_metadata() -> Dict:
    """
    Generate metadata about the reference tables.

    Returns:
        dict: Metadata including generation info and table descriptions
    """
    metadata = {
        "generated_by": "SmallML Framework",
        "description": "Reference tables documenting hyperparameters, convergence criteria, and software stack",
        "tables": {
            "table_4_2": {
                "title": "Base Model Hyperparameters",
                "section": "4.2.2",
                "type": "configuration",
                "rows": 9
            },
            "table_4_8": {
                "title": "Target Convergence Criteria",
                "section": "4.3.2",
                "type": "diagnostics",
                "rows": 4
            },
            "table_4_15": {
                "title": "Software Stack",
                "section": "4.5",
                "type": "dependencies",
                "rows": 9
            },
            "table_4_18": {
                "title": "Hyperparameter Defaults and Tuning Guidance",
                "section": "4.5",
                "type": "tuning_guide",
                "rows": 7
            }
        },
        "total_tables": 4,
        "total_parameters_documented": 29
    }

    return metadata


def main():
    """Main execution function."""
    print("=" * 70)
    print("SmallML Framework - Reference Tables Generation")
    print("=" * 70)
    print()

    # Ensure output directories exist
    ensure_directories()
    print()

    # Generate each table
    print("Generating reference tables...")
    print("-" * 70)

    # Table 4.2: CatBoost Hyperparameters
    table_4_2 = generate_table_4_2()
    save_table(table_4_2, "4_2", "Base Model Hyperparameters")
    print()

    # Table 4.8: Convergence Criteria
    table_4_8 = generate_table_4_8()
    save_table(table_4_8, "4_8", "Target Convergence Criteria")
    print()

    # Table 4.15: Software Stack
    table_4_15 = generate_table_4_15()
    save_table(table_4_15, "4_15", "Software Stack")
    print()

    # Table 4.18: Hyperparameter Tuning
    table_4_18 = generate_table_4_18()
    save_table(table_4_18, "4_18", "Hyperparameter Defaults and Tuning Guidance")
    print()

    # Generate and save metadata
    print("-" * 70)
    metadata = generate_metadata()
    metadata_path = "results/tables/reference_tables_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, indent=2, fp=f)
    print(f"✓ Saved metadata to {metadata_path}")
    print()

    # Summary
    print("=" * 70)
    print("✅ REFERENCE TABLES GENERATION COMPLETE")
    print("=" * 70)
    print()
    print("Generated Tables:")
    print("  • Table 4.2  - Base Model Hyperparameters (9 parameters)")
    print("  • Table 4.8  - Target Convergence Criteria (4 diagnostics)")
    print("  • Table 4.15 - Software Stack (9 dependencies)")
    print("  • Table 4.18 - Hyperparameter Tuning Guidance (7 parameters)")
    print()
    print(f"Total Parameters Documented: {metadata['total_parameters_documented']}")
    print()
    print("Files Created:")
    print("  • 4 CSV files (results/tables/table_4_*.csv)")
    print("  • 4 Markdown files (results/tables/table_4_*.md)")
    print("  • 1 Metadata file (reference_tables_metadata.json)")
    print("=" * 70)


if __name__ == "__main__":
    main()
