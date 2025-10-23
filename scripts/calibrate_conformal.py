"""
Calibrate Conformal Predictor (Algorithm 4.4)

This script splits SME datasets into training/calibration sets, loads the
hierarchical Bayesian model, generates predictions on the
calibration set, and computes the conformal prediction threshold q̂.

Usage:
    python scripts/calibrate_conformal.py [--pooled] [--conservative LAMBDA]

Options:
    --pooled            Use pooled calibration across all SMEs (default)
    --conservative      Conservative adjustment factor (0.0-0.3, default 0.0)

Generates:
    - models/conformal/calibration_threshold.pkl
    - models/conformal/calibration_scores.csv
    - models/conformal/calibration_diagnostics.json
    - results/tables/table_4_10.csv (Bayesian vs Conformal)
    - results/tables/table_4_11.csv (Calibration strategies)
    - results/tables/table_4_12.csv (Quality diagnostics)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import arviz as az
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer3_conformal import ConformalPredictor
from src.layer2_bayesian import SMEDataGenerator, HierarchicalBayesianModel


def load_hierarchical_trace(trace_path: Path) -> az.InferenceData:
    """Load MCMC trace."""
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Hierarchical model trace not found: {trace_path}\n"
            f"Please run first: python scripts/train_hierarchical_model.py"
        )

    print(f"Loading hierarchical model trace from {trace_path}...")
    trace = az.from_netcdf(trace_path)
    print(f"✓ Loaded trace with {trace.posterior.dims['draw']} draws × "
          f"{trace.posterior.dims['chain']} chains")

    return trace


def generate_bayesian_predictions(
    X: np.ndarray,
    trace: az.InferenceData,
    sme_id: int
) -> np.ndarray:
    """
    Generate posterior predictive probabilities from hierarchical model.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Feature matrix
    trace : az.InferenceData
        MCMC trace
    sme_id : int
        SME identifier (0-indexed)

    Returns
    -------
    predictions : np.ndarray, shape (n,)
        Predicted probabilities P̂(Y=1|X)
    """
    # Handle missing values by filling with column means
    X_clean = X.copy()
    if np.isnan(X_clean).any():
        # Fill NaN with column means
        col_means = np.nanmean(X_clean, axis=0)
        for i in range(X_clean.shape[1]):
            mask = np.isnan(X_clean[:, i])
            if mask.any():
                X_clean[mask, i] = col_means[i]

    # Extract posterior samples for beta_j
    beta_j_samples = trace.posterior['beta_j'].values  # (chains, draws, J, p)

    # Flatten chains and draws
    beta_j_samples = beta_j_samples.reshape(-1, beta_j_samples.shape[2], beta_j_samples.shape[3])
    # Shape: (n_samples, J, p)

    # Get samples for this SME
    beta_samples = beta_j_samples[:, sme_id, :]  # (n_samples, p)

    # Compute logit: X @ beta^T
    logits = X_clean @ beta_samples.T  # (n, n_samples)

    # Convert to probabilities
    probs = 1 / (1 + np.exp(-logits))  # (n, n_samples)

    # Average over posterior samples
    predictions = probs.mean(axis=1)  # (n,)

    return predictions


def create_table_4_10() -> pd.DataFrame:
    """Create Table 4.10: Bayesian vs Conformal Comparison."""
    data = {
        'Aspect': [
            'Assumptions',
            'Validity',
            'Interpretation',
            'Computational Cost',
            'Information Content',
            'SME Accessibility'
        ],
        'Bayesian Credible Intervals': [
            'Model correctness, MCMC convergence',
            'Depends on model',
            'Probability given model',
            'High (MCMC sampling)',
            'Rich (full posterior)',
            'Moderate (requires Bayes knowledge)'
        ],
        'Conformal Prediction Sets': [
            'Exchangeability only',
            'Guaranteed finite-sample',
            'Frequentist coverage',
            'Low (quantile computation)',
            'Minimal (set membership)',
            'High (intuitive frequency interpretation)'
        ]
    }

    return pd.DataFrame(data)


def create_table_4_11() -> pd.DataFrame:
    """Create Table 4.11: Recommended Calibration Strategy."""
    data = {
        'SMEs (J)': ['≥10', '5-10', '<5', '<5'],
        'Customers/SME (n_j)': ['≥100', '50-100', '≥100', '<100'],
        'Total Samples': ['≥1000', '250-1000', '<500', '<500'],
        'Strategy': [
            'Pooled',
            'Pooled',
            'Cross-conformal',
            'Cross-conformal + Conservative'
        ],
        'Expected Coverage': ['89-91%', '88-92%', '87-93%', '90-95%']
    }

    return pd.DataFrame(data)


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Calibrate conformal predictor"
    )
    parser.add_argument(
        '--pooled',
        action='store_true',
        default=True,
        help='Use pooled calibration across SMEs (default: True)'
    )
    parser.add_argument(
        '--conservative',
        type=float,
        default=0.0,
        help='Conservative adjustment factor λ (0.0-0.3, default: 0.0)'
    )
    args = parser.parse_args()

    print(f"{'='*80}")
    print("CONFORMAL PREDICTION CALIBRATION")
    print(f"{'='*80}")
    print(f"Strategy: {'Pooled' if args.pooled else 'Per-SME'}")
    print(f"Conservative adjustment: λ = {args.conservative:.2f}")
    print(f"Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = datetime.now()

    # Configuration
    alpha = 0.10  # 90% coverage target
    calibration_fraction = 0.25  # 25% for calibration
    random_seed = 42

    np.random.seed(random_seed)

    # Paths
    project_root = Path(__file__).parent.parent
    sme_dir = project_root / "data" / "sme_datasets"
    trace_path = project_root / "models" / "hierarchical" / "trace.nc"
    output_dir = project_root / "models" / "conformal"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = project_root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print("[Step 1/5] Loading prerequisites...")
    print(f"{'='*80}\n")

    # Load SME datasets
    print("Loading SME datasets...")
    sme_datasets, metadata = SMEDataGenerator.load_sme_datasets(sme_dir)
    J = len(sme_datasets)
    print(f"✓ Loaded {J} SME datasets")

    # Load hierarchical trace
    trace = load_hierarchical_trace(trace_path)

    print(f"\n{'='*80}")
    print("[Step 2/5] Splitting data into train/calibration...")
    print(f"{'='*80}\n")

    # Split each SME's data
    sme_train = {}
    sme_calibration = {}

    for sme_id in range(J):
        X = sme_datasets[sme_id]['X'].values
        y = sme_datasets[sme_id]['y'].values
        n = len(y)

        # Calculate split point
        n_cal = int(n * calibration_fraction)
        n_train = n - n_cal

        # Random shuffle for split
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:]

        sme_train[sme_id] = {
            'X': X[train_idx],
            'y': y[train_idx]
        }
        sme_calibration[sme_id] = {
            'X': X[cal_idx],
            'y': y[cal_idx]
        }

        print(f"SME {sme_id}: {n_train} train, {n_cal} calibration")

    total_cal = sum(len(sme_calibration[j]['y']) for j in range(J))
    print(f"\nTotal calibration samples: {total_cal}")

    print(f"\n{'='*80}")
    print("[Step 3/5] Generating Bayesian predictions on calibration set...")
    print(f"{'='*80}\n")

    # Generate predictions for calibration data
    if args.pooled:
        print("Using POOLED calibration strategy\n")

        y_cal_dict = {}
        pred_cal_dict = {}

        for sme_id in range(J):
            print(f"Generating predictions for SME {sme_id}...")
            X_cal = sme_calibration[sme_id]['X']
            y_cal = sme_calibration[sme_id]['y']

            # Generate predictions
            predictions = generate_bayesian_predictions(X_cal, trace, sme_id)

            y_cal_dict[sme_id] = y_cal
            pred_cal_dict[sme_id] = predictions

            print(f"  Mean prediction: {predictions.mean():.3f}")
            print(f"  Prediction range: [{predictions.min():.3f}, "
                  f"{predictions.max():.3f}]")

        # Initialize predictor
        cp = ConformalPredictor(
            alpha=alpha,
            conservative_adjustment=args.conservative,
            random_seed=random_seed
        )

        # Pooled calibration
        q_hat = cp.pooled_calibration(
            y_cal_dict,
            pred_cal_dict,
            verbose=True
        )

    else:
        print("Using PER-SME calibration strategy\n")
        # For simplicity, we'll just calibrate on SME 0 in per-SME mode
        # In practice, you'd calibrate separately for each SME
        sme_id = 0
        X_cal = sme_calibration[sme_id]['X']
        y_cal = sme_calibration[sme_id]['y']

        predictions = generate_bayesian_predictions(X_cal, trace, sme_id)

        cp = ConformalPredictor(
            alpha=alpha,
            conservative_adjustment=args.conservative,
            random_seed=random_seed
        )

        q_hat = cp.calibrate(y_cal, predictions, verbose=True)

    print(f"\n{'='*80}")
    print("[Step 4/5] Saving calibration results...")
    print(f"{'='*80}\n")

    # Save calibrated predictor
    cp.save_calibration(output_dir / "calibration_threshold.pkl")

    # Save calibration scores
    scores_df = pd.DataFrame({
        'score': cp.calibration_scores_
    })
    scores_df.to_csv(output_dir / "calibration_scores.csv", index=False)
    print(f"✓ Saved calibration scores to calibration_scores.csv")

    # Save diagnostics
    diagnostics = {
        'alpha': alpha,
        'q_hat': float(q_hat),
        'n_calibration': int(cp.n_cal_),
        'empirical_coverage_cal': float(cp.empirical_coverage_),
        'conservative_adjustment': args.conservative,
        'strategy': 'pooled' if args.pooled else 'per_sme',
        'metadata': cp.calibration_metadata_
    }

    with open(output_dir / "calibration_diagnostics.json", 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"✓ Saved diagnostics to calibration_diagnostics.json")

    print(f"\n{'='*80}")
    print("[Step 5/5] Generating tables...")
    print(f"{'='*80}\n")

    # Table 4.10: Bayesian vs Conformal
    table_4_10 = create_table_4_10()
    table_4_10.to_csv(tables_dir / "table_4_10.csv", index=False)
    print("✓ Generated Table 4.10: Bayesian vs Conformal Comparison")
    print(table_4_10.to_string(index=False))
    print()

    # Table 4.11: Calibration strategies
    table_4_11 = create_table_4_11()
    table_4_11.to_csv(tables_dir / "table_4_11.csv", index=False)
    print("✓ Generated Table 4.11: Calibration Strategy by Data Availability")
    print(table_4_11.to_string(index=False))
    print()

    # Table 4.12: Quality diagnostics
    table_4_12 = pd.DataFrame({
        'Metric': [
            'Empirical coverage',
            'Average set size',
            'Fraction empty sets',
            'Fraction {0,1} sets'
        ],
        'Target': [
            '[0.87, 0.93] for α=0.10',
            '1.2-1.5',
            '<5%',
            '10-30%'
        ],
        'Interpretation': [
            'Close to nominal 90%',
            'Mostly singletons, some uncertain',
            'Rare empty sets acceptable',
            'Reasonable uncertainty admission'
        ]
    })
    table_4_12.to_csv(tables_dir / "table_4_12.csv", index=False)
    print("✓ Generated Table 4.12: Calibration Quality Diagnostics")
    print(table_4_12.to_string(index=False))
    print()

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"{'='*80}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*80}")
    print(f"Calibrated threshold: q̂ = {q_hat:.4f}")
    print(f"Calibration samples: {cp.n_cal_}")
    print(f"Empirical coverage (calibration): {cp.empirical_coverage_:.3f}")
    print(f"Target coverage: {1-alpha:.3f}")
    print(f"\nElapsed time: {elapsed:.1f} seconds")
    print(f"End time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext step: python scripts/validate_coverage.py")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
