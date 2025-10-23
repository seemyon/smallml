"""
Train Hierarchical Bayesian Model via MCMC (NUTS)

This script implements Algorithm 4.3 from Section 4.3.2, fitting a hierarchical
logistic regression model to multiple SME datasets using PyMC.

Usage:
    python scripts/train_hierarchical_model.py [--quick-test]

Options:
    --quick-test    Fast convergence check (2 chains, 500 samples)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer2_bayesian import SMEDataGenerator, HierarchicalBayesianModel


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train hierarchical Bayesian model"
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (2 chains, 500 samples, ~5-10 min)'
    )
    args = parser.parse_args()

    print(f"{'=' * 80}")
    print("HIERARCHICAL BAYESIAN MODEL TRAINING")
    print(f"{'=' * 80}")
    print(f"Mode: {'QUICK TEST' if args.quick_test else 'FULL RUN'}")
    print(f"Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = datetime.now()

    # Configuration
    if args.quick_test:
        chains = 2
        draws = 500
        tune = 500
        target_accept = 0.85
    else:
        chains = 4          # Algorithm 4.3: K = 4 chains
        draws = 4000        # Algorithm 4.3: N_samples = 2000
        tune = 2000         # Algorithm 4.3: N_warmup = 1000
        target_accept = 0.95  # Target acceptance rate

    tau = 1.0               # Hyperparameter for sigma_industry prior (tighter for stability)
    random_seed = 42

    # Paths
    project_root = Path(__file__).parent.parent
    priors_path = project_root / "models" / "transfer_learning" / "priors.pkl"
    sme_dir = project_root / "data" / "sme_datasets"
    output_dir = project_root / "models" / "hierarchical"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 80}")
    print("[Step 1/5] Validating prerequisites...")
    print(f"{'=' * 80}")

    # Check priors from previous steps
    if not priors_path.exists():
        print(f"✗ ERROR: Priors file not found: {priors_path}")
        print(f"  Please complete prior extraction first:")
        print(f"  python scripts/extract_priors.py")
        sys.exit(1)
    print(f"✓ Found: priors.pkl")

    # Check SME datasets
    metadata_path = sme_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"✗ ERROR: SME datasets not found: {sme_dir}")
        print(f"  Please create SME datasets first:")
        print(f"  python scripts/create_sme_datasets.py")
        sys.exit(1)
    print(f"✓ Found: SME datasets")

    print(f"\n{'=' * 80}")
    print("[Step 2/5] Loading priors and SME datasets...")
    print(f"{'=' * 80}")

    try:
        # Load priors from previous steps
        with open(priors_path, 'rb') as f:
            priors = pickle.load(f)

        beta_0 = priors['beta_0']
        Sigma_0 = priors['Sigma_0']
        feature_names = priors['feature_names']
        p = len(beta_0)

        print(f"✓ Loaded priors from previous steps")
        print(f"  Features (p): {p}")
        print(f"  Prior mean range: [{beta_0.min():.4f}, {beta_0.max():.4f}]")
        print(f"  Prior std range: [{np.sqrt(np.diag(Sigma_0)).min():.4f}, " +
              f"{np.sqrt(np.diag(Sigma_0)).max():.4f}]")

        # Load SME datasets
        sme_datasets, metadata = SMEDataGenerator.load_sme_datasets(
            input_dir=str(sme_dir),
            verbose=True
        )

        J = metadata['n_smes']
        n_per_sme = metadata['n_per_sme']

        print(f"\n✓ Loaded SME datasets")
        print(f"  Number of SMEs (J): {J}")
        print(f"  Customers per SME (n): {n_per_sme}")
        print(f"  Total customers: {J * n_per_sme}")
        print(f"  Mean churn rate: {metadata['mean_churn_rate']:.3f}")

    except Exception as e:
        print(f"\n✗ ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 3/5] Specifying hierarchical model...")
    print(f"{'=' * 80}")
    print(f"Algorithm 4.3: Hierarchical Bayesian Inference via NUTS")
    print(f"\nModel Structure:")
    print(f"  Level 1 (Population):")
    print(f"    μ_industry ~ Normal(β₀, √Σ₀)")
    print(f"    σ_industry ~ HalfNormal({tau})")
    print(f"  Level 2 (SME-specific, non-centered):")
    print(f"    β_j_raw ~ Normal(0, 1), β_j = μ_industry + σ_industry * β_j_raw")
    print(f"    for j = 1,...,{J}")
    print(f"  Level 3 (Observations):")
    print(f"    y_ij ~ Bernoulli(logit⁻¹(β_j^T x_ij))")
    print(f"\nMCMC Configuration:")
    print(f"  Sampler: NUTS (No-U-Turn Sampler)")
    print(f"  Chains: {chains}")
    print(f"  Warmup iterations: {tune}")
    print(f"  Sampling iterations: {draws}")
    print(f"  Target acceptance: {target_accept}")
    print(f"  Total samples: {chains * draws}")
    print(f"  Random seed: {random_seed}")

    try:
        # Initialize model
        model = HierarchicalBayesianModel(
            beta_0=beta_0,
            Sigma_0=Sigma_0,
            tau=tau,
            random_seed=random_seed
        )

        print(f"\n✓ Model initialized")

    except Exception as e:
        print(f"\n✗ ERROR initializing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 4/5] Running MCMC sampling...")
    print(f"{'=' * 80}")

    if args.quick_test:
        print(f"⚠ QUICK TEST MODE")
        print(f"  This is for pipeline verification only")
        print(f"  For publication-quality results, run without --quick-test")
    else:
        print(f"⏰ ESTIMATED RUNTIME: 30-60 minutes")

    print(f"\nSampling will begin shortly...")
    print(f"PyMC will display progress bars for each chain\n")

    try:
        # Fit model via MCMC
        model.fit(
            sme_datasets=sme_datasets,
            chains=chains,
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            cores=None,  # Use all available cores
            verbose=True
        )

        elapsed = (datetime.now() - start_time).total_seconds() / 60

        print(f"\n✓ MCMC sampling completed")
        print(f"  Elapsed time: {elapsed:.1f} minutes")

    except Exception as e:
        print(f"\n✗ ERROR during MCMC sampling: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check memory usage (need ~2-4 GB available)")
        print(f"  2. Try --quick-test mode first")
        print(f"  3. Check PyMC installation: pip install pymc>=5.0")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 5/5] Checking convergence...")
    print(f"{'=' * 80}")
    print(f"Applying Table 4.8 convergence criteria:")
    print(f"  ✓ R̂ < 1.01 (Gelman-Rubin statistic)")
    print(f"  ✓ ESS > 400 (Effective sample size)")

    try:
        # Check convergence
        convergence = model.check_convergence(verbose=True)

        # Save convergence report
        convergence_path = output_dir / "convergence_diagnostics.json"
        with open(convergence_path, 'w') as f:
            json.dump(convergence, f, indent=2)
        print(f"\n✓ Convergence diagnostics saved: {convergence_path.name}")

        if not convergence['all_ok']:
            print(f"\n⚠ WARNING: Convergence criteria not fully met")
            print(f"  Consider re-running with more samples:")
            print(f"  - Increase draws to 4000")
            print(f"  - Increase tune to 2000")
            if not args.quick_test:
                print(f"\n  Continuing anyway...")
            else:
                print(f"\n  This is expected in quick test mode")
                print(f"  Run full mode for publication-quality results")

    except Exception as e:
        print(f"\n✗ ERROR checking convergence: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("Saving results...")
    print(f"{'=' * 80}")

    try:
        # Save trace
        trace_path = output_dir / "trace.nc"
        model.save_trace(str(trace_path), verbose=True)

        # Extract posterior means
        mu_mean, sigma_mean, beta_j_mean = model.extract_posterior_means()

        # Save posterior means
        posterior_summary = {
            'mu_industry_mean': mu_mean.tolist(),
            'sigma_industry_mean': float(sigma_mean),
            'beta_j_mean': beta_j_mean.tolist(),
            'feature_names': feature_names,
            'J': J,
            'p': p,
            'n_per_sme': n_per_sme
        }

        posterior_path = output_dir / "posterior_means.json"
        with open(posterior_path, 'w') as f:
            json.dump(posterior_summary, f, indent=2)
        print(f"✓ Posterior means saved: {posterior_path.name}")

        # Save model specification
        model_spec = {
            'algorithm': 'Algorithm 4.3: Hierarchical Bayesian Inference via NUTS',
            'section': '4.3.2',
            'mcmc_config': {
                'sampler': 'NUTS',
                'chains': chains,
                'draws': draws,
                'tune': tune,
                'target_accept': target_accept,
                'total_samples': chains * draws
            },
            'model_config': {
                'J': J,
                'p': p,
                'n_per_sme': n_per_sme,
                'tau': tau,
                'random_seed': random_seed
            },
            'data_sources': {
                'priors': 'SHAP extraction',
                'sme_datasets': 'Synthetic SMEs'
            },
            'convergence': convergence,
            'training_time_minutes': round(elapsed, 2),
            'timestamp': datetime.now().isoformat()
        }

        spec_path = output_dir / "model_specification.json"
        with open(spec_path, 'w') as f:
            json.dump(model_spec, f, indent=2)
        print(f"✓ Model specification saved: {spec_path.name}")

    except Exception as e:
        print(f"\n✗ ERROR saving results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"✓ Hierarchical Bayesian model training complete!")
    print(f"\nModel Statistics:")
    print(f"  SMEs (J): {J}")
    print(f"  Features (p): {p}")
    print(f"  Customers per SME: {n_per_sme}")
    print(f"  Total observations: {J * n_per_sme}")
    print(f"  Total parameters: {p + 1 + J * p}")
    print(f"\nMCMC Results:")
    print(f"  Chains: {chains}")
    print(f"  Samples per chain: {draws}")
    print(f"  Total posterior samples: {chains * draws}")
    print(f"  Training time: {elapsed:.1f} minutes")
    print(f"\nConvergence:")
    print(f"  R̂ max: {convergence['rhat_max']:.6f} (target: < 1.01)")
    print(f"  ESS min: {convergence['ess_min']:.0f} (target: > 400)")
    print(f"  Status: {'✓ PASS' if convergence['all_ok'] else '⚠ WARNING'}")
    print(f"\nPosterior Estimates:")
    print(f"  Population mean (μ_industry): [{mu_mean.min():.4f}, {mu_mean.max():.4f}]")
    print(f"  Between-SME std (σ_industry): {sigma_mean:.4f}")
    print(f"  SME-specific means (β_j): {beta_j_mean.shape}")
    print(f"\nGenerated Files:")
    print(f"  Directory: {output_dir}")
    print(f"  Trace: trace.nc (~{trace_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Posterior means: posterior_means.json")
    print(f"  Convergence: convergence_diagnostics.json")
    print(f"  Model spec: model_specification.json")

    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print(f"{'=' * 80}")
    print(f"1. Analyze shrinkage behavior:")
    print(f"   python scripts/analyze_shrinkage.py")
    print(f"\n2. Generate Table 4.9 and Figure 4.3")
    print(f"\n3. Explore results interactively:")
    print(f"   jupyter notebook notebooks/05_hierarchical_bayesian_model.ipynb")

    print(f"\n{'=' * 80}")
    print(f"End time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed: {elapsed:.1f} minutes")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
