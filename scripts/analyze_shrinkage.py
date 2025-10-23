"""
Analyze Shrinkage Behavior in Hierarchical Bayesian Model

This script compares MLE vs hierarchical posterior estimates, calculates
shrinkage weights, and generates Table 4.9 and Figure 4.3.

Usage:
    python scripts/analyze_shrinkage.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer2_bayesian import SMEDataGenerator, HierarchicalBayesianModel
from src.utils.plotting import SmallMLPlotStyle


def main():
    """Main execution function."""
    print(f"{'=' * 80}")
    print("SHRINKAGE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Paths
    project_root = Path(__file__).parent.parent
    trace_path = project_root / "models" / "hierarchical" / "trace.nc"
    posterior_path = project_root / "models" / "hierarchical" / "posterior_means.json"
    sme_dir = project_root / "data" / "sme_datasets"
    priors_path = project_root / "models" / "transfer_learning" / "priors.pkl"
    output_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 80}")
    print("[Step 1/6] Validating prerequisites...")
    print(f"{'=' * 80}")

    # Check trace exists
    if not trace_path.exists():
        print(f"✗ ERROR: Trace file not found: {trace_path}")
        print(f"  Please run training first:")
        print(f"  python scripts/train_hierarchical_model.py")
        sys.exit(1)
    print(f"✓ Found: trace.nc")

    # Check posterior means
    if not posterior_path.exists():
        print(f"✗ ERROR: Posterior means not found: {posterior_path}")
        sys.exit(1)
    print(f"✓ Found: posterior_means.json")

    # Check SME datasets
    if not (sme_dir / "metadata.json").exists():
        print(f"✗ ERROR: SME datasets not found")
        sys.exit(1)
    print(f"✓ Found: SME datasets")

    # Check priors
    if not priors_path.exists():
        print(f"✗ ERROR: Priors not found")
        sys.exit(1)
    print(f"✓ Found: priors.pkl")

    print(f"\n{'=' * 80}")
    print("[Step 2/6] Loading data and results...")
    print(f"{'=' * 80}")

    try:
        # Load posterior means
        with open(posterior_path, 'r') as f:
            posterior = json.load(f)

        mu_industry_mean = np.array(posterior['mu_industry_mean'])
        sigma_industry_mean = posterior['sigma_industry_mean']
        beta_j_mean = np.array(posterior['beta_j_mean'])
        feature_names = posterior['feature_names']
        J = posterior['J']
        p = posterior['p']

        print(f"✓ Loaded posterior means")
        print(f"  SMEs (J): {J}")
        print(f"  Features (p): {p}")
        print(f"  μ_industry: {mu_industry_mean.shape}")
        print(f"  β_j: {beta_j_mean.shape}")
        print(f"  σ_industry: {sigma_industry_mean:.4f}")

        # Load SME datasets
        sme_datasets, metadata = SMEDataGenerator.load_sme_datasets(
            input_dir=str(sme_dir),
            verbose=False
        )
        print(f"✓ Loaded SME datasets ({J} SMEs)")

        # Load priors for reference
        with open(priors_path, 'rb') as f:
            priors = pickle.load(f)
        beta_0 = priors['beta_0']
        Sigma_0 = priors['Sigma_0']
        print(f"✓ Loaded priors")

    except Exception as e:
        print(f"\n✗ ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 3/6] Computing MLE estimates...")
    print(f"{'=' * 80}")

    try:
        # Initialize model to use MLE computation method
        model = HierarchicalBayesianModel(
            beta_0=beta_0,
            Sigma_0=Sigma_0,
            random_seed=42
        )
        model.sme_datasets_ = sme_datasets
        model.J_ = J

        # Compute MLE (independent logistic regressions)
        beta_j_mle = model.compute_mle_estimates(verbose=True)

        print(f"\n✓ MLE estimates computed")
        print(f"  Shape: {beta_j_mle.shape}")
        print(f"  Range: [{beta_j_mle.min():.4f}, {beta_j_mle.max():.4f}]")

    except Exception as e:
        print(f"\n✗ ERROR computing MLE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 4/6] Calculating shrinkage weights...")
    print(f"{'=' * 80}")

    try:
        # Calculate shrinkage weights: λ_j = (β̂_j^hier - μ) / (β̂_j^MLE - μ)
        # Average across all features for each SME
        shrinkage_weights = []

        for j in range(J):
            lambda_j_features = []
            for k in range(p):
                denom = beta_j_mle[j, k] - mu_industry_mean[k]
                if abs(denom) > 0.01:  # Avoid division by near-zero
                    numer = beta_j_mean[j, k] - mu_industry_mean[k]
                    lambda_jk = numer / denom
                    lambda_j_features.append(lambda_jk)
                else:
                    lambda_j_features.append(1.0)  # No shrinkage if MLE ≈ population

            # Average shrinkage weight for SME j
            shrinkage_weights.append(np.mean(lambda_j_features))

        shrinkage_weights = np.array(shrinkage_weights)

        print(f"✓ Shrinkage weights calculated")
        print(f"  Mean: {shrinkage_weights.mean():.3f}")
        print(f"  Std: {shrinkage_weights.std():.3f}")
        print(f"  Range: [{shrinkage_weights.min():.3f}, {shrinkage_weights.max():.3f}]")
        print(f"\nInterpretation:")
        print(f"  λ ≈ 1.0: No shrinkage (estimates stay at MLE)")
        print(f"  λ ≈ 0.5: Partial pooling (halfway between MLE and population)")
        print(f"  λ ≈ 0.0: Complete pooling (shrunk to population mean)")

    except Exception as e:
        print(f"\n✗ ERROR calculating shrinkage: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 5/6] Generating Table 4.9 (Shrinkage Example)...")
    print(f"{'=' * 80}")

    try:
        # Select one representative feature (e.g., first feature with high importance)
        # For simplicity, use feature 0
        feature_idx = 0
        feature_name = feature_names[feature_idx]

        # Create table for first 5 SMEs
        table_data = []
        for j in range(min(5, J)):
            n_j = len(sme_datasets[j]['X'])
            row = {
                'SME': f'SME_{j}',
                'Sample Size (n_j)': n_j,
                'MLE Estimate': beta_j_mle[j, feature_idx],
                'Population Mean': mu_industry_mean[feature_idx],
                'Shrinkage Weight (λ_j)': shrinkage_weights[j],
                'Posterior Mean': beta_j_mean[j, feature_idx]
            }
            table_data.append(row)

        table_4_9 = pd.DataFrame(table_data)

        # Format for display
        table_4_9_display = table_4_9.copy()
        table_4_9_display['MLE Estimate'] = table_4_9_display['MLE Estimate'].apply(lambda x: f"{x:.4f}")
        table_4_9_display['Population Mean'] = table_4_9_display['Population Mean'].apply(lambda x: f"{x:.4f}")
        table_4_9_display['Shrinkage Weight (λ_j)'] = table_4_9_display['Shrinkage Weight (λ_j)'].apply(lambda x: f"{x:.3f}")
        table_4_9_display['Posterior Mean'] = table_4_9_display['Posterior Mean'].apply(lambda x: f"{x:.4f}")

        print(f"\nTable 4.9: Shrinkage Example for Feature '{feature_name}'")
        print(table_4_9_display.to_string(index=False))

        # Save CSV
        csv_path = output_dir / "table_4_9.csv"
        table_4_9.to_csv(csv_path, index=False)
        print(f"\n✓ Table 4.9 saved: {csv_path}")

        # Save Markdown
        md_path = output_dir / "table_4_9.md"
        with open(md_path, 'w') as f:
            f.write("# Table 4.9: Shrinkage Example\n\n")
            f.write(f"**Feature:** {feature_name}\n\n")
            f.write(table_4_9_display.to_markdown(index=False))
            f.write("\n\n**Interpretation:**\n")
            f.write("- **MLE Estimate:** Independent logistic regression (no pooling)\n")
            f.write("- **Population Mean:** Industry-wide mean from all SMEs\n")
            f.write("- **Shrinkage Weight (λ_j):** Degree of pooling (0=complete, 1=none)\n")
            f.write("- **Posterior Mean:** Hierarchical estimate (partial pooling)\n")
            f.write("\nSmaller SMEs experience more shrinkage toward the population mean.\n")
        print(f"✓ Table 4.9 saved: {md_path}")

    except Exception as e:
        print(f"\n✗ ERROR generating Table 4.9: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("[Step 6/6] Creating Figure 4.3 (Shrinkage Plot)...")
    print(f"{'=' * 80}")

    try:
        # Apply SmallML plot style
        SmallMLPlotStyle.apply()

        # Create shrinkage plot: MLE vs Hierarchical for selected feature
        fig, ax = SmallMLPlotStyle.create_figure(figsize=SmallMLPlotStyle.FIGSIZE_MAIN)

        # Scatter plot
        ax.scatter(
            beta_j_mle[:, feature_idx],
            beta_j_mean[:, feature_idx],
            s=100,
            alpha=0.7,
            color=SmallMLPlotStyle.COLORS['primary'],
            edgecolors='black',
            linewidth=1.5,
            zorder=3
        )

        # No shrinkage line (y = x)
        mle_min = beta_j_mle[:, feature_idx].min()
        mle_max = beta_j_mle[:, feature_idx].max()
        margin = (mle_max - mle_min) * 0.1
        ax.plot(
            [mle_min - margin, mle_max + margin],
            [mle_min - margin, mle_max + margin],
            'k--',
            linewidth=2,
            label='No shrinkage (y=x)',
            zorder=1
        )

        # Complete pooling line (y = population mean)
        ax.axhline(
            mu_industry_mean[feature_idx],
            color=SmallMLPlotStyle.COLORS['population'],
            linestyle='--',
            linewidth=2,
            label=f'Complete pooling (μ={mu_industry_mean[feature_idx]:.3f})',
            zorder=2
        )

        # Format axis
        SmallMLPlotStyle.format_axis(
            ax,
            title=f'Figure 4.3: Shrinkage Plot for Feature "{feature_name}"',
            xlabel='MLE Estimate (Independent Model)',
            ylabel='Posterior Mean (Hierarchical Model)',
            grid=True,
            legend=True
        )

        # Add annotations
        ax.text(
            0.05, 0.95,
            f'Mean λ = {shrinkage_weights.mean():.3f}\n' +
            f'σ_industry = {sigma_industry_mean:.3f}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # Save figure
        figure_path = figures_dir / "figure_4_3_shrinkage.png"
        SmallMLPlotStyle.save_figure(fig, str(figure_path))
        plt.close()

        # Also create a shrinkage by SME plot
        fig2, ax2 = SmallMLPlotStyle.create_figure(figsize=(10, 6))

        sme_ids = np.arange(J)
        ax2.bar(
            sme_ids,
            shrinkage_weights,
            color=SmallMLPlotStyle.COLORS['primary'],
            alpha=0.7,
            edgecolor='black',
            linewidth=1.5
        )
        ax2.axhline(
            shrinkage_weights.mean(),
            color=SmallMLPlotStyle.COLORS['danger'],
            linestyle='--',
            linewidth=2,
            label=f'Mean λ = {shrinkage_weights.mean():.3f}'
        )

        SmallMLPlotStyle.format_axis(
            ax2,
            title='Shrinkage Weights by SME',
            xlabel='SME Index',
            ylabel='Shrinkage Weight (λ)',
            grid=True,
            legend=True
        )

        ax2.set_xticks(sme_ids)
        ax2.set_xticklabels([f'SME_{j}' for j in sme_ids], rotation=45)

        figure2_path = figures_dir / "day12_shrinkage_by_sme.png"
        SmallMLPlotStyle.save_figure(fig2, str(figure2_path))
        plt.close()

        print(f"✓ Figure 4.3 saved: {figure_path}")
        print(f"✓ Shrinkage by SME plot saved: {figure2_path}")

    except Exception as e:
        print(f"\n✗ ERROR creating figures: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"✓ Shrinkage analysis complete!")
    print(f"\nShrinkage Statistics:")
    print(f"  Mean shrinkage weight: {shrinkage_weights.mean():.3f}")
    print(f"  Std shrinkage weight: {shrinkage_weights.std():.3f}")
    print(f"  Range: [{shrinkage_weights.min():.3f}, {shrinkage_weights.max():.3f}]")
    print(f"  Between-SME std (σ_industry): {sigma_industry_mean:.4f}")
    print(f"\nGenerated Outputs:")
    print(f"  Tables:")
    print(f"    - {output_dir / 'table_4_9.csv'}")
    print(f"    - {output_dir / 'table_4_9.md'}")
    print(f"  Figures:")
    print(f"    - {figures_dir / 'figure_4_3_shrinkage.png'}")
    print(f"    - {figures_dir / 'day12_shrinkage_by_sme.png'}")
    print(f"\nKey Finding:")
    if 0.3 <= shrinkage_weights.mean() <= 0.7:
        print(f"  ✓ Partial pooling working as expected")
        print(f"    (mean λ in [0.3, 0.7] indicates balanced shrinkage)")
    elif shrinkage_weights.mean() < 0.3:
        print(f"  ⚠ Strong shrinkage (mean λ < 0.3)")
        print(f"    SMEs heavily pooled toward population mean")
    else:
        print(f"  ⚠ Weak shrinkage (mean λ > 0.7)")
        print(f"    SMEs close to independent MLEs")

    print(f"\n{'=' * 80}")
    print(f"End time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
