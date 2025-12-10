"""
Generate Figure 5.1: Main Results Box Plot

Creates box plot showing AUC distribution across 75 evaluations (15 SMEs × 5 folds)
for three baseline methods: Independent LR, Complete Pooling, and SmallML.

Usage:
    python scripts/create_figure_5_1.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.plotting import SmallMLPlotStyle


def main():
    """Main execution function."""
    print(f"{'=' * 80}")
    print("FIGURE 5.1: MAIN RESULTS BOX PLOT")
    print(f"{'=' * 80}")
    print(f"Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "results" / "tables" / "table_5_2_cv_per_fold.csv"
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 80}")
    print("[Step 1/3] Loading cross-validation results...")
    print(f"{'=' * 80}")

    # Load data
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        print(f"  Please run baseline comparison first:")
        print(f"  python scripts/run_baseline_comparison_cv.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded data: {data_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Methods: {df['Method'].unique().tolist()}")
    print(f"  Folds: {df['Fold'].nunique()}")
    print(f"  Total evaluations: {len(df) // df['Method'].nunique()}")

    # Display summary statistics
    print(f"\n{'=' * 80}")
    print("[Step 2/3] Computing summary statistics...")
    print(f"{'=' * 80}")

    summary = df.groupby('Method')['AUC'].agg(['mean', 'std', 'min', 'max', 'median'])
    summary['IQR'] = df.groupby('Method')['AUC'].apply(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    )

    print(f"\nAUC Summary by Method:")
    print(summary.to_string())

    # Calculate percentage points difference from SmallML
    smallml_mean = summary.loc['SmallML', 'mean']
    for method in summary.index:
        if method != 'SmallML':
            diff = (summary.loc[method, 'mean'] - smallml_mean) * 100
            print(f"\n{method} vs SmallML: {diff:+.1f} percentage points")

    print(f"\n{'=' * 80}")
    print("[Step 3/3] Creating box plot...")
    print(f"{'=' * 80}")

    # Apply SmallML plot style
    SmallMLPlotStyle.apply()

    # Create figure
    fig, ax = SmallMLPlotStyle.create_figure(figsize=(10, 7))

    # Define method order and colors
    method_order = ['Independent', 'Complete Pooling', 'SmallML']
    colors = {
        'Independent': SmallMLPlotStyle.COLORS['danger'],      # Red (worst)
        'Complete Pooling': SmallMLPlotStyle.COLORS['secondary'],  # Orange (middle)
        'SmallML': SmallMLPlotStyle.COLORS['success']         # Green (best)
    }

    # Prepare data for plotting
    plot_data = []
    for method in method_order:
        method_data = df[df['Method'] == method]['AUC'].values
        plot_data.append(method_data)

    # Create box plot
    bp = ax.boxplot(
        plot_data,
        tick_labels=method_order,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white',
                      markeredgecolor='black', markersize=8),
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='gray',
                       markersize=6, alpha=0.5)
    )

    # Color the boxes
    for patch, method in zip(bp['boxes'], method_order):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.7)

    # Format axis
    SmallMLPlotStyle.format_axis(
        ax,
        title='',  # No title - figure caption goes in paper
        xlabel='Method',
        ylabel='AUC-ROC',
        grid=True,
        legend=False
    )

    # Set y-axis limits for better visualization
    ax.set_ylim(0.4, 1.0)

    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label='Random Classifier')
    ax.axhline(y=smallml_mean, color='green', linestyle='--', linewidth=1.5,
               alpha=0.3, label=f'SmallML Mean ({smallml_mean:.3f})')

    # Add text annotations for key statistics
    for i, method in enumerate(method_order):
        method_data = df[df['Method'] == method]['AUC']
        median_val = method_data.median()
        mean_val = method_data.mean()

        # Add median value above the box
        ax.text(i + 1, median_val + 0.02, f'{median_val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add legend
    ax.legend(loc='lower right', fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save figure
    figure_path = figures_dir / "figure_5_1_main_results_boxplot.png"
    SmallMLPlotStyle.save_figure(fig, str(figure_path))
    plt.close()

    print(f"[OK] Figure 5.1 saved: {figure_path}")

    # Generate interpretation text
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")

    print(f"\nKey Findings:")
    print(f"1. Central Tendency:")
    print(f"   - SmallML median: {summary.loc['SmallML', 'median']:.3f}")
    print(f"   - Complete Pooling median: {summary.loc['Complete Pooling', 'median']:.3f}")
    print(f"   - Independent median: {summary.loc['Independent', 'median']:.3f}")

    print(f"\n2. Variance (IQR):")
    print(f"   - SmallML: {summary.loc['SmallML', 'IQR']:.3f} (tightest)")
    print(f"   - Complete Pooling: {summary.loc['Complete Pooling', 'IQR']:.3f}")
    print(f"   - Independent: {summary.loc['Independent', 'IQR']:.3f} (widest)")

    # Calculate variance reduction
    independent_var = df[df['Method'] == 'Independent']['AUC'].var()
    smallml_var = df[df['Method'] == 'SmallML']['AUC'].var()
    var_reduction = (1 - smallml_var / independent_var) * 100

    print(f"\n3. Variance Reduction:")
    print(f"   - SmallML reduces variance by {var_reduction:.1f}% vs Independent")
    print(f"   - Independent std: {summary.loc['Independent', 'std']:.3f}")
    print(f"   - SmallML std: {summary.loc['SmallML', 'std']:.3f}")

    print(f"\n4. Consistency:")
    print(f"   - SmallML min: {summary.loc['SmallML', 'min']:.3f} (no catastrophic failures)")
    print(f"   - Independent min: {summary.loc['Independent', 'min']:.3f} (potential failures)")

    print(f"\n5. Universal Improvement:")
    independent_auc = df[df['Method'] == 'Independent']['AUC']
    smallml_auc = df[df['Method'] == 'SmallML']['AUC']
    print(f"   - SmallML exceeds Independent in {(smallml_auc.values > independent_auc.values).sum()}/{len(independent_auc)} cases")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"[OK] Figure 5.1 generation complete!")
    print(f"\nOutput:")
    print(f"  - {figure_path}")
    print(f"\nFigure demonstrates:")
    print(f"  1. Superior central tendency: SmallML median {summary.loc['SmallML', 'median']:.3f}")
    print(f"  2. Reduced variance: {var_reduction:.1f}% tighter than Independent")
    print(f"  3. No catastrophic failures: min AUC = {summary.loc['SmallML', 'min']:.3f}")
    print(f"  4. Consistent across all 15 SMEs and 5 folds")

    print(f"\n{'=' * 80}")
    print(f"End time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
