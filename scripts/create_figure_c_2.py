"""
Generate Figure C.2: Multi-α Calibration Curves

Creates two-panel figure showing conformal prediction calibration across
multiple alpha values (α ∈ {0.05, 0.10, 0.15, 0.20}).

Data source: White Paper Table 5.4 / Investigation findings

Usage:
    python scripts/create_figure_c_2.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.plotting import SmallMLPlotStyle


def main():
    """Main execution function."""
    print(f"{'=' * 80}")
    print("FIGURE C.2: MULTI-ALPHA CALIBRATION CURVES")
    print(f"{'=' * 80}\n")

    # Apply SmallML plot style
    SmallMLPlotStyle.apply()

    # Data from investigation report / White Paper Table 5.4
    # α values: miscoverage rates
    alpha_values = np.array([0.05, 0.10, 0.15, 0.20])

    # Target coverage levels (1 - α)
    target_coverage = (1 - alpha_values) * 100

    # Empirical coverage from experiments
    empirical_coverage = np.array([96.2, 92.0, 86.8, 82.4])

    # Singleton set percentages
    singleton_rates = np.array([88.7, 94.3, 96.8, 98.1])

    print("Data loaded:")
    print(f"  Alpha values: {alpha_values}")
    print(f"  Target coverage: {target_coverage}%")
    print(f"  Empirical coverage: {empirical_coverage}%")
    print(f"  Singleton rates: {singleton_rates}%\n")

    # Create figure with 2 panels
    fig, axes = SmallMLPlotStyle.create_subplots(
        nrows=1, ncols=2,
        figsize=(12, 5)
    )

    # ========== Panel A: Coverage Calibration ==========
    ax1 = axes[0]

    # Plot diagonal (perfect calibration)
    coverage_range = np.array([75, 100])
    ax1.plot(coverage_range, coverage_range,
             'k--', linewidth=2, alpha=0.5,
             label='Perfect calibration', zorder=1)

    # Plot empirical vs target
    ax1.plot(target_coverage, empirical_coverage,
             'o-', color=SmallMLPlotStyle.COLORS['primary'],
             markersize=10, linewidth=2.5,
             label='SmallML (observed)', zorder=3)

    # Add error region (±3% tolerance)
    ax1.fill_between(coverage_range,
                     coverage_range - 3,
                     coverage_range + 3,
                     alpha=0.1, color='gray',
                     label='±3% tolerance', zorder=2)

    # Annotate each point
    for i, alpha in enumerate(alpha_values):
        ax1.annotate(f'α={alpha:.2f}',
                    xy=(target_coverage[i], empirical_coverage[i]),
                    xytext=(5, -10), textcoords='offset points',
                    fontsize=9, alpha=0.7)

    SmallMLPlotStyle.format_axis(
        ax1,
        title='(A) Coverage Calibration',
        xlabel='Target Coverage (1-α) %',
        ylabel='Empirical Coverage %',
        grid=True,
        legend=True
    )
    ax1.set_xlim([75, 100])
    ax1.set_ylim([75, 100])
    ax1.set_aspect('equal')

    # ========== Panel B: Singleton Rate Trade-off ==========
    ax2 = axes[1]

    # Plot singleton rate vs alpha
    ax2.plot(alpha_values * 100, singleton_rates,
             'o-', color=SmallMLPlotStyle.COLORS['success'],
             markersize=10, linewidth=2.5,
             label='Definitive predictions')

    # Highlight default α=0.10
    default_idx = 1  # α=0.10
    ax2.scatter([alpha_values[default_idx] * 100],
               [singleton_rates[default_idx]],
               s=200, color=SmallMLPlotStyle.COLORS['danger'],
               marker='*', zorder=5,
               label='Default (α=0.10)')

    # Annotate key points
    for i, alpha in enumerate(alpha_values):
        ax2.annotate(f'{singleton_rates[i]:.1f}%',
                    xy=(alpha * 100, singleton_rates[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

    SmallMLPlotStyle.format_axis(
        ax2,
        title='(B) Decisiveness Trade-off',
        xlabel='Miscoverage Rate α (%)',
        ylabel='Singleton Sets %',
        grid=True,
        legend=True
    )
    ax2.set_xlim([0, 25])
    ax2.set_ylim([85, 100])

    # Add interpretation text box
    textstr = 'Lower α → Higher coverage\nbut fewer definitive predictions'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax2.text(0.05, 0.05, textstr, transform=ax2.transAxes,
            fontsize=9, verticalalignment='bottom', bbox=props)

    # Adjust layout and save
    plt.tight_layout()

    # Save figure
    figures_dir = Path(__file__).parent.parent / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "figure_c_2_multi_alpha_calibration.png"

    SmallMLPlotStyle.save_figure(fig, str(figure_path))
    plt.close()

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"[OK] Figure C.2 generated successfully!")
    print(f"\nOutput:")
    print(f"  - {figure_path}")
    print(f"\nKey Findings:")
    print(f"  1. Calibration: All alpha values within +/-3% tolerance")
    print(f"  2. Default (alpha=0.10): {singleton_rates[1]:.1f}% singleton rate")
    print(f"  3. Trade-off: Lower alpha provides more coverage but less decisiveness")
    print(f"\nFigure demonstrates conformal prediction's robustness across")
    print(f"multiple miscoverage levels, validating distribution-free guarantees.")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
