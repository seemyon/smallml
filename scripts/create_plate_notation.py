"""
Create Figure 4.2: Hierarchical Model Plate Notation

This script generates a plate notation diagram showing the hierarchical structure
of the Bayesian model in Layer 2. Plate notation is the standard way to represent
repeated structures in graphical models.

Figure Generated:
- Figure 4.2: Hierarchical Bayesian Model - Plate Notation

Usage:
    python scripts/create_plate_notation.py

Author: SmallML Framework
Date: October 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path


def setup_figure():
    """Create and configure the figure."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    return fig, ax


def draw_node(ax, x, y, radius, label, observed=False, color='lightblue'):
    """Draw a node in the graphical model."""
    if observed:
        # Double circle for observed variables
        outer = Circle((x, y), radius, color='white', ec='black', linewidth=2, zorder=2)
        inner = Circle((x, y), radius * 0.92, color=color, ec='black', linewidth=2, zorder=3)
        ax.add_patch(outer)
        ax.add_patch(inner)
    else:
        # Single circle for latent variables
        circle = Circle((x, y), radius, color=color, ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)

    # Add label
    ax.text(x, y, label, fontsize=11, ha='center', va='center',
            fontweight='bold', zorder=4)


def draw_plate(ax, x, y, width, height, label, corner_label):
    """Draw a plate (rectangle with rounded corners)."""
    plate = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='gray',
        facecolor='none',
        linewidth=2.5,
        linestyle='--',
        zorder=1
    )
    ax.add_patch(plate)

    # Main label (bottom right)
    ax.text(x + width - 0.3, y + 0.2, label,
            fontsize=11, ha='right', va='bottom',
            style='italic', color='gray', fontweight='bold')

    # Corner label (top left)
    ax.text(x + 0.2, y + height - 0.2, corner_label,
            fontsize=10, ha='left', va='top',
            color='gray', fontweight='bold')


def draw_edge(ax, x1, y1, x2, y2, color='black', style='-'):
    """Draw a directed edge between nodes."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.3,head_length=0.3',
        color=color,
        linewidth=2,
        linestyle=style,
        zorder=1
    )
    ax.add_patch(arrow)


def create_plate_notation():
    """Create the hierarchical model plate notation diagram."""
    fig, ax = setup_figure()

    # ============================================================
    # LEVEL 1: HYPERPRIORS (Population Level)
    # ============================================================
    level1_y = 8.5

    # Prior mean β₀ (fixed, from transfer learning)
    draw_node(ax, 2, level1_y, 0.5, "β₀", observed=True, color='#E8F4F8')
    ax.text(2, level1_y - 0.9, "Prior mean\n(Transfer learning)",
            fontsize=9, ha='center', style='italic', color='gray')

    # Prior covariance Σ₀ (fixed, from transfer learning)
    draw_node(ax, 4.5, level1_y, 0.5, "Σ₀", observed=True, color='#E8F4F8')
    ax.text(4.5, level1_y - 0.9, "Prior covariance\n(Transfer learning)",
            fontsize=9, ha='center', style='italic', color='gray')

    # Hyperprior τ
    draw_node(ax, 7, level1_y, 0.5, "τ", observed=True, color='#E8F4F8')
    ax.text(7, level1_y - 0.9, "Industry variance\nhyperprior",
            fontsize=9, ha='center', style='italic', color='gray')

    # ============================================================
    # LEVEL 2: POPULATION PARAMETERS
    # ============================================================
    level2_y = 6.5

    # Population mean μ
    draw_node(ax, 3, level2_y, 0.6, "μ", observed=False, color='#FFE5CC')
    ax.text(3, level2_y - 1.0, "Industry mean\n(Population)",
            fontsize=9, ha='center', style='italic', color='gray')

    # Population std σ
    draw_node(ax, 6, level2_y, 0.6, "σ", observed=False, color='#FFE5CC')
    ax.text(6, level2_y - 1.0, "Industry std\n(Population)",
            fontsize=9, ha='center', style='italic', color='gray')

    # Edges from hyperpriors to population parameters
    draw_edge(ax, 2.3, level1_y - 0.5, 2.7, level2_y + 0.5, color='darkgreen')
    draw_edge(ax, 4.3, level1_y - 0.5, 3.3, level2_y + 0.5, color='darkgreen')
    draw_edge(ax, 7, level1_y - 0.5, 6.3, level2_y + 0.5, color='darkgreen')

    # ============================================================
    # LEVEL 3: SME-SPECIFIC PARAMETERS (Plate for J SMEs)
    # ============================================================

    # Draw outer plate for J SMEs
    plate_j_x, plate_j_y = 0.5, 1.5
    plate_j_width, plate_j_height = 11, 3.5
    draw_plate(ax, plate_j_x, plate_j_y, plate_j_width, plate_j_height,
               "j = 1, ..., J", "J SMEs")

    level3_y = 3.5

    # SME-specific coefficients β_j
    draw_node(ax, 4.5, level3_y, 0.6, "β_j", observed=False, color='#FFD6E0')
    ax.text(4.5, level3_y - 1.0, "SME coefficients",
            fontsize=9, ha='center', style='italic', color='gray')

    # Edges from population to SME-specific
    draw_edge(ax, 3.3, level2_y - 0.6, 4.2, level3_y + 0.5, color='darkblue')
    draw_edge(ax, 5.7, level2_y - 0.6, 4.8, level3_y + 0.5, color='darkblue')

    # ============================================================
    # LEVEL 4: OBSERVATIONS (Plate for n_j customers per SME)
    # ============================================================

    # Draw inner plate for n_j observations
    plate_n_x, plate_n_y = 6.5, 1.8
    plate_n_width, plate_n_height = 4.5, 2.9
    draw_plate(ax, plate_n_x, plate_n_y, plate_n_width, plate_n_height,
               "i = 1, ..., n_j", "n_j customers")

    level4_y = 3.0

    # Features x_ij (observed)
    draw_node(ax, 7.5, level4_y, 0.5, "x_ij", observed=True, color='#E8F4F8')
    ax.text(7.5, level4_y - 0.9, "Features",
            fontsize=9, ha='center', style='italic', color='gray')

    # Outcome y_ij (observed)
    draw_node(ax, 9.5, level4_y, 0.5, "y_ij", observed=True, color='#B3E5FC')
    ax.text(9.5, level4_y - 0.9, "Churn label",
            fontsize=9, ha='center', style='italic', color='gray')

    # Edges to observations
    draw_edge(ax, 5.1, level3_y - 0.2, 7.2, level4_y + 0.3, color='darkred')
    draw_edge(ax, 5.1, level3_y - 0.2, 9.2, level4_y + 0.3, color='darkred')
    draw_edge(ax, 7.8, level4_y, 9.2, level4_y, color='darkred')

    # ============================================================
    # ANNOTATIONS AND LEGEND
    # ============================================================

    # Title
    ax.text(6, 9.6, "Hierarchical Bayesian Model - Plate Notation",
            fontsize=16, fontweight='bold', ha='center')

    # Model equations
    equations = [
        "Level 1 (Hyperpriors):    μ ~ N(β₀, Σ₀),  σ ~ HalfNormal(τ)",
        "Level 2 (SME-specific):   β_j ~ N(μ, σ²I)   for j=1,...,J",
        "Level 3 (Observations):   y_ij ~ Bernoulli(σ(β_j^T x_ij))   for i=1,...,n_j"
    ]

    for i, eq in enumerate(equations):
        ax.text(6, 0.9 - i * 0.3, eq, fontsize=9, ha='center',
                family='monospace', bbox=dict(boxstyle='round',
                                              facecolor='lightyellow',
                                              alpha=0.5))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E8F4F8', edgecolor='black',
                       label='Fixed (Observed/Hyperprior)', linewidth=2),
        mpatches.Patch(facecolor='#FFE5CC', edgecolor='black',
                       label='Population Parameters (Latent)', linewidth=2),
        mpatches.Patch(facecolor='#FFD6E0', edgecolor='black',
                       label='SME Parameters (Latent)', linewidth=2),
        mpatches.Patch(facecolor='#B3E5FC', edgecolor='black',
                       label='Data (Observed)', linewidth=2)
    ]

    ax.legend(handles=legend_elements, loc='upper right',
              frameon=True, fontsize=9)

    # Description
    description = (
        "The hierarchical structure enables partial pooling: each SME's parameters (β_j) are drawn from\n"
        "a common population distribution (μ, σ), which itself is informed by transfer learning priors (β₀, Σ₀).\n"
        "This allows information sharing across J SMEs while respecting individual heterogeneity."
    )
    ax.text(6, 9.3, description, fontsize=8, ha='center', style='italic',
            color='gray')

    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("=" * 70)
    print("SmallML Framework - Plate Notation Diagram Generation")
    print("=" * 70)
    print()

    # Ensure output directory exists
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    print("✓ Output directory ready")
    print()

    # Create the diagram
    print("Creating Figure 4.2: Hierarchical Model Plate Notation...")
    fig = create_plate_notation()

    # Save the figure
    output_path = "results/figures/figure_4_2_plate_notation.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved to {output_path}")
    print()

    # Summary
    print("=" * 70)
    print("✅ PLATE NOTATION DIAGRAM COMPLETE")
    print("=" * 70)
    print()
    print("Generated Figure:")
    print("  • Figure 4.2 - Hierarchical Model Plate Notation")
    print()
    print("Diagram Components:")
    print("  • Level 1: Hyperpriors (β₀, Σ₀, τ) - Transfer learning inputs")
    print("  • Level 2: Population parameters (μ, σ) - Industry-level")
    print("  • Level 3: SME-specific parameters (β_j) - Per-business")
    print("  • Level 4: Observations (x_ij, y_ij) - Customer data")
    print("  • Outer plate: J SMEs")
    print("  • Inner plate: n_j customers per SME")


if __name__ == "__main__":
    main()
