"""
Create Figure 4.1: Three-Layer Architecture Diagram

This script generates a professional architectural diagram showing the complete
SmallML framework pipeline with three layers: Transfer Learning, Hierarchical
Bayesian, and Conformal Prediction.

Figure Generated:
- Figure 4.1: Three-Layer Architecture Diagram

Usage:
    python scripts/create_architecture_diagram.py

Author: SmallML Framework
Date: October 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path


def setup_figure():
    """Create and configure the figure."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    return fig, ax


def draw_layer_box(ax, x, y, width, height, color, label, alpha=0.15):
    """Draw a layer background box."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        linewidth=2
    )
    ax.add_patch(box)

    # Add layer label
    ax.text(
        x + width / 2, y + height - 0.3,
        label,
        fontsize=13,
        fontweight='bold',
        ha='center',
        va='top',
        color=color
    )


def draw_component_box(ax, x, y, width, height, color, label, sublabel=""):
    """Draw a component box with label."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        edgecolor=color,
        facecolor='white',
        linewidth=2
    )
    ax.add_patch(box)

    # Main label
    ax.text(
        x + width / 2, y + height / 2 + 0.15,
        label,
        fontsize=10,
        fontweight='bold',
        ha='center',
        va='center'
    )

    # Sublabel if provided
    if sublabel:
        ax.text(
            x + width / 2, y + height / 2 - 0.15,
            sublabel,
            fontsize=8,
            ha='center',
            va='center',
            style='italic',
            color='gray'
        )


def draw_arrow(ax, x1, y1, x2, y2, label="", color='black'):
    """Draw an arrow between components."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=color,
        linewidth=2,
        zorder=1
    )
    ax.add_patch(arrow)

    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x, mid_y + 0.15,
            label,
            fontsize=8,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none')
        )


def create_architecture_diagram():
    """Create the complete three-layer architecture diagram."""
    fig, ax = setup_figure()

    # Define colors for each layer
    color_layer1 = '#2E86AB'  # Blue - Transfer Learning
    color_layer2 = '#A23B72'  # Purple - Hierarchical Bayesian
    color_layer3 = '#F18F01'  # Orange - Conformal Prediction
    color_io = '#6C757D'      # Gray - Input/Output

    # ============================================================
    # INPUT LAYER (Bottom)
    # ============================================================
    input_y = 0.5
    draw_component_box(ax, 1, input_y, 2.5, 0.8, color_io,
                       "SME Customer Data", "n=50-500 samples")

    # ============================================================
    # LAYER 1: Transfer Learning (Bottom Layer)
    # ============================================================
    layer1_y = 2.5
    draw_layer_box(ax, 0.3, layer1_y, 13.4, 2.2, color_layer1,
                   "Layer 1: Transfer Learning Foundation")

    # Public datasets component
    draw_component_box(ax, 1, layer1_y + 1.2, 2, 0.7, color_layer1,
                       "Public Datasets", "N≈100K+")

    # CatBoost component
    draw_component_box(ax, 4, layer1_y + 1.2, 2, 0.7, color_layer1,
                       "CatBoost Model", "Gradient Boosting")

    # SHAP component
    draw_component_box(ax, 7, layer1_y + 1.2, 2, 0.7, color_layer1,
                       "SHAP Extractor", "Feature Importance")

    # Priors component
    draw_component_box(ax, 10.5, layer1_y + 1.2, 2.3, 0.7, color_layer1,
                       "Extracted Priors", "β₀, Σ₀")

    # Arrows within Layer 1
    draw_arrow(ax, 3, layer1_y + 1.55, 4, layer1_y + 1.55, "", color_layer1)
    draw_arrow(ax, 6, layer1_y + 1.55, 7, layer1_y + 1.55, "", color_layer1)
    draw_arrow(ax, 9, layer1_y + 1.55, 10.5, layer1_y + 1.55, "", color_layer1)

    # ============================================================
    # LAYER 2: Hierarchical Bayesian (Middle Layer)
    # ============================================================
    layer2_y = 5.2
    draw_layer_box(ax, 0.3, layer2_y, 13.4, 2.2, color_layer2,
                   "Layer 2: Hierarchical Bayesian Core")

    # Prior incorporation
    draw_component_box(ax, 1, layer2_y + 1.2, 2.3, 0.7, color_layer2,
                       "Prior Distribution", "N(β₀, Σ₀)")

    # Hierarchical model
    draw_component_box(ax, 4.5, layer2_y + 1.2, 2.5, 0.7, color_layer2,
                       "Hierarchical Model", "3-Level Structure")

    # MCMC sampling
    draw_component_box(ax, 8, layer2_y + 1.2, 2, 0.7, color_layer2,
                       "MCMC (NUTS)", "4 chains × 2K draws")

    # Posterior
    draw_component_box(ax, 10.8, layer2_y + 1.2, 2, 0.7, color_layer2,
                       "Posteriors", "μ, σ, β_j")

    # Arrows within Layer 2
    draw_arrow(ax, 3.3, layer2_y + 1.55, 4.5, layer2_y + 1.55, "", color_layer2)
    draw_arrow(ax, 7, layer2_y + 1.55, 8, layer2_y + 1.55, "", color_layer2)
    draw_arrow(ax, 10, layer2_y + 1.55, 10.8, layer2_y + 1.55, "", color_layer2)

    # ============================================================
    # LAYER 3: Conformal Prediction (Top Layer)
    # ============================================================
    layer3_y = 7.9
    draw_layer_box(ax, 0.3, layer3_y, 13.4, 1.6, color_layer3,
                   "Layer 3: Conformal Prediction Wrapper")

    # Calibration component
    draw_component_box(ax, 1.5, layer3_y + 0.6, 2.5, 0.7, color_layer3,
                       "Calibration Set", "25% holdout")

    # Nonconformity scores
    draw_component_box(ax, 5, layer3_y + 0.6, 2.5, 0.7, color_layer3,
                       "Nonconformity Scores", "s_i = |y_i - ŷ_i|")

    # Threshold
    draw_component_box(ax, 8.5, layer3_y + 0.6, 1.8, 0.7, color_layer3,
                       "Threshold q̂", "α=0.10")

    # Prediction sets
    draw_component_box(ax, 11, layer3_y + 0.6, 2, 0.7, color_layer3,
                       "Prediction Sets", "C(x)")

    # Arrows within Layer 3
    draw_arrow(ax, 4, layer3_y + 0.95, 5, layer3_y + 0.95, "", color_layer3)
    draw_arrow(ax, 7.5, layer3_y + 0.95, 8.5, layer3_y + 0.95, "", color_layer3)
    draw_arrow(ax, 10.3, layer3_y + 0.95, 11, layer3_y + 0.95, "", color_layer3)

    # ============================================================
    # OUTPUT LAYER (Top)
    # ============================================================
    output_y = 9.8
    draw_component_box(ax, 5, output_y, 4, 0.8, color_io,
                       "Predictions + Uncertainty", "Point + Intervals + Sets")

    # ============================================================
    # VERTICAL CONNECTIONS (Between Layers)
    # ============================================================

    # Input → Layer 1
    draw_arrow(ax, 2.25, input_y + 0.8, 2, layer1_y + 1.2,
               "Raw Data", color_io)

    # Layer 1 → Layer 2 (Priors)
    draw_arrow(ax, 11.65, layer1_y + 1.9, 2.15, layer2_y + 1.2,
               "Priors", 'darkgreen')

    # Input → Layer 2 (SME Data)
    draw_arrow(ax, 2.25, input_y + 0.8, 5.75, layer2_y + 1.2,
               "SME Data", color_io)

    # Layer 2 → Layer 3 (Posteriors)
    draw_arrow(ax, 11.8, layer2_y + 1.9, 2.75, layer3_y + 0.6,
               "Posteriors", 'darkgreen')

    # Layer 3 → Output
    draw_arrow(ax, 12, layer3_y + 1.3, 7, output_y,
               "Final", 'darkgreen')

    # ============================================================
    # TITLE AND LEGEND
    # ============================================================

    ax.text(7, 9.6,
            "SmallML Framework: Three-Layer Architecture",
            fontsize=16, fontweight='bold', ha='center', va='bottom')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=color_layer1, alpha=0.3, edgecolor=color_layer1,
                       label='Layer 1: Transfer Learning', linewidth=2),
        mpatches.Patch(facecolor=color_layer2, alpha=0.3, edgecolor=color_layer2,
                       label='Layer 2: Hierarchical Bayesian', linewidth=2),
        mpatches.Patch(facecolor=color_layer3, alpha=0.3, edgecolor=color_layer3,
                       label='Layer 3: Conformal Prediction', linewidth=2)
    ]

    ax.legend(handles=legend_elements, loc='upper left',
              frameon=True, fontsize=10, bbox_to_anchor=(0.02, 0.98))

    # Add framework description
    description = (
        "The SmallML framework processes small datasets (n=50-500) through three layers:\n"
        "1) Transfer learning extracts priors from large public data\n"
        "2) Hierarchical Bayesian inference pools strength across SMEs\n"
        "3) Conformal prediction provides distribution-free uncertainty"
    )
    ax.text(7, 0.2, description, fontsize=8, ha='center', va='top',
            style='italic', color='gray')

    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("=" * 70)
    print("Architecture Diagram Generation")
    print("=" * 70)
    print()

    # Ensure output directory exists
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    print("✓ Output directory ready")
    print()

    # Create the diagram
    print("Creating Figure 4.1: Three-Layer Architecture Diagram...")
    fig = create_architecture_diagram()

    # Save the figure
    output_path = "results/figures/figure_4_1_architecture.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved to {output_path}")
    print()

    # Summary
    print("=" * 70)
    print("✅ ARCHITECTURE DIAGRAM COMPLETE")
    print("=" * 70)
    print()
    print("Generated Figure:")
    print("  • Figure 4.1 - Three-Layer Architecture Diagram")
    print()
    print("Diagram Components:")
    print("  • Input Layer: SME Customer Data")
    print("  • Layer 1: Transfer Learning (CatBoost → SHAP → Priors)")
    print("  • Layer 2: Hierarchical Bayesian (Prior → Model → MCMC → Posterior)")
    print("  • Layer 3: Conformal Prediction (Calibration → Scores → Sets)")
    print("  • Output Layer: Predictions with Uncertainty")


if __name__ == "__main__":
    main()
