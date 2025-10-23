"""
Create Figure 4.6: End-to-End Deployment Pipeline

This script generates a flowchart showing the complete SmallML deployment
process from initial setup through production predictions, including
retraining schedules and monitoring.

Figure Generated:
- Figure 4.6: End-to-End Deployment Pipeline

Usage:
    python scripts/create_deployment_diagram.py

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path


def setup_figure():
    """Create and configure the figure."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')
    return fig, ax


def draw_stage_box(ax, x, y, width, height, color, label, sublabel="", style='round'):
    """Draw a pipeline stage box."""
    if style == 'round':
        boxstyle = "round,pad=0.1"
    elif style == 'database':
        boxstyle = "round,pad=0.15"
    else:
        boxstyle = "square,pad=0.1"

    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=boxstyle,
        edgecolor=color,
        facecolor='white',
        linewidth=2.5
    )
    ax.add_patch(box)

    # Main label
    ax.text(
        x + width / 2, y + height / 2 + 0.15,
        label,
        fontsize=11,
        fontweight='bold',
        ha='center',
        va='center'
    )

    # Sublabel
    if sublabel:
        ax.text(
            x + width / 2, y + height / 2 - 0.20,
            sublabel,
            fontsize=8,
            ha='center',
            va='center',
            style='italic',
            color='gray'
        )


def draw_arrow(ax, x1, y1, x2, y2, label="", color='black', style='-'):
    """Draw an arrow between stages."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=color,
        linewidth=2.5,
        linestyle=style,
        zorder=1
    )
    ax.add_patch(arrow)

    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x + 0.3, mid_y,
            label,
            fontsize=8,
            ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='none', alpha=0.9)
        )


def create_deployment_diagram():
    """Create the complete deployment pipeline diagram."""
    fig, ax = setup_figure()

    # Define colors
    color_setup = '#6C757D'      # Gray
    color_training = '#2E86AB'   # Blue
    color_deploy = '#F18F01'     # Orange
    color_monitor = '#C73E1D'    # Red
    color_retrain = '#6A994E'    # Green

    # ============================================================
    # PHASE 1: INITIAL SETUP (Top)
    # ============================================================
    setup_y = 9.5

    draw_stage_box(ax, 0.5, setup_y, 2.5, 0.9, color_setup,
                   "1. Environment\nSetup", "Install dependencies")

    draw_stage_box(ax, 3.5, setup_y, 2.5, 0.9, color_setup,
                   "2. Data\nCollection", "Gather datasets")

    draw_stage_box(ax, 6.5, setup_y, 2.5, 0.9, color_setup,
                   "3. Data\nHarmonization", "Feature alignment")

    # Arrows
    draw_arrow(ax, 3, setup_y + 0.45, 3.5, setup_y + 0.45)
    draw_arrow(ax, 6, setup_y + 0.45, 6.5, setup_y + 0.45)

    # ============================================================
    # PHASE 2: TRAINING PIPELINE (Upper Middle)
    # ============================================================
    training_y = 7.5

    # Main training flow
    draw_stage_box(ax, 0.5, training_y, 2.2, 0.9, color_training,
                   "4. Transfer\nLearning", "CatBoost + SHAP")

    draw_stage_box(ax, 3.2, training_y, 2.2, 0.9, color_training,
                   "5. Hierarchical\nBayesian", "PyMC MCMC")

    draw_stage_box(ax, 6, training_y, 2.2, 0.9, color_training,
                   "6. Conformal\nCalibration", "Threshold q̂")

    # Model artifacts (database)
    draw_stage_box(ax, 9.5, training_y, 2.8, 0.9, color_training,
                   "Trained Models", "Priors, Trace, q̂", style='database')

    # Arrows
    draw_arrow(ax, 7.75, setup_y, 1.6, training_y + 0.9, "Data")
    draw_arrow(ax, 2.7, training_y + 0.45, 3.2, training_y + 0.45, "Priors")
    draw_arrow(ax, 5.4, training_y + 0.45, 6, training_y + 0.45, "Posteriors")
    draw_arrow(ax, 8.2, training_y + 0.45, 9.5, training_y + 0.45)

    # Time annotations
    ax.text(1.6, training_y - 0.3, "4-6 hours", fontsize=7,
            ha='center', style='italic', color='gray')
    ax.text(4.3, training_y - 0.3, "30-60 min", fontsize=7,
            ha='center', style='italic', color='gray')
    ax.text(7.1, training_y - 0.3, "5-10 min", fontsize=7,
            ha='center', style='italic', color='gray')

    # ============================================================
    # PHASE 3: VALIDATION (Middle)
    # ============================================================
    validation_y = 6.0

    draw_stage_box(ax, 2.5, validation_y, 2.5, 0.9, color_training,
                   "7. Convergence\nCheck", "R̂ < 1.01, ESS > 400")

    draw_stage_box(ax, 5.5, validation_y, 2.5, 0.9, color_training,
                   "8. Coverage\nValidation", "87-93% empirical")

    # Decision diamonds
    ax.text(3.75, validation_y - 0.5, "Pass?", fontsize=9,
            ha='center', fontweight='bold', color=color_training)
    ax.text(6.75, validation_y - 0.5, "Pass?", fontsize=9,
            ha='center', fontweight='bold', color=color_training)

    # Arrows
    draw_arrow(ax, 10.85, training_y, 3.75, validation_y + 0.9)
    draw_arrow(ax, 5, validation_y + 0.45, 5.5, validation_y + 0.45)

    # Failure loops (dashed)
    draw_arrow(ax, 3, validation_y, 4.3, training_y, "Retrain", color_training, style='--')
    draw_arrow(ax, 6.5, validation_y, 7.1, training_y, "Recalibrate", color_training, style='--')

    # ============================================================
    # PHASE 4: DEPLOYMENT (Lower Middle)
    # ============================================================
    deploy_y = 4.2

    draw_stage_box(ax, 1, deploy_y, 2.2, 0.9, color_deploy,
                   "9. Package\nModel", "Serialize artifacts")

    draw_stage_box(ax, 3.8, deploy_y, 2.2, 0.9, color_deploy,
                   "10. Deploy to\nProduction", "API/Service")

    draw_stage_box(ax, 6.8, deploy_y, 2.2, 0.9, color_deploy,
                   "11. Integration\nTesting", "End-to-end test")

    # Arrows
    draw_arrow(ax, 6.75, validation_y, 2.1, deploy_y + 0.9, "Validated")
    draw_arrow(ax, 3.2, deploy_y + 0.45, 3.8, deploy_y + 0.45)
    draw_arrow(ax, 6, deploy_y + 0.45, 6.8, deploy_y + 0.45)

    # ============================================================
    # PHASE 5: PRODUCTION (Lower)
    # ============================================================
    prod_y = 2.4

    draw_stage_box(ax, 1.5, prod_y, 2.5, 0.9, color_monitor,
                   "12. Serve\nPredictions", "Real-time API")

    draw_stage_box(ax, 5, prod_y, 2.5, 0.9, color_monitor,
                   "13. Monitor\nPerformance", "Coverage, AUC")

    draw_stage_box(ax, 8.5, prod_y, 2.5, 0.9, color_monitor,
                   "14. Log Data\n& Drift", "Track distributions")

    # Arrows
    draw_arrow(ax, 7.9, deploy_y, 2.75, prod_y + 0.9, "Live")
    draw_arrow(ax, 4, prod_y + 0.45, 5, prod_y + 0.45)
    draw_arrow(ax, 7.5, prod_y + 0.45, 8.5, prod_y + 0.45)

    # ============================================================
    # PHASE 6: RETRAINING CYCLE (Bottom)
    # ============================================================
    retrain_y = 0.5

    draw_stage_box(ax, 2, retrain_y, 3, 0.9, color_retrain,
                   "15. Scheduled Retraining", "Monthly or on drift")

    # Feedback arrow to training
    draw_arrow(ax, 11, prod_y + 0.3, 11.5, training_y + 0.3, "", color_retrain)
    draw_arrow(ax, 11.5, training_y + 0.3, 11.5, 1.4, "", color_retrain)
    draw_arrow(ax, 11.5, 1.4, 5, 1.4, "", color_retrain)
    draw_arrow(ax, 5, 1.4, 3.5, retrain_y + 0.9, "Drift detected", color_retrain)

    # Scheduled retrain arrow
    draw_arrow(ax, 3.5, retrain_y, 1.6, training_y, "New data", color_retrain, style='--')

    # ============================================================
    # RETRAINING SCHEDULE BOX (Right side)
    # ============================================================
    schedule_x, schedule_y = 9.8, 3.5
    schedule_box = FancyBboxPatch(
        (schedule_x, schedule_y), 3.7, 3.5,
        boxstyle="round,pad=0.15",
        edgecolor=color_retrain,
        facecolor='#F0FFF0',
        linewidth=2,
        linestyle='--'
    )
    ax.add_patch(schedule_box)

    ax.text(schedule_x + 1.85, schedule_y + 3.2, "Retraining Schedule",
            fontsize=11, fontweight='bold', ha='center', color=color_retrain)

    schedule_text = [
        "Layer 1 (Transfer):",
        "  • Quarterly/Semi-annual",
        "  • 4-6 hours",
        "",
        "Layer 2 (Hierarchical):",
        "  • Monthly per SME",
        "  • 15-30 minutes",
        "",
        "Layer 3 (Conformal):",
        "  • After Layer 2 update",
        "  • <1 minute"
    ]

    y_pos = schedule_y + 2.8
    for line in schedule_text:
        ax.text(schedule_x + 0.2, y_pos, line,
                fontsize=8, ha='left', va='top', family='monospace')
        y_pos -= 0.22

    # ============================================================
    # TITLE AND LEGEND
    # ============================================================
    ax.text(7, 10.7, "SmallML Deployment Pipeline: End-to-End Workflow",
            fontsize=16, fontweight='bold', ha='center')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor=color_setup,
                       label='Setup Phase', linewidth=2.5),
        mpatches.Patch(facecolor='white', edgecolor=color_training,
                       label='Training Phase', linewidth=2.5),
        mpatches.Patch(facecolor='white', edgecolor=color_deploy,
                       label='Deployment Phase', linewidth=2.5),
        mpatches.Patch(facecolor='white', edgecolor=color_monitor,
                       label='Production Phase', linewidth=2.5),
        mpatches.Patch(facecolor='white', edgecolor=color_retrain,
                       label='Retraining Cycle', linewidth=2.5)
    ]

    ax.legend(handles=legend_elements, loc='lower left',
              frameon=True, fontsize=9, ncol=3)

    # Description
    description = (
        "Complete deployment pipeline showing 15 stages from initial setup through production monitoring and retraining. "
        "Dashed arrows indicate\nfailure recovery paths and scheduled maintenance cycles. "
        "Green box shows recommended retraining frequency for each layer."
    )
    ax.text(7, 0.15, description, fontsize=8, ha='center', style='italic', color='gray')

    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("=" * 70)
    print("SmallML Framework - Deployment Diagram Generation")
    print("=" * 70)
    print()

    # Ensure output directory exists
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    print("✓ Output directory ready")
    print()

    # Create the diagram
    print("Creating Figure 4.6: End-to-End Deployment Pipeline...")
    fig = create_deployment_diagram()

    # Save the figure
    output_path = "results/figures/figure_4_6_deployment_pipeline.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved to {output_path}")
    print()

    # Summary
    print("=" * 70)
    print("✅ DEPLOYMENT DIAGRAM COMPLETE")
    print("=" * 70)
    print()
    print("Generated Figure:")
    print("  • Figure 4.6 - End-to-End Deployment Pipeline")
    print()
    print("Pipeline Phases:")
    print("  1. Setup (Gray): Environment, data collection, harmonization")
    print("  2. Training (Blue): Transfer learning, hierarchical, conformal")
    print("  3. Validation (Blue): Convergence checks, coverage validation")
    print("  4. Deployment (Orange): Packaging, deployment, integration testing")
    print("  5. Production (Red): Serve predictions, monitoring, logging")
    print("  6. Retraining (Green): Scheduled updates and drift response")
    print()
    print("Key Features:")
    print("  • 15 distinct stages with time estimates")
    print("  • Failure recovery paths (dashed arrows)")
    print("  • Retraining schedule box (right side)")
    print("  • Complete feedback loop from production to training")


if __name__ == "__main__":
    main()
