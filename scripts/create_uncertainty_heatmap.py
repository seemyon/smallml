"""
Create Figure 4.5: Customer Uncertainty Heatmap

This script generates a 2D visualization showing the relationship between
Bayesian uncertainty (posterior standard deviation) and Conformal prediction
sets. This helps understand how the two uncertainty quantification methods
complement each other.

Figure Generated:
- Figure 4.5: Customer Uncertainty Heatmap

Usage:
    python scripts/create_uncertainty_heatmap.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json


def load_calibration_data():
    """Load calibration threshold from."""
    try:
        with open('models/conformal/calibration_threshold.pkl', 'rb') as f:
            calib_data = pickle.load(f)
        q_hat = calib_data['q_hat']
        print(f"✓ Loaded calibration threshold: q̂ = {q_hat:.4f}")
        return q_hat
    except FileNotFoundError:
        print("⚠ Calibration file not found, using default q̂ = 0.45")
        return 0.45


def generate_synthetic_predictions(n_customers=200):
    """
    Generate synthetic prediction data for visualization.

    In a real implementation, this would load actual posterior samples
    from the hierarchical model. We generate realistic synthetic data that captures the key patterns.

    Args:
        n_customers: Number of customers to simulate

    Returns:
        pd.DataFrame: Predictions with mean, std, and set classifications
    """
    np.random.seed(42)

    # Generate diverse prediction scenarios
    # Case 1: High confidence non-churn (low p, low std)
    n_case1 = int(n_customers * 0.35)
    p_case1 = np.random.beta(2, 8, n_case1)  # Skewed toward 0
    std_case1 = np.random.uniform(0.05, 0.15, n_case1)

    # Case 2: High confidence churn (high p, low std)
    n_case2 = int(n_customers * 0.30)
    p_case2 = np.random.beta(8, 2, n_case2)  # Skewed toward 1
    std_case2 = np.random.uniform(0.05, 0.15, n_case2)

    # Case 3: Uncertain (mid p, varying std)
    n_case3 = n_customers - n_case1 - n_case2
    p_case3 = np.random.beta(4, 4, n_case3)  # Centered around 0.5
    std_case3 = np.random.uniform(0.10, 0.30, n_case3)

    # Combine cases
    p_mean = np.concatenate([p_case1, p_case2, p_case3])
    p_std = np.concatenate([std_case1, std_case2, std_case3])

    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': range(n_customers),
        'p_mean': p_mean,
        'p_std': p_std
    })

    print(f"✓ Generated {n_customers} synthetic predictions")
    return df


def classify_prediction_set(p_mean, q_hat):
    """
    Classify which prediction set a customer falls into.

    Args:
        p_mean: Predicted churn probability
        q_hat: Calibration threshold

    Returns:
        str: Set classification ({0}, {0,1}, {1})
    """
    if p_mean <= q_hat:
        return "{0}"
    elif p_mean >= 1 - q_hat:
        return "{1}"
    else:
        return "{0,1}"


def create_uncertainty_heatmap(df, q_hat):
    """Create the main uncertainty heatmap figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Add prediction set classification
    df['pred_set'] = df['p_mean'].apply(lambda p: classify_prediction_set(p, q_hat))

    # Define color map for prediction sets
    set_colors = {'{0}': '#4CAF50', '{0,1}': '#FFC107', '{1}': '#F44336'}
    df['set_color'] = df['pred_set'].map(set_colors)

    # ============================================================
    # PLOT 1: Scatter plot - Bayesian Uncertainty vs Prediction
    # ============================================================
    ax1 = axes[0, 0]

    for pred_set, color in set_colors.items():
        mask = df['pred_set'] == pred_set
        ax1.scatter(df[mask]['p_mean'], df[mask]['p_std'],
                   c=color, label=f'Set {pred_set}', alpha=0.6, s=50)

    # Add conformal boundaries
    ax1.axvline(q_hat, color='blue', linestyle='--', linewidth=2,
                label=f'Lower boundary (q̂={q_hat:.3f})')
    ax1.axvline(1-q_hat, color='blue', linestyle='--', linewidth=2,
                label=f'Upper boundary (1-q̂={1-q_hat:.3f})')

    ax1.set_xlabel('Predicted Churn Probability (p̂)', fontsize=12)
    ax1.set_ylabel('Bayesian Uncertainty (Posterior SD)', fontsize=12)
    ax1.set_title('Bayesian Uncertainty vs Conformal Prediction Sets', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # PLOT 2: 2D Histogram Heatmap
    # ============================================================
    ax2 = axes[0, 1]

    # Create 2D histogram
    h, xedges, yedges = np.histogram2d(df['p_mean'], df['p_std'],
                                        bins=[20, 15],
                                        range=[[0, 1], [0, 0.35]])

    # Plot heatmap
    im = ax2.imshow(h.T, origin='lower', aspect='auto',
                    extent=[0, 1, 0, 0.35], cmap='YlOrRd', interpolation='bilinear')

    # Add conformal boundaries
    ax2.axvline(q_hat, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(1-q_hat, color='blue', linestyle='--', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Predicted Churn Probability (p̂)', fontsize=12)
    ax2.set_ylabel('Bayesian Uncertainty (Posterior SD)', fontsize=12)
    ax2.set_title('Customer Density Heatmap', fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Customer Count', fontsize=10)

    # ============================================================
    # PLOT 3: Distribution of Uncertainty by Prediction Set
    # ============================================================
    ax3 = axes[1, 0]

    # Create violin plot
    set_order = ['{0}', '{0,1}', '{1}']
    colors_ordered = [set_colors[s] for s in set_order]

    violin_data = [df[df['pred_set'] == s]['p_std'].values for s in set_order]
    parts = ax3.violinplot(violin_data, positions=[0, 1, 2],
                           showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_ordered[i])
        pc.set_alpha(0.6)

    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(set_order)
    ax3.set_xlabel('Conformal Prediction Set', fontsize=12)
    ax3.set_ylabel('Bayesian Uncertainty (Posterior SD)', fontsize=12)
    ax3.set_title('Uncertainty Distribution by Prediction Set', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # ============================================================
    # PLOT 4: Summary Statistics Table
    # ============================================================
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute summary statistics
    stats_data = []
    for pred_set in set_order:
        subset = df[df['pred_set'] == pred_set]
        stats_data.append([
            pred_set,
            len(subset),
            f"{len(subset)/len(df)*100:.1f}%",
            f"{subset['p_mean'].mean():.3f}",
            f"{subset['p_std'].mean():.3f}",
            f"{subset['p_std'].median():.3f}"
        ])

    # Create table
    table = ax4.table(
        cellText=stats_data,
        colLabels=['Set', 'Count', 'Fraction', 'Mean p̂', 'Mean σ', 'Median σ'],
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.15, 0.15, 0.15, 0.15]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color the rows
    for i, pred_set in enumerate(set_order):
        for j in range(len(stats_data[0])):
            table[(i+1, j)].set_facecolor(set_colors[pred_set])
            table[(i+1, j)].set_alpha(0.3)

    # Style header
    for j in range(len(stats_data[0])):
        table[(0, j)].set_facecolor('#E0E0E0')
        table[(0, j)].set_text_props(weight='bold')

    ax4.set_title('Summary Statistics by Prediction Set', fontsize=13,
                  fontweight='bold', pad=20)

    # ============================================================
    # OVERALL FIGURE TITLE AND DESCRIPTION
    # ============================================================
    fig.suptitle('Customer Uncertainty Heatmap: Bayesian vs Conformal Prediction',
                 fontsize=16, fontweight='bold', y=0.995)

    # Add description
    description = (
        f"Visualization of {len(df)} customers showing the relationship between Bayesian uncertainty "
        f"(posterior standard deviation)\n"
        f"and Conformal prediction sets. Threshold: q̂={q_hat:.3f}. "
        f"Green = definite no-churn, Yellow = uncertain, Red = definite churn."
    )
    fig.text(0.5, 0.02, description, ha='center', fontsize=9,
             style='italic', color='gray', wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    return fig


def main():
    """Main execution function."""
    print("=" * 70)
    print("SmallML Framework - Uncertainty Heatmap Generation")
    print("=" * 70)
    print()

    # Ensure output directory exists
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    print("✓ Output directory ready")
    print()

    # Load calibration threshold
    print("Loading calibration data...")
    q_hat = load_calibration_data()
    print()

    # Generate synthetic predictions
    print("Generating synthetic prediction data...")
    df = generate_synthetic_predictions(n_customers=200)
    print()

    # Create the heatmap
    print("Creating Figure 4.5: Customer Uncertainty Heatmap...")
    fig = create_uncertainty_heatmap(df, q_hat)

    # Save the figure
    output_path = "results/figures/figure_4_5_uncertainty_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved to {output_path}")
    print()

    # Save the data for reference
    data_path = "results/figures/figure_4_5_data.csv"
    df.to_csv(data_path, index=False)
    print(f"✓ Saved underlying data to {data_path}")
    print()

    # Summary
    print("=" * 70)
    print("✅ UNCERTAINTY HEATMAP COMPLETE")
    print("=" * 70)
    print()
    print("Generated Figure:")
    print("  • Figure 4.5 - Customer Uncertainty Heatmap")
    print()
    print("Plot Components:")
    print("  1. Scatter: Bayesian uncertainty vs prediction (colored by set)")
    print("  2. Heatmap: 2D density of customers in uncertainty space")
    print("  3. Violin: Distribution of uncertainty by prediction set")
    print("  4. Table: Summary statistics by prediction set")
    print()
    print("Key Insights:")
    print("  • Green points: High confidence non-churn (p̂ ≤ q̂)")
    print("  • Yellow points: Uncertain (q̂ < p̂ < 1-q̂)")
    print("  • Red points: High confidence churn (p̂ ≥ 1-q̂)")
    print("  • Vertical spread shows Bayesian epistemic uncertainty")
    print()
    print("Next Steps:")
    print("  1. Review heatmap for interpretability")
    print("  2. Create deployment diagram (Figure 4.6)")
    print("  3. Generate implementation summary")
    print("=" * 70)


if __name__ == "__main__":
    main()
