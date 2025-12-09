"""
SmallML Quickstart Example
==========================

This example demonstrates the complete SmallML workflow:
1. Load multi-entity data
2. Fit hierarchical Bayesian + conformal predictor
3. Make predictions with uncertainty
4. Evaluate on test data

NOTE: This example uses synthetic data for demonstration.
Replace with your own data in production.
"""

import pandas as pd
import numpy as np

# NOTE: If priors are not yet added, you'll get a warning but the pipeline will still work
from smallml import SmallMLPipeline

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("SmallML Quickstart Example")
print("="*70)

# ===== 1. Prepare Data =====
print("\n[Step 1] Generating synthetic multi-entity data...")
print("(In production, load your own CSV files here)\n")


def generate_store_data(n_customers, store_id, churn_rate=0.3):
    """Generate synthetic customer data for one store."""
    data = {
        'recency': np.random.exponential(scale=30, size=n_customers),
        'frequency': np.random.poisson(lam=5, size=n_customers),
        'monetary': np.random.lognormal(mean=4, sigma=1, size=n_customers),
        'tenure': np.random.uniform(1, 36, size=n_customers),
        'age': np.random.normal(45, 15, size=n_customers),
    }

    # Generate churn labels (higher recency/lower frequency → more churn)
    logit = (
        -2.0
        + 0.03 * data['recency']
        - 0.15 * data['frequency']
        - 0.0002 * data['monetary']
        - 0.02 * data['tenure']
        + np.random.normal(0, 0.5, n_customers)  # Store-specific noise
    )
    data['churned'] = (1 / (1 + np.exp(-logit)) > np.random.rand(n_customers)).astype(int)

    return pd.DataFrame(data)


# Create multi-store dataset
sme_data = {
    'store_1': generate_store_data(80, 1),
    'store_2': generate_store_data(120, 2),
    'store_3': generate_store_data(95, 3),
    'store_4': generate_store_data(150, 4),
    'store_5': generate_store_data(70, 5),
}

print(f"Generated data for {len(sme_data)} stores:")
for name, df in sme_data.items():
    churn_rate = df['churned'].mean()
    print(f"  {name}: {len(df)} customers, {churn_rate:.1%} churn rate")


# ===== 2. Create and Fit Pipeline =====
print("\n" + "="*70)
print("[Step 2] Fitting SmallML Pipeline")
print("="*70)
print("\nThis will take 15-30 minutes depending on your hardware.")
print("Progress will be shown below...\n")

pipeline = SmallMLPipeline(
    use_pretrained_priors=True,  # Will use bundled priors if available
    quick_mode=False,            # Use full MCMC for reliable results
    random_seed=42
)

# Fit pipeline (automatic convergence validation)
pipeline.fit(
    sme_data,
    target_col='churned',
    calibration_fraction=0.25,   # 25% reserved for conformal calibration
    validate_convergence=True    # Raises error if R̂ ≥ 1.01
)


# ===== 3. Check Convergence Diagnostics =====
print("\n" + "="*70)
print("[Step 3] MCMC Convergence Diagnostics")
print("="*70 + "\n")

diagnostics = pipeline.get_convergence_diagnostics()
print(diagnostics.head(10))
print(f"\n✓ Max R̂: {diagnostics['r_hat'].max():.4f} (should be < 1.01)")
print(f"✓ Min ESS: {diagnostics['ess_bulk'].min():.0f} (should be > 400)")


# ===== 4. Make Predictions =====
print("\n" + "="*70)
print("[Step 4] Making Predictions on New Customers")
print("="*70 + "\n")

# Generate new customers for store_1
new_customers = generate_store_data(20, 1).drop('churned', axis=1)

predictions = pipeline.predict(
    new_customers,
    sme_id='store_1',
    return_uncertainty=True
)

print("Predictions with uncertainty:\n")
print(predictions.to_string(index=False))

# Interpret results
print("\n" + "-"*70)
print("Interpretation:")
print("-"*70)
certain_no_churn = (predictions['conformal_set'] == '{0}').sum()
certain_churn = (predictions['conformal_set'] == '{1}').sum()
uncertain = (predictions['conformal_set'].str.contains('0, 1')).sum()

print(f"\n  ✓ {certain_no_churn} customers: Certain NO churn (low priority)")
print(f"  ⚠ {certain_churn} customers: Certain CHURN (high priority - intervene!)")
print(f"  ? {uncertain} customers: Uncertain (moderate priority)")


# ===== 5. Evaluate on Test Data =====
print("\n" + "="*70)
print("[Step 5] Evaluating on Test Data")
print("="*70 + "\n")

# Generate test data
test_data = generate_store_data(100, 1)
X_test = test_data.drop('churned', axis=1)
y_test = test_data['churned']

metrics = pipeline.evaluate(X_test, y_test, sme_id='store_1')

print("Performance Metrics:")
print("-"*70)
print(f"  AUC:                 {metrics['auc']:.3f}")
print(f"  Accuracy:            {metrics['accuracy']:.3f}")
print(f"  F1 Score:            {metrics['f1_score']:.3f}")
print(f"  Conformal Coverage:  {metrics['conformal_coverage']:.3f}  (target: 0.90)")
print(f"  Mean Set Size:       {metrics['mean_set_size']:.2f}  (1.0 = all certain)")


# ===== 6. Save Pipeline =====
print("\n" + "="*70)
print("[Step 6] Saving Fitted Pipeline")
print("="*70 + "\n")

pipeline.save('smallml_pipeline.pkl')
print("\n✓ Pipeline saved successfully!")
print("\nTo load later:")
print("  from smallml import SmallMLPipeline")
print("  pipeline = SmallMLPipeline.load('smallml_pipeline.pkl')")


# ===== Summary =====
print("\n" + "="*70)
print("✅ Quickstart Complete!")
print("="*70)
print("\nNext Steps:")
print("  1. Replace synthetic data with your own CSV files")
print("  2. Ensure all entities have the same feature names")
print("  3. Add your pre-trained priors to smallml/data/priors_churn.pkl")
print("  4. Run this script with your data")
print("  5. Deploy the saved pipeline to production")
print("\nFor more examples, see:")
print("  - examples/quickstart.ipynb (Jupyter notebook)")
print("  - docs/ (API documentation)")
print("\n" + "="*70 + "\n")
