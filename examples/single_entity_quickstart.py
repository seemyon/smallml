"""
SmallML Single-Entity Mode Example
==================================

This example demonstrates SmallML for users with only ONE dataset
(e.g., one store, one business unit).

Key differences from multi-entity mode:
- Pass a single DataFrame directly (no dict needed)
- Pre-trained priors are strongly recommended
- No cross-entity pooling, but priors provide statistical strength

Use this mode when:
- You have only one business location
- You're a solo entrepreneur with limited data
- You want to test SmallML before expanding to multiple entities
"""

import pandas as pd
import numpy as np

from smallml import Pipeline


def generate_customer_data(n_customers, churn_rate=0.3):
    """Generate synthetic customer data for a single business."""
    data = {
        'recency': np.random.exponential(scale=30, size=n_customers),
        'frequency': np.random.poisson(lam=5, size=n_customers),
        'monetary': np.random.lognormal(mean=4, sigma=1, size=n_customers),
        'tenure': np.random.uniform(1, 36, size=n_customers),
        'age': np.random.normal(45, 15, size=n_customers),
    }

    # Generate churn labels
    logit = (
        -2.0
        + 0.03 * data['recency']
        - 0.15 * data['frequency']
        - 0.0002 * data['monetary']
        - 0.02 * data['tenure']
        + np.random.normal(0, 0.5, n_customers)
    )
    data['churned'] = (1 / (1 + np.exp(-logit)) > np.random.rand(n_customers)).astype(int)

    return pd.DataFrame(data)


if __name__ == '__main__':
    np.random.seed(42)

    print("="*70)
    print("SmallML Single-Entity Mode Example")
    print("="*70)

    # ===== 1. Prepare Data =====
    print("\n[Step 1] Loading single-entity data...")
    print("(In production, load your CSV: pd.read_csv('my_customers.csv'))\n")

    # Generate single dataset (e.g., your one coffee shop's customer data)
    my_data = generate_customer_data(100)

    print(f"Dataset: {len(my_data)} customers, {my_data['churned'].mean():.1%} churn rate")
    print(f"Features: {[c for c in my_data.columns if c != 'churned']}")


    # ===== 2. Create and Fit Pipeline =====
    print("\n" + "="*70)
    print("[Step 2] Fitting SmallML Pipeline (Single-Entity Mode)")
    print("="*70)
    print("\nNote: Pre-trained priors strongly recommended for single-entity mode.")
    print("This provides the statistical strength that multi-entity pooling would offer.\n")

    # Initialize pipeline with pre-trained priors
    pipeline = Pipeline(
        use_pretrained_priors=True,   # IMPORTANT for single-entity mode
        quick_mode=False,             # Use full MCMC for reliable results
        random_seed=42
    )

    # Fit pipeline with a SINGLE DataFrame (no dict needed!)
    pipeline.fit(
        my_data,                      # Single DataFrame, not a dict
        target_col='churned',
        calibration_fraction=0.25,
        validate_convergence=False    # Allow completion with synthetic data
    )


    # ===== 3. Make Predictions =====
    print("\n" + "="*70)
    print("[Step 3] Making Predictions")
    print("="*70 + "\n")

    # Generate new customers (your new leads/customers)
    new_customers = generate_customer_data(15).drop('churned', axis=1)

    # No need to specify sme_id in single-entity mode!
    predictions = pipeline.predict(new_customers, return_uncertainty=True)

    print("Predictions with uncertainty:\n")
    print(predictions.to_string(index=False))


    # ===== 4. Interpret Results =====
    print("\n" + "-"*70)
    print("Business Interpretation:")
    print("-"*70)

    certain_no_churn = (predictions['conformal_set'] == '{0}').sum()
    certain_churn = (predictions['conformal_set'] == '{1}').sum()
    uncertain = (predictions['conformal_set'].str.contains('0, 1')).sum()

    print(f"\n  LOW PRIORITY:    {certain_no_churn} customers (certain no churn)")
    print(f"  HIGH PRIORITY:   {certain_churn} customers (certain churn - contact now!)")
    print(f"  MODERATE:        {uncertain} customers (uncertain - monitor closely)")


    # ===== 5. Evaluate Performance =====
    print("\n" + "="*70)
    print("[Step 4] Evaluating Model Performance")
    print("="*70 + "\n")

    test_data = generate_customer_data(50)
    X_test = test_data.drop('churned', axis=1)
    y_test = test_data['churned']

    metrics = pipeline.evaluate(X_test, y_test)  # No sme_id needed!

    print("Performance Metrics:")
    print("-"*70)
    print(f"  AUC:                 {metrics['auc']:.3f}")
    print(f"  Accuracy:            {metrics['accuracy']:.3f}")
    print(f"  F1 Score:            {metrics['f1_score']:.3f}")
    print(f"  Conformal Coverage:  {metrics['conformal_coverage']:.3f}  (target: 0.90)")
    print(f"  Mean Set Size:       {metrics['mean_set_size']:.2f}")


    # ===== Summary =====
    print("\n" + "="*70)
    print("Single-Entity Mode Complete!")
    print("="*70)
    print("\nKey Points:")
    print("  - Single-entity mode works with just ONE dataset")
    print("  - Pre-trained priors provide the statistical 'borrowed strength'")
    print("  - Accuracy may be 5-10% lower than multi-entity mode")
    print("  - As you grow (add stores/branches), switch to multi-entity mode")
    print("\nTo save and reload:")
    print("  pipeline.save('my_pipeline.pkl')")
    print("  pipeline = Pipeline.load('my_pipeline.pkl')")
    print("\n" + "="*70 + "\n")
