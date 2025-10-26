"""
Generate Baseline Comparison Data for Section 5

This script trains baseline models and compares them against SmallML's
hierarchical Bayesian approach.

Prerequisites:
- SME datasets created (Day 12)
- Hierarchical model trained (Day 12)
- Transfer learning priors extracted (Day 11)

Output:
- Table 5.2: Complete model comparison
- Per-SME performance statistics
- Statistical significance tests
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss
)
from scipy import stats
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("BASELINE COMPARISON FOR SECTION 5")
print("=" * 80)
print(f"Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# STEP 1: Load SME Datasets from CSVs
# =============================================================================

print("STEP 1: Loading SME datasets...")

project_root = Path(__file__).parent.parent
sme_dir = project_root / "data" / "sme_datasets"
models_dir = project_root / "models"
results_dir = project_root / "results" / "tables"
results_dir.mkdir(parents=True, exist_ok=True)

# Load metadata
with open(sme_dir / "metadata.json", 'r') as f:
    metadata = json.load(f)

J = metadata['n_smes']
n_per_sme = metadata['n_per_sme']
print(f"  J = {J} SMEs")
print(f"  n_j = {n_per_sme} customers per SME")

# Load each SME dataset from CSV
sme_datasets = []
for j in range(J):
    df = pd.read_csv(sme_dir / f"sme_{j}.csv")
    X = df.drop('churned', axis=1).values
    y = df['churned'].values

    # Handle missing values: fill NaN with 0 (mean imputation already done in harmonization)
    # Any remaining NaNs are from missing indicators
    X = np.nan_to_num(X, nan=0.0)

    sme_datasets.append({'X': X, 'y': y, 'df': df})
    print(f"  ✓ Loaded SME {j}: {X.shape[0]} samples, {X.shape[1]} features")

print(f"✓ Loaded {J} SME datasets\n")

# =============================================================================
# STEP 2: Create Train/Test Splits
# =============================================================================

print("STEP 2: Creating train/test splits...")

# Use 75/25 split for each SME
test_datasets = {}
train_datasets = {}

np.random.seed(42)

for j in range(J):
    n_j = len(sme_datasets[j]['X'])
    n_test = int(0.25 * n_j)  # 25% for testing

    # Random shuffle indices
    indices = np.random.permutation(n_j)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_full = sme_datasets[j]['X']
    y_full = sme_datasets[j]['y']

    test_datasets[j] = {
        'X': X_full[test_idx],
        'y': y_full[test_idx]
    }

    train_datasets[j] = {
        'X': X_full[train_idx],
        'y': y_full[train_idx]
    }

    print(f"  SME {j}: {len(train_idx)} train, {len(test_idx)} test")

print(f"✓ Created {J} train/test splits\n")

# =============================================================================
# STEP 3: Train Baseline 1 - Independent Models (No Pooling)
# =============================================================================

print("STEP 3: Training independent models (no pooling)...")

independent_results = {
    'auc': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'log_loss': []
}

for j in range(J):
    print(f"  Training independent model for SME {j}...")

    # Train logistic regression on just this SME's data
    lr = LogisticRegression(
        penalty='l2',
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    X_train = train_datasets[j]['X']
    y_train = train_datasets[j]['y']
    X_test = test_datasets[j]['X']
    y_test = test_datasets[j]['y']

    # Check if we have both classes in training data
    if len(np.unique(y_train)) < 2:
        print(f"    Warning: SME {j} has only one class in training data")
        # Use defaults for this SME
        independent_results['auc'].append(0.5)
        independent_results['accuracy'].append(0.5)
        independent_results['precision'].append(0.0)
        independent_results['recall'].append(0.0)
        independent_results['f1'].append(0.0)
        independent_results['log_loss'].append(1.0)
        continue

    # Fit model
    lr.fit(X_train, y_train)

    # Predict on test set
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    y_pred = lr.predict(X_test)

    # Compute metrics
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ll = log_loss(y_test, y_pred_proba)

        independent_results['auc'].append(auc)
        independent_results['accuracy'].append(acc)
        independent_results['precision'].append(prec)
        independent_results['recall'].append(rec)
        independent_results['f1'].append(f1)
        independent_results['log_loss'].append(ll)

        print(f"    AUC: {auc:.3f}, Accuracy: {acc:.3f}")
    except Exception as e:
        print(f"    Warning: Could not compute metrics for SME {j}: {e}")
        independent_results['auc'].append(0.5)
        independent_results['accuracy'].append(0.5)
        independent_results['precision'].append(0.0)
        independent_results['recall'].append(0.0)
        independent_results['f1'].append(0.0)
        independent_results['log_loss'].append(1.0)

print("✓ Independent models trained\n")

# =============================================================================
# STEP 4: Train Baseline 2 - Complete Pooling
# =============================================================================

print("STEP 4: Training complete pooling model...")

# Combine all training data from all SMEs
X_all_train = np.vstack([train_datasets[j]['X'] for j in range(J)])
y_all_train = np.hstack([train_datasets[j]['y'] for j in range(J)])

print(f"  Combined training data: {X_all_train.shape[0]} samples")

# Train single model on all data
lr_pooled = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

lr_pooled.fit(X_all_train, y_all_train)

# Evaluate on each SME's test set
complete_pooling_results = {
    'auc': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'log_loss': []
}

for j in range(J):
    X_test = test_datasets[j]['X']
    y_test = test_datasets[j]['y']

    y_pred_proba = lr_pooled.predict_proba(X_test)[:, 1]
    y_pred = lr_pooled.predict(X_test)

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ll = log_loss(y_test, y_pred_proba)

        complete_pooling_results['auc'].append(auc)
        complete_pooling_results['accuracy'].append(acc)
        complete_pooling_results['precision'].append(prec)
        complete_pooling_results['recall'].append(rec)
        complete_pooling_results['f1'].append(f1)
        complete_pooling_results['log_loss'].append(ll)
    except Exception as e:
        print(f"  Warning: Could not compute metrics for SME {j}: {e}")
        complete_pooling_results['auc'].append(0.5)
        complete_pooling_results['accuracy'].append(0.5)
        complete_pooling_results['precision'].append(0.0)
        complete_pooling_results['recall'].append(0.0)
        complete_pooling_results['f1'].append(0.0)
        complete_pooling_results['log_loss'].append(1.0)

print(f"  Average AUC across SMEs: {np.mean(complete_pooling_results['auc']):.3f}")
print("✓ Complete pooling model trained\n")

# =============================================================================
# STEP 5: Extract SmallML (Hierarchical) Results
# =============================================================================

print("STEP 5: Extracting SmallML hierarchical results...")

# Load posterior means
posterior_path = models_dir / "hierarchical" / "posterior_means.json"
with open(posterior_path, 'r') as f:
    posterior = json.load(f)

# Extract beta_j posterior means
# posterior_means.json has format: {beta_j_mean: [[...], [...], ...]}
beta_j_mean = np.array(posterior["beta_j_mean"])  # Shape: (J, p+1) with intercept
print(f"  Loaded posterior means: {beta_j_mean.shape}")

# Evaluate on each SME's test set
smallml_results = {
    'auc': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'log_loss': []
}

for j in range(J):
    X_test = test_datasets[j]['X']
    y_test = test_datasets[j]['y']

    # Compute predictions using posterior mean
    # Note: beta_j already includes intercept effect through hierarchical structure
    logit_p = X_test @ beta_j_mean[j]
    y_pred_proba = 1 / (1 + np.exp(-logit_p))

    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
    y_pred = (y_pred_proba > 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ll = log_loss(y_test, y_pred_proba)

        smallml_results['auc'].append(auc)
        smallml_results['accuracy'].append(acc)
        smallml_results['precision'].append(prec)
        smallml_results['recall'].append(rec)
        smallml_results['f1'].append(f1)
        smallml_results['log_loss'].append(ll)

        print(f"  SME {j}: AUC={auc:.3f}")
    except Exception as e:
        print(f"  Warning: Could not compute metrics for SME {j}: {e}")
        smallml_results['auc'].append(0.5)
        smallml_results['accuracy'].append(0.5)
        smallml_results['precision'].append(0.0)
        smallml_results['recall'].append(0.0)
        smallml_results['f1'].append(0.0)
        smallml_results['log_loss'].append(1.0)

print("✓ SmallML results extracted\n")

# =============================================================================
# STEP 6: Compute Summary Statistics
# =============================================================================

print("STEP 6: Computing summary statistics...")

# Create Table 5.2
table_5_2 = pd.DataFrame({
    'Model': ['Independent (No Pooling)', 'Complete Pooling', 'SmallML (Hierarchical)'],
    'AUC': [
        np.mean(independent_results['auc']),
        np.mean(complete_pooling_results['auc']),
        np.mean(smallml_results['auc'])
    ],
    'Accuracy': [
        np.mean(independent_results['accuracy']),
        np.mean(complete_pooling_results['accuracy']),
        np.mean(smallml_results['accuracy'])
    ],
    'Precision': [
        np.mean(independent_results['precision']),
        np.mean(complete_pooling_results['precision']),
        np.mean(smallml_results['precision'])
    ],
    'Recall': [
        np.mean(independent_results['recall']),
        np.mean(complete_pooling_results['recall']),
        np.mean(smallml_results['recall'])
    ],
    'F1-Score': [
        np.mean(independent_results['f1']),
        np.mean(complete_pooling_results['f1']),
        np.mean(smallml_results['f1'])
    ],
    'Log Loss': [
        np.mean(independent_results['log_loss']),
        np.mean(complete_pooling_results['log_loss']),
        np.mean(smallml_results['log_loss'])
    ]
})

print("\n" + "=" * 80)
print("TABLE 5.2: Model Performance Comparison")
print("=" * 80)
print(table_5_2.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print("=" * 80)

# Per-SME statistics
per_sme_stats = pd.DataFrame({
    'Method': ['Independent', 'Complete Pooling', 'SmallML'],
    'Mean AUC': [
        np.mean(independent_results['auc']),
        np.mean(complete_pooling_results['auc']),
        np.mean(smallml_results['auc'])
    ],
    'Std Dev': [
        np.std(independent_results['auc']),
        np.std(complete_pooling_results['auc']),
        np.std(smallml_results['auc'])
    ],
    'Min AUC': [
        np.min(independent_results['auc']),
        np.min(complete_pooling_results['auc']),
        np.min(smallml_results['auc'])
    ],
    'Max AUC': [
        np.max(independent_results['auc']),
        np.max(complete_pooling_results['auc']),
        np.max(smallml_results['auc'])
    ]
})

print("\nPer-SME Performance Variability:")
print(per_sme_stats.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# =============================================================================
# STEP 7: Statistical Significance Tests
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: Computing statistical significance...")
print("=" * 80)

# Paired t-tests (since we're comparing on same SMEs)
t_stat_ind, p_value_ind = stats.ttest_rel(
    smallml_results['auc'],
    independent_results['auc']
)

t_stat_pool, p_value_pool = stats.ttest_rel(
    smallml_results['auc'],
    complete_pooling_results['auc']
)

print(f"\nStatistical Significance Tests (Paired t-tests):")
print(f"  SmallML vs. Independent:")
print(f"    Mean difference: {np.mean(smallml_results['auc']) - np.mean(independent_results['auc']):.4f}")
print(f"    t-statistic: {t_stat_ind:.4f}")
print(f"    p-value: {p_value_ind:.6f}")
print(f"    Significant? {'YES (p<0.05)' if p_value_ind < 0.05 else 'NO'}")

print(f"\n  SmallML vs. Complete Pooling:")
print(f"    Mean difference: {np.mean(smallml_results['auc']) - np.mean(complete_pooling_results['auc']):.4f}")
print(f"    t-statistic: {t_stat_pool:.4f}")
print(f"    p-value: {p_value_pool:.6f}")
print(f"    Significant? {'YES (p<0.05)' if p_value_pool < 0.05 else 'NO'}")

# =============================================================================
# STEP 8: Save Results
# =============================================================================

print("\n" + "=" * 80)
print("STEP 8: Saving results...")
print("=" * 80)

# Save to CSV
table_5_2.to_csv(results_dir / 'table_5_2_comparison.csv', index=False, float_format='%.4f')
per_sme_stats.to_csv(results_dir / 'table_5_2_per_sme_statistics.csv', index=False, float_format='%.4f')

# Save statistical test results
stat_tests = pd.DataFrame({
    'Comparison': ['SmallML vs. Independent', 'SmallML vs. Complete Pooling'],
    't-statistic': [t_stat_ind, t_stat_pool],
    'p-value': [p_value_ind, p_value_pool],
    'Significant (p<0.05)': [p_value_ind < 0.05, p_value_pool < 0.05]
})
stat_tests.to_csv(results_dir / 'table_5_2_statistical_tests.csv', index=False, float_format='%.6f')

# Save complete results for reference
complete_results = {
    'independent': independent_results,
    'complete_pooling': complete_pooling_results,
    'smallml': smallml_results
}

with open(project_root / 'results' / 'baseline_comparison_results.pkl', 'wb') as f:
    pickle.dump(complete_results, f)

print("✓ Results saved to:")
print(f"  - {results_dir / 'table_5_2_comparison.csv'}")
print(f"  - {results_dir / 'table_5_2_per_sme_statistics.csv'}")
print(f"  - {results_dir / 'table_5_2_statistical_tests.csv'}")
print(f"  - {project_root / 'results' / 'baseline_comparison_results.pkl'}")

# =============================================================================
# STEP 9: Summary for Section 5
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY FOR SECTION 5 DATA COLLECTION TEMPLATE")
print("=" * 80)

print("\n✓ TABLE 5.2: Model Performance Comparison")
print(f"  Independent: AUC={np.mean(independent_results['auc']):.3f}")
print(f"  Complete Pooling: AUC={np.mean(complete_pooling_results['auc']):.3f}")
print(f"  SmallML: AUC={np.mean(smallml_results['auc']):.3f}")

print("\n✓ Per-SME Performance Variability")
print(f"  SmallML: Mean={np.mean(smallml_results['auc']):.3f}, Std={np.std(smallml_results['auc']):.3f}")
print(f"  Independent: Mean={np.mean(independent_results['auc']):.3f}, Std={np.std(independent_results['auc']):.3f}")
print(f"  Range: [{np.min(smallml_results['auc']):.3f}, {np.max(smallml_results['auc']):.3f}]")

print("\n✓ Statistical Significance")
print(f"  SmallML vs. Independent: p={p_value_ind:.4f}")
print(f"  SmallML vs. Complete: p={p_value_pool:.4f}")

improvement_ind = ((np.mean(smallml_results['auc']) - np.mean(independent_results['auc'])) /
                   np.mean(independent_results['auc']) * 100)
improvement_pool = ((np.mean(smallml_results['auc']) - np.mean(complete_pooling_results['auc'])) /
                    np.mean(complete_pooling_results['auc']) * 100)

print("\n✓ Improvements")
print(f"  SmallML vs. Independent: {improvement_ind:+.1f}%")
print(f"  SmallML vs. Complete Pooling: {improvement_pool:+.1f}%")

print("\n" + "=" * 80)
print(f"Completed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
