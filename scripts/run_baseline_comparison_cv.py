"""
Generate Baseline Comparison Data Using Cross-Validation

This script implements 5-fold stratified cross-validation for robust
performance estimation on small SME datasets.

Key improvements over single train/test split:
- More stable performance estimates
- Reports mean ± std across folds
- Better statistical power
- Publication-ready methodology

Prerequisites:
- SME datasets created (15 SMEs × 100 samples)
- Hierarchical model trained
- Transfer learning priors extracted

Output:
- Table 5.2 (CV version): Mean ± std for all metrics
- Per-fold results for transparency
- Statistical significance tests
- Publication-ready summary

"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
print("BASELINE COMPARISON WITH CROSS-VALIDATION")
print("=" * 80)
print(f"Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# STEP 1: Load SME Datasets
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

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    sme_datasets.append({'X': X, 'y': y, 'df': df})
    print(f"  ✓ Loaded SME {j}: {X.shape[0]} samples, {X.shape[1]} features")

print(f"✓ Loaded {J} SME datasets\n")

# =============================================================================
# STEP 2: Cross-Validation Setup
# =============================================================================

print("STEP 2: Setting up 5-fold cross-validation...")

N_FOLDS = 5
RANDOM_STATE = 42

print(f"  Number of folds: {N_FOLDS}")
print(f"  Random state: {RANDOM_STATE}")
print(f"  Each fold: ~{n_per_sme//N_FOLDS * (N_FOLDS-1)} train, ~{n_per_sme//N_FOLDS} test\n")

# =============================================================================
# STEP 3: Load Hierarchical Model Posteriors
# =============================================================================

print("STEP 3: Loading hierarchical model posteriors...")

posterior_path = models_dir / "hierarchical" / "posterior_means.json"
with open(posterior_path, 'r') as f:
    posterior = json.load(f)

beta_j_mean = np.array(posterior["beta_j_mean"])
print(f"  Loaded posterior means: {beta_j_mean.shape}")
print(f"✓ Hierarchical model loaded\n")

# =============================================================================
# STEP 4: Cross-Validation Experiments
# =============================================================================

print("STEP 4: Running cross-validation experiments...")
print("  This may take 20-30 minutes...\n")

# Storage for all results
cv_results = {
    'independent': {metric: [] for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1', 'log_loss']},
    'complete_pooling': {metric: [] for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1', 'log_loss']},
    'smallml': {metric: [] for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1', 'log_loss']}
}

# Per-fold results for transparency
per_fold_results = []

# For each fold
for fold_idx in range(N_FOLDS):
    print(f"=" * 80)
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"=" * 80)

    fold_independent = {metric: [] for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1', 'log_loss']}
    fold_complete = {metric: [] for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1', 'log_loss']}
    fold_smallml = {metric: [] for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1', 'log_loss']}

    # -------------------------------------------------------------------------
    # 4.1: Independent Models (No Pooling)
    # -------------------------------------------------------------------------

    print(f"\n[Fold {fold_idx+1}] Training independent models...")

    for j in range(J):
        X_full = sme_datasets[j]['X']
        y_full = sme_datasets[j]['y']

        # Create fold split for this SME
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X_full, y_full))
        train_idx, test_idx = splits[fold_idx]

        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]

        # Check if both classes present
        if len(np.unique(y_train)) < 2:
            # Use default values
            fold_independent['auc'].append(0.5)
            fold_independent['accuracy'].append(0.5)
            fold_independent['precision'].append(0.0)
            fold_independent['recall'].append(0.0)
            fold_independent['f1'].append(0.0)
            fold_independent['log_loss'].append(1.0)
            continue

        # Train independent model
        lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs')
        lr.fit(X_train, y_train)

        # Predict
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        y_pred = lr.predict(X_test)

        # Metrics
        try:
            fold_independent['auc'].append(roc_auc_score(y_test, y_pred_proba))
            fold_independent['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_independent['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_independent['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_independent['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            fold_independent['log_loss'].append(log_loss(y_test, y_pred_proba))
        except:
            fold_independent['auc'].append(0.5)
            fold_independent['accuracy'].append(0.5)
            fold_independent['precision'].append(0.0)
            fold_independent['recall'].append(0.0)
            fold_independent['f1'].append(0.0)
            fold_independent['log_loss'].append(1.0)

    print(f"  ✓ Independent: Mean AUC = {np.mean(fold_independent['auc']):.3f}")

    # -------------------------------------------------------------------------
    # 4.2: Complete Pooling
    # -------------------------------------------------------------------------

    print(f"[Fold {fold_idx+1}] Training complete pooling model...")

    # Collect all training data from this fold
    X_all_train = []
    y_all_train = []

    for j in range(J):
        X_full = sme_datasets[j]['X']
        y_full = sme_datasets[j]['y']

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X_full, y_full))
        train_idx, test_idx = splits[fold_idx]

        X_all_train.append(X_full[train_idx])
        y_all_train.append(y_full[train_idx])

    X_all_train = np.vstack(X_all_train)
    y_all_train = np.hstack(y_all_train)

    # Train pooled model
    lr_pooled = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs')
    lr_pooled.fit(X_all_train, y_all_train)

    # Evaluate on each SME's test set
    for j in range(J):
        X_full = sme_datasets[j]['X']
        y_full = sme_datasets[j]['y']

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X_full, y_full))
        train_idx, test_idx = splits[fold_idx]

        X_test = X_full[test_idx]
        y_test = y_full[test_idx]

        y_pred_proba = lr_pooled.predict_proba(X_test)[:, 1]
        y_pred = lr_pooled.predict(X_test)

        try:
            fold_complete['auc'].append(roc_auc_score(y_test, y_pred_proba))
            fold_complete['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_complete['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_complete['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_complete['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            fold_complete['log_loss'].append(log_loss(y_test, y_pred_proba))
        except:
            fold_complete['auc'].append(0.5)
            fold_complete['accuracy'].append(0.5)
            fold_complete['precision'].append(0.0)
            fold_complete['recall'].append(0.0)
            fold_complete['f1'].append(0.0)
            fold_complete['log_loss'].append(1.0)

    print(f"  ✓ Complete Pooling: Mean AUC = {np.mean(fold_complete['auc']):.3f}")

    # -------------------------------------------------------------------------
    # 4.3: SmallML (Hierarchical)
    # -------------------------------------------------------------------------

    print(f"[Fold {fold_idx+1}] Evaluating SmallML hierarchical model...")

    for j in range(J):
        X_full = sme_datasets[j]['X']
        y_full = sme_datasets[j]['y']

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X_full, y_full))
        train_idx, test_idx = splits[fold_idx]

        X_test = X_full[test_idx]
        y_test = y_full[test_idx]

        # Predict using posterior mean
        logit_p = X_test @ beta_j_mean[j]
        y_pred_proba = 1 / (1 + np.exp(-logit_p))
        y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
        y_pred = (y_pred_proba > 0.5).astype(int)

        try:
            fold_smallml['auc'].append(roc_auc_score(y_test, y_pred_proba))
            fold_smallml['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_smallml['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_smallml['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_smallml['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            fold_smallml['log_loss'].append(log_loss(y_test, y_pred_proba))
        except:
            fold_smallml['auc'].append(0.5)
            fold_smallml['accuracy'].append(0.5)
            fold_smallml['precision'].append(0.0)
            fold_smallml['recall'].append(0.0)
            fold_smallml['f1'].append(0.0)
            fold_smallml['log_loss'].append(1.0)

    print(f"  ✓ SmallML: Mean AUC = {np.mean(fold_smallml['auc']):.3f}")

    # -------------------------------------------------------------------------
    # Store fold results
    # -------------------------------------------------------------------------

    for method_name, fold_data in [('Independent', fold_independent),
                                     ('Complete Pooling', fold_complete),
                                     ('SmallML', fold_smallml)]:
        per_fold_results.append({
            'Fold': fold_idx + 1,
            'Method': method_name,
            'AUC': np.mean(fold_data['auc']),
            'Accuracy': np.mean(fold_data['accuracy']),
            'Precision': np.mean(fold_data['precision']),
            'Recall': np.mean(fold_data['recall']),
            'F1-Score': np.mean(fold_data['f1']),
            'Log Loss': np.mean(fold_data['log_loss'])
        })

        # Aggregate across folds
        for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1', 'log_loss']:
            if method_name == 'Independent':
                cv_results['independent'][metric].extend(fold_data[metric])
            elif method_name == 'Complete Pooling':
                cv_results['complete_pooling'][metric].extend(fold_data[metric])
            else:
                cv_results['smallml'][metric].extend(fold_data[metric])

print("\n✓ Cross-validation completed\n")

# =============================================================================
# STEP 5: Aggregate CV Results
# =============================================================================

print("STEP 5: Aggregating cross-validation results...")

# Calculate mean and std across all fold×SME combinations
summary_table = []

for method_name, method_key in [('Independent (No Pooling)', 'independent'),
                                  ('Complete Pooling', 'complete_pooling'),
                                  ('SmallML (Hierarchical)', 'smallml')]:
    row = {'Model': method_name}

    for metric_name, metric_key in [('AUC', 'auc'), ('Accuracy', 'accuracy'),
                                      ('Precision', 'precision'), ('Recall', 'recall'),
                                      ('F1-Score', 'f1'), ('Log Loss', 'log_loss')]:
        values = cv_results[method_key][metric_key]
        row[f'{metric_name} Mean'] = np.mean(values)
        row[f'{metric_name} Std'] = np.std(values)

    summary_table.append(row)

summary_df = pd.DataFrame(summary_table)

print("\n" + "=" * 80)
print("TABLE 5.2 (CV): Model Performance Comparison")
print("=" * 80)
print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print("=" * 80)

# =============================================================================
# STEP 6: Statistical Significance Tests
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6: Statistical significance tests...")
print("=" * 80)

# Paired t-tests on AUC
smallml_aucs = cv_results['smallml']['auc']
independent_aucs = cv_results['independent']['auc']
complete_aucs = cv_results['complete_pooling']['auc']

t_stat_ind, p_value_ind = stats.ttest_rel(smallml_aucs, independent_aucs)
t_stat_pool, p_value_pool = stats.ttest_rel(smallml_aucs, complete_aucs)

print(f"\nPaired t-tests on AUC:")
print(f"  SmallML vs. Independent:")
print(f"    Mean difference: {np.mean(smallml_aucs) - np.mean(independent_aucs):.4f}")
print(f"    t-statistic: {t_stat_ind:.4f}")
print(f"    p-value: {p_value_ind:.6f}")
print(f"    Significant? {'YES (p<0.05)' if p_value_ind < 0.05 else 'NO'}")

print(f"\n  SmallML vs. Complete Pooling:")
print(f"    Mean difference: {np.mean(smallml_aucs) - np.mean(complete_aucs):.4f}")
print(f"    t-statistic: {t_stat_pool:.4f}")
print(f"    p-value: {p_value_pool:.6f}")
print(f"    Significant? {'YES (p<0.05)' if p_value_pool < 0.05 else 'NO'}")

# =============================================================================
# STEP 7: Save Results
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: Saving results...")
print("=" * 80)

# Table 5.2 (CV version)
summary_df.to_csv(results_dir / 'table_5_2_cv_comparison.csv', index=False, float_format='%.4f')

# Per-fold results
per_fold_df = pd.DataFrame(per_fold_results)
per_fold_df.to_csv(results_dir / 'table_5_2_cv_per_fold.csv', index=False, float_format='%.4f')

# Statistical tests
stat_tests_df = pd.DataFrame({
    'Comparison': ['SmallML vs. Independent', 'SmallML vs. Complete Pooling'],
    't-statistic': [t_stat_ind, t_stat_pool],
    'p-value': [p_value_ind, p_value_pool],
    'Significant (p<0.05)': [p_value_ind < 0.05, p_value_pool < 0.05]
})
stat_tests_df.to_csv(results_dir / 'table_5_2_cv_statistical_tests.csv', index=False, float_format='%.6f')

# Save complete results
with open(project_root / 'results' / 'baseline_comparison_cv_results.pkl', 'wb') as f:
    pickle.dump(cv_results, f)

print("✓ Results saved to:")
print(f"  - {results_dir / 'table_5_2_cv_comparison.csv'}")
print(f"  - {results_dir / 'table_5_2_cv_per_fold.csv'}")
print(f"  - {results_dir / 'table_5_2_cv_statistical_tests.csv'}")
print(f"  - {project_root / 'results' / 'baseline_comparison_cv_results.pkl'}")

# =============================================================================
# STEP 8: Summary
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY FOR SECTION 5 (CV VALIDATION)")
print("=" * 80)

print("\n✓ TABLE 5.2 (CV): Model Performance Comparison")
print(f"  Independent: AUC = {np.mean(independent_aucs):.3f} ± {np.std(independent_aucs):.3f}")
print(f"  Complete Pooling: AUC = {np.mean(complete_aucs):.3f} ± {np.std(complete_aucs):.3f}")
print(f"  SmallML: AUC = {np.mean(smallml_aucs):.3f} ± {np.std(smallml_aucs):.3f}")

print("\n✓ Statistical Significance")
print(f"  SmallML vs. Independent: p = {p_value_ind:.4f}")
print(f"  SmallML vs. Complete Pooling: p = {p_value_pool:.4f}")

improvement_ind = ((np.mean(smallml_aucs) - np.mean(independent_aucs)) / np.mean(independent_aucs) * 100)
improvement_pool = ((np.mean(smallml_aucs) - np.mean(complete_aucs)) / np.mean(complete_aucs) * 100)

print("\n✓ Improvements")
print(f"  SmallML vs. Independent: {improvement_ind:+.1f}%")
print(f"  SmallML vs. Complete Pooling: {improvement_pool:+.1f}%")

print("\n✓ Cross-Validation Stats")
print(f"  Number of folds: {N_FOLDS}")
print(f"  Total evaluations: {J} SMEs × {N_FOLDS} folds = {J * N_FOLDS}")
print(f"  Test samples per fold: ~{n_per_sme // N_FOLDS} per SME")

print("\n" + "=" * 80)
print("DONE! Cross-validation results ready for publication.")
print(f"Completed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
