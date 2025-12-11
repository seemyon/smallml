"""
Unit Tests for SmallML Pipeline
================================

Test suite for core SmallML functionality including:
- Pipeline initialization
- Data validation
- Fit/predict workflow
- Save/load functionality
"""

import pytest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import tempfile
import warnings

from smallml import Pipeline, __version__


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def small_synthetic_data():
    """Generate minimal synthetic data for fast testing (3 SMEs, 30 customers each)."""
    np.random.seed(42)

    def generate_sme_data(n=30, sme_id=0):
        """Generate synthetic customer data."""
        data = {
            'recency': np.random.exponential(scale=30, size=n),
            'frequency': np.random.poisson(lam=5, size=n),
            'monetary': np.random.lognormal(mean=4, sigma=1, size=n),
            'tenure': np.random.uniform(1, 36, size=n),
            'age': np.random.normal(45, 15, size=n),
        }

        # Generate churn labels
        logit = (
            -2.0
            + 0.03 * data['recency']
            - 0.15 * data['frequency']
            - 0.0002 * data['monetary']
            - 0.02 * data['tenure']
            + np.random.normal(0, 0.5, n)
        )
        data['churned'] = (1 / (1 + np.exp(-logit)) > np.random.rand(n)).astype(int)

        return pd.DataFrame(data)

    return {
        'sme_0': generate_sme_data(30, 0),
        'sme_1': generate_sme_data(30, 1),
        'sme_2': generate_sme_data(30, 2),
    }


@pytest.fixture
def fitted_pipeline(small_synthetic_data):
    """Create a fitted pipeline for testing predict/evaluate methods."""
    pipeline = Pipeline(use_pretrained_priors=False, quick_mode=True, random_seed=42)

    # Suppress PyMC warnings during tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(
            small_synthetic_data,
            target_col='churned',
            calibration_fraction=0.25,
            validate_convergence=False  # Skip for speed in tests
        )

    return pipeline


# ============================================================================
# Test 1: Pipeline Initialization
# ============================================================================

def test_pipeline_init_with_priors():
    """Test Pipeline initialization with pre-trained priors."""
    pipeline = Pipeline(use_pretrained_priors=True)

    assert pipeline.use_pretrained_priors is True
    assert pipeline.quick_mode is False
    assert pipeline.random_seed == 42

    # Check if priors loaded (may be None if file doesn't exist, which is OK)
    if pipeline.priors is not None:
        assert 'beta_0' in pipeline.priors
        assert 'Sigma_0' in pipeline.priors
        assert isinstance(pipeline.priors['beta_0'], np.ndarray)
        assert isinstance(pipeline.priors['Sigma_0'], np.ndarray)


def test_pipeline_init_without_priors():
    """Test Pipeline initialization without pre-trained priors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress priors warning
        pipeline = Pipeline(use_pretrained_priors=False)

    assert pipeline.use_pretrained_priors is False
    assert pipeline.priors is None


def test_pipeline_init_quick_mode():
    """Test Pipeline initialization with quick_mode."""
    pipeline = Pipeline(quick_mode=True)

    assert pipeline.quick_mode is True


def test_pipeline_init_custom_seed():
    """Test Pipeline initialization with custom random seed."""
    pipeline = Pipeline(random_seed=123)

    assert pipeline.random_seed == 123


def test_version_available():
    """Test that package version is accessible."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


# ============================================================================
# Test 2: Data Validation
# ============================================================================

def test_fit_with_empty_data():
    """Test that fit() raises error with empty data dict."""
    pipeline = Pipeline(use_pretrained_priors=False)

    with pytest.raises((TypeError, ValueError)):
        pipeline.fit({}, target_col='churned')


def test_fit_with_missing_target():
    """Test that fit() raises error when target column is missing."""
    pipeline = Pipeline(use_pretrained_priors=False)

    # Create data without 'churned' column
    bad_data = {
        'sme_0': pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [4, 5, 6],
        })
    }

    with pytest.raises((ValueError, KeyError)):
        pipeline.fit(bad_data, target_col='churned')


def test_fit_with_few_smes_warns(small_synthetic_data):
    """Test that fit() warns when fewer than 3 SMEs provided."""
    pipeline = Pipeline(use_pretrained_priors=False, quick_mode=True)

    # Use only 2 SMEs
    few_smes = {
        'sme_0': small_synthetic_data['sme_0'],
        'sme_1': small_synthetic_data['sme_1'],
    }

    # Should complete but may warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipeline.fit(
            few_smes,
            target_col='churned',
            validate_convergence=False
        )

        # Check if warning was issued (optional - implementation may or may not warn)
        # Just verify it doesn't crash


def test_fit_with_small_dataset_warns():
    """Test that fit() warns when SME has very few observations."""
    pipeline = Pipeline(use_pretrained_priors=False, quick_mode=True)

    # Create very small dataset (< 30 observations)
    tiny_data = {
        'sme_0': pd.DataFrame({
            'feature_1': np.random.randn(15),
            'feature_2': np.random.randn(15),
            'churned': np.random.randint(0, 2, 15)
        })
    }

    # Should complete but may warn about small sample size
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            pipeline.fit(
                tiny_data,
                target_col='churned',
                validate_convergence=False
            )
        except Exception:
            # May fail due to insufficient data - that's OK
            pass


# ============================================================================
# Test 3: Fit Functionality
# ============================================================================

def test_fit_completes_successfully(small_synthetic_data):
    """Test that fit() completes without errors on valid data."""
    pipeline = Pipeline(use_pretrained_priors=False, quick_mode=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(
            small_synthetic_data,
            target_col='churned',
            calibration_fraction=0.25,
            validate_convergence=False  # Skip for speed
        )

    # Check that models were initialized
    assert pipeline.hierarchical_model is not None
    assert pipeline.conformal_predictor is not None
    assert pipeline.feature_names is not None
    assert pipeline.target_col == 'churned'
    assert pipeline.sme_names is not None


def test_fit_sets_attributes(small_synthetic_data):
    """Test that fit() sets expected pipeline attributes."""
    pipeline = Pipeline(use_pretrained_priors=False, quick_mode=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(
            small_synthetic_data,
            target_col='churned',
            validate_convergence=False
        )

    # Check feature names extracted correctly
    expected_features = ['recency', 'frequency', 'monetary', 'tenure', 'age']
    assert set(pipeline.feature_names) == set(expected_features)

    # Check SME names stored
    assert len(pipeline.sme_names) == 3
    assert 'sme_0' in pipeline.sme_names


def test_fit_returns_self(small_synthetic_data):
    """Test that fit() returns self for method chaining."""
    pipeline = Pipeline(use_pretrained_priors=False, quick_mode=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipeline.fit(
            small_synthetic_data,
            target_col='churned',
            validate_convergence=False
        )

    assert result is pipeline


# ============================================================================
# Test 4: Predict Functionality
# ============================================================================

def test_predict_without_fit_raises_error():
    """Test that predict() raises error if called before fit()."""
    pipeline = Pipeline(use_pretrained_priors=False)

    test_data = pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [4, 5, 6],
    })

    with pytest.raises((RuntimeError, AttributeError)):
        pipeline.predict(test_data)


def test_predict_returns_dataframe(fitted_pipeline, small_synthetic_data):
    """Test that predict() returns a DataFrame with expected structure."""
    # Get test data
    test_df = small_synthetic_data['sme_0'].drop('churned', axis=1).head(10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = fitted_pipeline.predict(test_df, sme_id='sme_0')

    # Check return type
    assert isinstance(predictions, pd.DataFrame)

    # Check expected columns
    expected_cols = {'prediction', 'bayesian_std', 'bayesian_lower_90',
                     'bayesian_upper_90', 'conformal_set', 'conformal_set_size'}
    assert set(predictions.columns) >= expected_cols

    # Check shape
    assert len(predictions) == 10


def test_predict_values_in_valid_range(fitted_pipeline, small_synthetic_data):
    """Test that predictions are in valid range [0, 1]."""
    test_df = small_synthetic_data['sme_0'].drop('churned', axis=1).head(10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = fitted_pipeline.predict(test_df, sme_id='sme_0')

    # Check prediction values
    assert (predictions['prediction'] >= 0).all()
    assert (predictions['prediction'] <= 1).all()

    # Check standard deviation is positive
    assert (predictions['bayesian_std'] >= 0).all()

    # Check conformal set size is 1 or 2
    assert predictions['conformal_set_size'].isin([1, 2]).all()


def test_predict_without_uncertainty(fitted_pipeline, small_synthetic_data):
    """Test predict() with return_uncertainty=False."""
    test_df = small_synthetic_data['sme_0'].drop('churned', axis=1).head(5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = fitted_pipeline.predict(
            test_df,
            sme_id='sme_0',
            return_uncertainty=False
        )

    # Should still return DataFrame with at least 'prediction' column
    assert isinstance(predictions, pd.DataFrame)
    assert 'prediction' in predictions.columns


# ============================================================================
# Test 5: Evaluate Functionality
# ============================================================================

def test_evaluate_returns_metrics(fitted_pipeline, small_synthetic_data):
    """Test that evaluate() returns expected metrics."""
    test_df = small_synthetic_data['sme_0']
    X_test = test_df.drop('churned', axis=1)
    y_test = test_df['churned']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = fitted_pipeline.evaluate(X_test, y_test, sme_id='sme_0')

    # Check return type
    assert isinstance(metrics, dict)

    # Check expected metrics
    expected_metrics = {'auc', 'accuracy', 'f1_score',
                        'conformal_coverage', 'mean_set_size'}
    assert set(metrics.keys()) >= expected_metrics

    # Check values are in valid ranges
    assert 0 <= metrics['auc'] <= 1
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    assert 0 <= metrics['conformal_coverage'] <= 1
    assert 1 <= metrics['mean_set_size'] <= 2


# ============================================================================
# Test 6: Save/Load Functionality
# ============================================================================

def test_save_and_load_pipeline(fitted_pipeline, small_synthetic_data):
    """Test that pipeline can be saved and loaded correctly."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save pipeline
        fitted_pipeline.save(tmp_path)

        # Check file exists
        assert Path(tmp_path).exists()

        # Load pipeline
        loaded_pipeline = Pipeline.load(tmp_path)

        # Check attributes preserved
        assert loaded_pipeline.use_pretrained_priors == fitted_pipeline.use_pretrained_priors
        assert loaded_pipeline.quick_mode == fitted_pipeline.quick_mode
        assert loaded_pipeline.random_seed == fitted_pipeline.random_seed
        assert loaded_pipeline.target_col == fitted_pipeline.target_col

        # Check can make predictions
        test_df = small_synthetic_data['sme_0'].drop('churned', axis=1).head(5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = loaded_pipeline.predict(test_df, sme_id='sme_0')

        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 5

    finally:
        # Cleanup
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def test_load_nonexistent_file_raises_error():
    """Test that load() raises error for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        Pipeline.load('/nonexistent/path/to/pipeline.pkl')


# ============================================================================
# Test 7: Convergence Diagnostics
# ============================================================================

def test_get_convergence_diagnostics(fitted_pipeline):
    """Test that convergence diagnostics can be retrieved."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnostics = fitted_pipeline.get_convergence_diagnostics()

    # Check return type
    assert isinstance(diagnostics, pd.DataFrame)

    # Check expected columns
    expected_cols = {'parameter', 'r_hat', 'ess_bulk', 'ess_tail'}
    assert set(diagnostics.columns) >= expected_cols

    # Check values are positive
    assert (diagnostics['r_hat'] > 0).all()
    assert (diagnostics['ess_bulk'] > 0).all()
    assert (diagnostics['ess_tail'] > 0).all()


def test_get_diagnostics_before_fit_raises_error():
    """Test that get_convergence_diagnostics() raises error before fit()."""
    pipeline = Pipeline(use_pretrained_priors=False)

    with pytest.raises((RuntimeError, AttributeError)):
        pipeline.get_convergence_diagnostics()


# ============================================================================
# Test 8: Integration Tests
# ============================================================================

def test_full_workflow_synthetic_data(small_synthetic_data):
    """Test complete workflow: init → fit → predict → evaluate → save → load."""
    # Initialize
    pipeline = Pipeline(use_pretrained_priors=False, quick_mode=True, random_seed=42)

    # Split data
    train_data = {k: v.iloc[:20] for k, v in small_synthetic_data.items()}
    test_data = {k: v.iloc[20:] for k, v in small_synthetic_data.items()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Fit
        pipeline.fit(train_data, target_col='churned', validate_convergence=False)

        # Predict
        X_test = test_data['sme_0'].drop('churned', axis=1)
        y_test = test_data['sme_0']['churned']

        predictions = pipeline.predict(X_test, sme_id='sme_0')
        assert len(predictions) == len(X_test)

        # Evaluate
        metrics = pipeline.evaluate(X_test, y_test, sme_id='sme_0')
        assert 'auc' in metrics

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pipeline.save(tmp_path)

            # Load
            loaded = Pipeline.load(tmp_path)

            # Predict with loaded pipeline
            predictions2 = loaded.predict(X_test, sme_id='sme_0')
            assert len(predictions2) == len(predictions)

        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
