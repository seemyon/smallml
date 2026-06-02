"""
Unit tests for the domain-agnostic SmallML pipeline.

Covers:
- Pipeline init / validation
- Multi-entity binary, regression, and count tasks (domain-agnosticism)
- The standard five-field output contract (identical across tasks)
- Minimum-data enforcement (reference size, J, n_j)
- Feature-space alignment (intersection + warning)
- Layer 1 PriorExtractor
- Save / load and convergence diagnostics
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smallml import Pipeline, PriorExtractor, __version__
from smallml.pipeline import CONTRACT_COLUMNS

RNG = np.random.RandomState(42)


# ============================================================================
# Synthetic data helpers
# ============================================================================


def _binary_frame(n, shift=0.0, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n) + shift
    b = rng.randn(n)
    c = rng.randn(n)
    logit = 1.4 * a - 0.7 * b + 0.3 * c - 0.2
    y = (1 / (1 + np.exp(-logit)) > rng.rand(n)).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "outcome": y})


def _regression_frame(n, shift=0.0, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n) + shift
    b = rng.randn(n)
    c = rng.randn(n)
    y = 2.0 * a - 1.0 * b + 0.5 * c + rng.randn(n) * 0.4
    return pd.DataFrame({"a": a, "b": b, "c": c, "outcome": y})


def _count_frame(n, shift=0.0, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n) + shift
    b = rng.randn(n)
    y = rng.poisson(np.exp(0.5 * a - 0.3 * b + 0.2))
    return pd.DataFrame({"a": a, "b": b, "outcome": y})


def _multi_entity(maker, n_entities=6, n_each=35):
    frames = []
    for k in range(n_entities):
        df = maker(n_each, shift=0.2 * k, seed=100 + k)
        df["group"] = f"g{k}"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _fit_pipeline(task, maker, **kwargs):
    target = _multi_entity(maker)
    reference = maker(1500, seed=999)
    pipe = Pipeline(task=task, quick_mode=True, random_seed=42, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(
            target,
            target_col="outcome",
            entity_col="group",
            reference_data=reference,
            validate_convergence=False,
        )
    return pipe, target


@pytest.fixture(scope="module")
def binary_pipeline():
    return _fit_pipeline("binary", _binary_frame)


# ============================================================================
# Initialization
# ============================================================================


def test_version_available():
    assert isinstance(__version__, str) and len(__version__) > 0


def test_invalid_task_raises():
    with pytest.raises(ValueError):
        Pipeline(task="multiclass")


def test_invalid_confidence_level_raises():
    with pytest.raises(ValueError):
        Pipeline(confidence_level=1.5)


def test_default_task_is_binary():
    assert Pipeline().task == "binary"


# ============================================================================
# Binary task + output contract
# ============================================================================


def test_binary_fit_sets_attributes(binary_pipeline):
    pipe, _ = binary_pipeline
    assert pipe.hierarchical_model is not None
    assert pipe.conformal_predictor is not None
    assert pipe.target_col == "outcome"
    assert len(pipe.entity_names) == 6
    assert pipe.feature_names is not None


def test_binary_predict_contract(binary_pipeline):
    pipe, target = binary_pipeline
    X = target[target.group == "g0"].drop(columns=["outcome", "group"]).head(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = pipe.predict(X, entity_id="g0")

    assert list(preds.columns) == CONTRACT_COLUMNS
    assert len(preds) == 10
    # Probabilities in [0, 1]
    assert (preds["point_prediction"] >= 0).all()
    assert (preds["point_prediction"] <= 1).all()
    # Conformal sets are well-formed
    assert preds["conformal_set"].isin(["{0}", "{1}", "{0, 1}", "{}"]).all()
    # Confidence flags
    assert preds["confidence_flag"].isin(["HIGH", "UNCERTAIN"]).all()
    # Posterior samples present per row
    samples = np.asarray(preds["posterior_distribution"].iloc[0])
    assert samples.ndim == 1 and samples.size > 0


def test_binary_evaluate(binary_pipeline):
    pipe, target = binary_pipeline
    sub = target[target.group == "g0"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = pipe.evaluate(
            sub.drop(columns=["outcome", "group"]), sub["outcome"], entity_id="g0"
        )
    for key in ["auc", "accuracy", "f1_score", "conformal_coverage", "mean_set_size"]:
        assert key in metrics
    assert 0 <= metrics["auc"] <= 1
    assert 1 <= metrics["mean_set_size"] <= 2


def test_predict_before_fit_raises():
    pipe = Pipeline()
    with pytest.raises(RuntimeError):
        pipe.predict(pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]}))


# ============================================================================
# Regression + count tasks (domain-agnosticism)
# ============================================================================


def test_regression_end_to_end():
    pipe, target = _fit_pipeline("regression", _regression_frame)
    X = target[target.group == "g1"].drop(columns=["outcome", "group"]).head(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = pipe.predict(X, entity_id="g1")
    assert list(preds.columns) == CONTRACT_COLUMNS
    # Regression conformal regions are intervals
    assert preds["conformal_set"].str.startswith("[").all()
    assert preds["credible_lower"].le(preds["credible_upper"]).all()


def test_count_end_to_end():
    pipe, target = _fit_pipeline("count", _count_frame)
    X = target[target.group == "g2"].drop(columns=["outcome", "group"]).head(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = pipe.predict(X, entity_id="g2")
    assert list(preds.columns) == CONTRACT_COLUMNS
    # Expected counts are non-negative
    assert (preds["point_prediction"] >= 0).all()


# ============================================================================
# Conformal regression: CQR (default) + split-conformal fallback
# ============================================================================


def _interval_widths(conformal_set_series):
    return conformal_set_series.apply(
        lambda s: float(s[1:-1].split(",")[1]) - float(s[1:-1].split(",")[0])
    )


def test_regression_uses_cqr_with_enough_data():
    # 6 entities x 35 obs -> ~157 pooled training rows >= cqr_min_train (150).
    pipe, target = _fit_pipeline("regression", _regression_frame)
    assert pipe.conformal_method_ == "cqr"
    assert pipe.conformal_regressor is not None
    X = target[target.group == "g0"].drop(columns=["outcome", "group"]).head(12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = pipe.predict(X, entity_id="g0")
    # CQR is locally adaptive: interval widths vary across observations.
    assert _interval_widths(preds["conformal_set"]).nunique() > 1


def test_count_uses_cqr_with_enough_data():
    pipe, _ = _fit_pipeline("count", _count_frame)
    assert pipe.conformal_method_ == "cqr"


def test_regression_falls_back_to_split_when_small():
    # 5 entities x 22 obs -> ~82 pooled training rows < cqr_min_train (150).
    frames = []
    for k in range(5):
        df = _regression_frame(22, shift=0.2 * k, seed=300 + k)
        df["group"] = f"g{k}"
        frames.append(df)
    target = pd.concat(frames, ignore_index=True)
    reference = _regression_frame(800, seed=7)
    pipe = Pipeline(task="regression", quick_mode=True, cqr_min_train=150)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(
            target,
            target_col="outcome",
            entity_col="group",
            reference_data=reference,
            validate_convergence=False,
        )
    assert pipe.conformal_method_ == "split_normalized"
    assert pipe.conformal_regressor is None
    # Fallback is normalized split conformal (locally adaptive via posterior std).
    assert pipe.conformal_predictor.normalized_ is True
    X = target[target.group == "g0"].drop(columns=["outcome", "group"]).head(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = pipe.predict(X, entity_id="g0")
    assert _interval_widths(preds["conformal_set"]).nunique() > 1


def test_output_contract_identical_across_tasks(binary_pipeline):
    """The five-field contract must be the same regardless of task/domain."""
    bpipe, btarget = binary_pipeline
    rpipe, rtarget = _fit_pipeline("regression", _regression_frame)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bp = bpipe.predict(
            btarget[btarget.group == "g0"].drop(columns=["outcome", "group"]).head(3),
            entity_id="g0",
        )
        rp = rpipe.predict(
            rtarget[rtarget.group == "g0"].drop(columns=["outcome", "group"]).head(3),
            entity_id="g0",
        )
    assert list(bp.columns) == list(rp.columns) == CONTRACT_COLUMNS


# ============================================================================
# Minimum-data enforcement
# ============================================================================


def test_too_few_entities_raises():
    target = _multi_entity(_binary_frame, n_entities=3)
    pipe = Pipeline(task="binary", quick_mode=True)
    with pytest.raises(ValueError, match="at least"):
        pipe.fit(target, target_col="outcome", entity_col="group")


def test_allow_single_entity_bypasses_floor():
    target = _binary_frame(60, seed=5)  # one entity, 60 obs
    pipe = Pipeline(task="binary", quick_mode=True, allow_single_entity=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(target, target_col="outcome", validate_convergence=False)
    assert pipe.single_entity_mode is True


def test_too_few_obs_per_entity_raises():
    # 6 entities but one only has 10 observations
    frames = []
    for k in range(6):
        n = 10 if k == 0 else 35
        df = _binary_frame(n, seed=200 + k)
        df["group"] = f"g{k}"
        frames.append(df)
    target = pd.concat(frames, ignore_index=True)
    pipe = Pipeline(task="binary", quick_mode=True)
    with pytest.raises(ValueError, match="observations"):
        pipe.fit(target, target_col="outcome", entity_col="group")


def test_reference_too_small_raises():
    target = _multi_entity(_binary_frame)  # 210 rows
    small_ref = _binary_frame(100, seed=1)  # < 5x
    pipe = Pipeline(task="binary", quick_mode=True)
    with pytest.raises(ValueError, match="Reference dataset too small"):
        pipe.fit(
            target,
            target_col="outcome",
            entity_col="group",
            reference_data=small_ref,
        )


def test_missing_target_column_raises():
    target = _multi_entity(_binary_frame).rename(columns={"outcome": "y"})
    pipe = Pipeline(task="binary", quick_mode=True)
    with pytest.raises(ValueError):
        pipe.fit(target, target_col="outcome", entity_col="group")


# ============================================================================
# Feature alignment
# ============================================================================


def test_feature_alignment_intersection_warns():
    target = _multi_entity(_binary_frame)
    target["only_in_target"] = RNG.randn(len(target))
    reference = _binary_frame(1500, seed=7)
    reference["only_in_reference"] = RNG.randn(len(reference))

    pipe = Pipeline(task="binary", quick_mode=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe.fit(
            target,
            target_col="outcome",
            entity_col="group",
            reference_data=reference,
            validate_convergence=False,
        )
    messages = " ".join(str(x.message) for x in w)
    assert "intersection" in messages.lower()
    # Aligned feature set excludes the non-shared raw columns
    assert "only_in_target" not in pipe.feature_input_cols_
    assert "only_in_reference" not in pipe.feature_input_cols_


# ============================================================================
# Layer 1 PriorExtractor
# ============================================================================


def test_prior_extractor_shapes():
    ref = _binary_frame(800, seed=3)
    pe = PriorExtractor(task="binary").fit(ref, "outcome", verbose=False)
    priors = pe.get_priors()
    p = len([c for c in ref.columns if c != "outcome"])
    assert priors["beta_0"].shape == (p,)
    assert priors["Sigma_0"].shape == (p, p)
    assert priors["feature_names"] == ["a", "b", "c"]


def test_prior_extractor_rejects_tiny_reference():
    ref = _binary_frame(20, seed=4)
    with pytest.raises(ValueError, match="too small"):
        PriorExtractor(task="binary", min_reference_size=100).fit(
            ref, "outcome", verbose=False
        )


# ============================================================================
# Diagnostics + persistence
# ============================================================================


def test_convergence_diagnostics(binary_pipeline):
    pipe, _ = binary_pipeline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = pipe.get_convergence_diagnostics()
    assert isinstance(diag, pd.DataFrame)
    assert {"parameter", "r_hat", "ess_bulk", "ess_tail"} <= set(diag.columns)


def test_save_and_load(binary_pipeline):
    pipe, target = binary_pipeline
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = tmp.name
    try:
        pipe.save(path)
        assert Path(path).exists()
        loaded = Pipeline.load(path)
        assert loaded.target_col == pipe.target_col
        X = target[target.group == "g0"].drop(columns=["outcome", "group"]).head(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = loaded.predict(X, entity_id="g0")
        assert list(preds.columns) == CONTRACT_COLUMNS
        assert len(preds) == 4
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        Pipeline.load("/nonexistent/path/pipeline.pkl")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
