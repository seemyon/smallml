"""
SmallML Single-Entity Mode Example
==================================

For users with only ONE dataset (one group / unit). Single-entity mode bypasses
the J >= 5 minimum via ``allow_single_entity=True``. There is no cross-entity
pooling, so providing a reference dataset (for transfer-learning priors) is
strongly recommended to supply the "borrowed strength" that pooling normally
would.

The interface and the five-field output contract are identical to multi-entity
mode — only the data shape and the ``allow_single_entity`` flag change.
"""

import numpy as np
import pandas as pd

from smallml import Pipeline


def make_frame(n, shift=0.0, seed=0):
    """Synthetic binary data: 3 features, one binary outcome."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n) + shift
    b = rng.randn(n)
    c = rng.randn(n)
    logit = 1.4 * a - 0.7 * b + 0.3 * c - 0.2
    outcome = (1 / (1 + np.exp(-logit)) > rng.rand(n)).astype(int)
    return pd.DataFrame(
        {"feature_a": a, "feature_b": b, "feature_c": c, "outcome": outcome}
    )


if __name__ == "__main__":
    # A large reference dataset supplies the priors that pooling would otherwise.
    reference = make_frame(2000, seed=999)

    # A single target dataset (no entity column needed).
    my_data = make_frame(120, seed=1)
    print(f"Single dataset: {len(my_data)} observations")

    pipe = Pipeline(task="binary", allow_single_entity=True)
    pipe.fit(my_data, target_col="outcome", reference_data=reference)

    # No entity_id needed in single-entity mode.
    new_rows = make_frame(10, seed=2).drop(columns=["outcome"])
    preds = pipe.predict(new_rows)
    print("\nPredictions (standard output contract):")
    print(
        preds[
            [
                "point_prediction",
                "credible_lower",
                "credible_upper",
                "conformal_set",
                "confidence_flag",
            ]
        ]
        .round(3)
        .to_string()
    )

    test = make_frame(60, seed=3)
    metrics = pipe.evaluate(test.drop(columns=["outcome"]), test["outcome"])
    print("\nMetrics:", {k: round(v, 3) for k, v in metrics.items()})

    print("\nSave / reload:")
    print("  pipe.save('my_pipeline.pkl')")
    print("  pipe = Pipeline.load('my_pipeline.pkl')")
