"""
SmallML Quickstart — Domain-Agnostic Binary Classification
==========================================================

SmallML is a domain-agnostic Bayesian transfer-learning framework for small
data. You bring:

  1. a large *reference* dataset (your own — SmallML ships no domain data),
  2. a small *target* dataset with labeled entities,
  3. a task type ('binary', 'regression', or 'count').

You always get back the same five-field output contract:

    point_prediction | posterior_distribution | credible_lower/upper
    conformal_set    | confidence_flag

This example uses synthetic data; replace it with your own CSVs in production.
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
    # 1) Reference dataset (large) — used by Layer 1 to extract priors.
    reference = make_frame(3000, seed=999)

    # 2) Target dataset (small): one row per observation, grouped by entity.
    frames = []
    for k in range(6):
        df = make_frame(40, shift=0.25 * k, seed=k)
        df["entity"] = f"group_{k}"
        frames.append(df)
    target = pd.concat(frames, ignore_index=True)

    # 3) Fit: the domain comes entirely from the data + config, not the code.
    pipe = Pipeline(task="binary", confidence_level=0.90)
    pipe.fit(
        target,
        target_col="outcome",
        entity_col="entity",
        reference_data=reference,
    )

    # 4) Predict with the standard contract.
    new_rows = (
        target[target.entity == "group_0"].drop(columns=["outcome", "entity"]).head(5)
    )
    preds = pipe.predict(new_rows, entity_id="group_0")
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

    # 5) Evaluate.
    sub = target[target.entity == "group_0"]
    metrics = pipe.evaluate(
        sub.drop(columns=["outcome", "entity"]),
        sub["outcome"],
        entity_id="group_0",
    )
    print("\nMetrics:", {k: round(v, 3) for k, v in metrics.items()})
