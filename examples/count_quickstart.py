"""
SmallML Quickstart — Count data (non-negative integers)
=======================================================

Same interface as the other quickstarts; ``task='count'`` uses a Poisson
likelihood. The point prediction is an expected count and the conformal region
is an integer prediction interval. The output contract is identical.
"""

import numpy as np
import pandas as pd

from smallml import Pipeline


def make_frame(n, shift=0.0, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n) + shift
    b = rng.randn(n)
    outcome = rng.poisson(np.exp(0.5 * a - 0.3 * b + 0.2))
    return pd.DataFrame({"feature_a": a, "feature_b": b, "outcome": outcome})


if __name__ == "__main__":
    reference = make_frame(3000, seed=999)

    frames = []
    for k in range(6):
        df = make_frame(40, shift=0.2 * k, seed=k)
        df["entity"] = f"group_{k}"
        frames.append(df)
    target = pd.concat(frames, ignore_index=True)

    pipe = Pipeline(task="count", confidence_level=0.90)
    pipe.fit(
        target, target_col="outcome", entity_col="entity", reference_data=reference
    )

    new_rows = (
        target[target.entity == "group_0"].drop(columns=["outcome", "entity"]).head(5)
    )
    preds = pipe.predict(new_rows, entity_id="group_0")
    print("\nCount predictions (integer interval = conformal_set):")
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

    sub = target[target.entity == "group_0"]
    metrics = pipe.evaluate(
        sub.drop(columns=["outcome", "entity"]),
        sub["outcome"],
        entity_id="group_0",
    )
    print(
        "\nMetrics:",
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()},
    )
