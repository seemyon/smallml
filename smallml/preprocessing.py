"""
Feature Preprocessing and Alignment

Domain-agnostic preprocessing for arbitrary tabular schemas:
- numeric scaling (standardization)
- categorical encoding (one-hot)
- missing-value imputation

Plus :func:`align_features`, which intersects the reference and target feature
spaces and reports any dropped columns (so transfer learning proceeds on the
shared feature set rather than failing on a mismatch).
"""

import warnings
from typing import List, Optional, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def align_features(
    reference_columns: Sequence[str],
    target_columns: Sequence[str],
    warn: bool = True,
) -> List[str]:
    """
    Intersect reference and target feature columns.

    Returns the shared columns (in reference order) and, when ``warn`` is True,
    issues a warning listing the columns dropped from each side.

    Raises
    ------
    ValueError
        If there are no shared feature columns.
    """
    ref = list(dict.fromkeys(reference_columns))
    tgt_set = set(target_columns)
    shared = [c for c in ref if c in tgt_set]

    if len(shared) == 0:
        raise ValueError(
            "Reference and target datasets share no feature columns. "
            "Transfer learning requires at least one common feature."
        )

    if warn:
        dropped_ref = [c for c in reference_columns if c not in shared]
        dropped_tgt = [c for c in target_columns if c not in shared]
        if dropped_ref or dropped_tgt:
            warnings.warn(
                "Feature spaces differ; proceeding on the intersection "
                f"({len(shared)} shared features). "
                f"Dropped from reference: {dropped_ref or 'none'}. "
                f"Dropped from target: {dropped_tgt or 'none'}."
            )

    return shared


class FeaturePreprocessor:
    """
    Fit-once / transform-many preprocessor for arbitrary tabular features.

    Numeric columns are imputed (mean) and standardized; categorical columns
    (object/category/bool dtypes, or any explicitly listed) are imputed
    (most-frequent) and one-hot encoded with ``handle_unknown='ignore'``.

    Parameters
    ----------
    scale : bool, default=True
        Standardize numeric features.
    encode_categoricals : bool, default=True
        One-hot encode categorical features (if False, non-numeric columns are
        dropped).
    categorical_cols : sequence of str, optional
        Force these columns to be treated as categorical.

    Attributes
    ----------
    input_columns_ : List[str]
        Raw feature columns seen at fit time.
    feature_names_out_ : List[str]
        Transformed (output) feature names.
    """

    def __init__(
        self,
        scale: bool = True,
        encode_categoricals: bool = True,
        categorical_cols: Optional[Sequence[str]] = None,
    ):
        self.scale = scale
        self.encode_categoricals = encode_categoricals
        self.categorical_cols = list(categorical_cols) if categorical_cols else None

        self.transformer_: Optional[ColumnTransformer] = None
        self.input_columns_: Optional[List[str]] = None
        self.numeric_cols_: Optional[List[str]] = None
        self.categorical_cols_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None

    def fit(
        self, X: pd.DataFrame, feature_names: Optional[Sequence[str]] = None
    ) -> "FeaturePreprocessor":
        """Learn imputation/scaling/encoding from ``X``."""
        X = self._select(X, feature_names)
        self.input_columns_ = list(X.columns)

        if self.categorical_cols is not None:
            cat = [c for c in self.categorical_cols if c in X.columns]
        else:
            cat = [
                c
                for c in X.columns
                if str(X[c].dtype) in ("object", "category", "bool")
            ]
        num = [c for c in X.columns if c not in cat]

        self.numeric_cols_ = num
        self.categorical_cols_ = cat if self.encode_categoricals else []

        transformers = []
        if num:
            steps = [("impute", SimpleImputer(strategy="mean"))]
            if self.scale:
                steps.append(("scale", StandardScaler()))
            transformers.append(("num", SkPipeline(steps), num))
        if self.categorical_cols_:
            transformers.append(
                (
                    "cat",
                    SkPipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    self.categorical_cols_,
                )
            )

        if not transformers:
            raise ValueError("No usable feature columns found for preprocessing.")

        self.transformer_ = ColumnTransformer(transformers, remainder="drop")
        self.transformer_.fit(X)
        self.feature_names_out_ = [
            self._clean_name(n) for n in self.transformer_.get_feature_names_out()
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted preprocessing, returning a numeric DataFrame."""
        if self.transformer_ is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        # Reindex to the columns seen at fit time (missing -> NaN -> imputed).
        X = X.reindex(columns=self.input_columns_)
        arr = self.transformer_.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_out_, index=X.index)

    def fit_transform(
        self, X: pd.DataFrame, feature_names: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        return self.fit(X, feature_names).transform(X)

    def get_feature_names_out(self) -> List[str]:
        if self.feature_names_out_ is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return list(self.feature_names_out_)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _select(
        X: pd.DataFrame, feature_names: Optional[Sequence[str]]
    ) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if feature_names is not None:
            return X[list(feature_names)].copy()
        return X.copy()

    @staticmethod
    def _clean_name(name: str) -> str:
        """Strip the ColumnTransformer ``num__``/``cat__`` prefixes."""
        for prefix in ("num__", "cat__"):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name
