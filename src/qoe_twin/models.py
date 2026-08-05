"""Predictive model construction for future poor-QoE forecasting."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_network_logistic_regression(
    numeric_features: list[str],
    random_seed: int = 42,
    max_iter: int = 2_000,
    class_weight: str = "balanced",
) -> Pipeline:
    """Create the required network-only Logistic Regression pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=random_seed,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def build_cross_layer_random_forest(
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int = 42,
    n_estimators: int = 250,
    max_depth: int | None = 18,
    min_samples_leaf: int = 5,
    n_jobs: int = 4,
    max_samples: float | None = None,
    class_weight: str = "balanced_subsample",
) -> Pipeline:
    """Create the required cross-layer Random Forest pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        max_samples=max_samples,
        random_state=random_seed,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
