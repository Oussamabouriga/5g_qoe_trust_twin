"""Train baseline, network-only and cross-layer QoE models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from qoe_twin.artifact_lineage import (
    load_resolved_configuration,
    require_feature_columns,
    write_model_lineage,
)
from qoe_twin.baselines import (
    CurrentQoEPersistenceBaseline,
)
from qoe_twin.features import (
    get_cross_layer_feature_names,
    get_network_feature_names,
)
from qoe_twin.metrics import (
    classification_metrics,
    find_best_f1_threshold,
)
from qoe_twin.models import (
    build_cross_layer_random_forest,
    build_network_logistic_regression,
)

DATA_PATH = Path(
    "data/processed/qoe_features_final.parquet"
)

MODEL_DIRECTORY = Path("models/uncalibrated")
RESULT_DIRECTORY = Path("results/metrics")
TABLE_DIRECTORY = Path("results/tables")
PREDICTION_DIRECTORY = Path("results/predictions")
CONFIG_DIRECTORY = Path("configs")

TARGET_COLUMN = "future_poor_qoe"
RANDOM_SEED = 42


def final_random_forest_artifact_path(
    model_directory: Path = MODEL_DIRECTORY,
) -> Path:
    """Return the artifact path produced by final RF training."""
    return (
        model_directory
        / "cross_layer_random_forest_final.joblib"
    )


def save_final_random_forest(
    model: object,
    model_directory: Path = MODEL_DIRECTORY,
) -> Path:
    """Persist the final RF to its contracted artifact path."""
    artifact_path = final_random_forest_artifact_path(
        model_directory
    )
    joblib.dump(model, artifact_path)
    return artifact_path


def clean_feature_lists(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Return the complete required feature lists or fail clearly."""
    network_features = list(get_network_feature_names())
    cross_layer_features = list(get_cross_layer_feature_names())

    categorical_features = [
        feature
        for feature in [
            "resolution",
        ]
        if feature in cross_layer_features
    ]

    cross_numeric_features = [
        feature
        for feature in cross_layer_features
        if feature not in categorical_features
    ]

    require_feature_columns(
        frame.columns,
        network_features,
        context="final network model",
    )
    require_feature_columns(
        frame.columns,
        [*cross_numeric_features, *categorical_features],
        context="final Random Forest",
    )

    return (
        network_features,
        cross_numeric_features,
        categorical_features,
    )


def prepare_split(
    frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Return one valid split."""
    split = frame[
        frame["split"] == split_name
    ].copy()

    split = split[
        split[TARGET_COLUMN].notna()
    ].copy()

    split[TARGET_COLUMN] = split[
        TARGET_COLUMN
    ].astype(int)

    return split


def save_predictions(
    frame: pd.DataFrame,
    model_name: str,
    probability: np.ndarray,
    threshold: float,
) -> None:
    """Save validation predictions for later analysis."""
    output = frame[
        [
            "session_id",
            "user_id",
            "timestamp",
            "future_timestamp",
            "prediction_lead_seconds",
            TARGET_COLUMN,
        ]
    ].copy()

    output["model"] = model_name
    output["probability_poor_qoe"] = probability
    output["decision_threshold"] = threshold
    output["predicted_poor_qoe"] = (
        probability >= threshold
    ).astype(int)

    path = (
        PREDICTION_DIRECTORY
        / f"{model_name}_validation_predictions.parquet"
    )

    output.to_parquet(
        path,
        index=False,
    )


def main() -> None:
    """Train and evaluate all required prototype models."""
    configuration = load_resolved_configuration(
        CONFIG_DIRECTORY
    )

    print(f"Reading {DATA_PATH}")
    frame = pd.read_parquet(DATA_PATH)

    train = prepare_split(frame, "train")
    validation = prepare_split(frame, "validation")

    (
        network_features,
        cross_numeric_features,
        categorical_features,
    ) = clean_feature_lists(frame)

    cross_features = [
        *cross_numeric_features,
        *categorical_features,
    ]

    print("\nDATA")
    print("=" * 70)
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(
        "Training poor-QoE rate:",
        f"{train[TARGET_COLUMN].mean():.2%}",
    )
    print(
        "Validation poor-QoE rate:",
        f"{validation[TARGET_COLUMN].mean():.2%}",
    )
    print(f"Network features: {len(network_features)}")
    print(f"Cross-layer features: {len(cross_features)}")

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    TABLE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    PREDICTION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_train = train[TARGET_COLUMN].to_numpy()
    y_validation = validation[TARGET_COLUMN].to_numpy()

    results: list[dict] = []
    thresholds: dict[str, float] = {}

    # --------------------------------------------------
    # Persistence baseline
    # --------------------------------------------------

    print("\nTraining/evaluating persistence baseline...")

    persistence = CurrentQoEPersistenceBaseline(
        poor_mos_threshold=3.0
    )

    persistence_probability = persistence.predict_proba(
        validation[["current_mos"]]
    )[:, 1]

    persistence_metrics = classification_metrics(
        y_validation,
        persistence_probability,
        threshold=0.5,
    )

    persistence_metrics["model"] = "persistence"
    results.append(persistence_metrics)
    thresholds["persistence"] = 0.5

    save_predictions(
        validation,
        "persistence",
        persistence_probability,
        0.5,
    )

    # --------------------------------------------------
    # Network Logistic Regression
    # --------------------------------------------------

    print("Training network-only Logistic Regression...")

    logistic = build_network_logistic_regression(
        numeric_features=network_features,
        random_seed=RANDOM_SEED,
    )

    logistic.fit(
        train[network_features],
        y_train,
    )

    logistic_probability = logistic.predict_proba(
        validation[network_features]
    )[:, 1]

    logistic_threshold, logistic_curve = (
        find_best_f1_threshold(
            y_validation,
            logistic_probability,
        )
    )

    logistic_metrics = classification_metrics(
        y_validation,
        logistic_probability,
        threshold=logistic_threshold,
    )

    logistic_metrics["model"] = "network_logistic"
    results.append(logistic_metrics)
    thresholds["network_logistic"] = logistic_threshold

    joblib.dump(
        logistic,
        MODEL_DIRECTORY
        / "network_logistic_final.joblib",
    )

    pd.DataFrame(logistic_curve).to_csv(
        TABLE_DIRECTORY
        / "final_network_logistic_threshold_curve.csv",
        index=False,
    )

    save_predictions(
        validation,
        "network_logistic",
        logistic_probability,
        logistic_threshold,
    )

    # --------------------------------------------------
    # Cross-layer Random Forest
    # --------------------------------------------------

    print("Training cross-layer Random Forest...")

    random_forest = build_cross_layer_random_forest(
        numeric_features=cross_numeric_features,
        categorical_features=categorical_features,
        random_seed=RANDOM_SEED,
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=20,
        n_jobs=4,
        max_samples=0.50,
    )

    random_forest.fit(
        train[cross_features],
        y_train,
    )

    forest_probability = random_forest.predict_proba(
        validation[cross_features]
    )[:, 1]

    forest_threshold, forest_curve = (
        find_best_f1_threshold(
            y_validation,
            forest_probability,
        )
    )

    forest_metrics = classification_metrics(
        y_validation,
        forest_probability,
        threshold=forest_threshold,
    )

    forest_metrics["model"] = "cross_layer_random_forest"
    results.append(forest_metrics)
    thresholds[
        "cross_layer_random_forest"
    ] = forest_threshold

    random_forest_path = save_final_random_forest(
        random_forest,
        MODEL_DIRECTORY,
    )
    write_model_lineage(
        random_forest_path,
        configuration=configuration,
        numeric_features=cross_numeric_features,
        categorical_features=categorical_features,
    )

    pd.DataFrame(forest_curve).to_csv(
        TABLE_DIRECTORY
        / "final_cross_layer_random_forest_threshold_curve.csv",
        index=False,
    )

    save_predictions(
        validation,
        "cross_layer_random_forest",
        forest_probability,
        forest_threshold,
    )

    # --------------------------------------------------
    # Save comparison
    # --------------------------------------------------

    comparison = pd.DataFrame(results)

    column_order = [
        "model",
        "threshold",
        "f1",
        "precision",
        "recall",
        "pr_auc",
        "brier_score",
        "false_alarm_rate",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    comparison = comparison[column_order].sort_values(
        ["f1", "pr_auc"],
        ascending=False,
    )

    comparison_path = (
        TABLE_DIRECTORY
        / "final_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    metadata = {
        "data_path": str(DATA_PATH),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "network_features": network_features,
        "cross_numeric_features": cross_numeric_features,
        "categorical_features": categorical_features,
        "selected_thresholds": thresholds,
    }

    metadata_path = (
        RESULT_DIRECTORY
        / "final_training_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nMODEL COMPARISON — VALIDATION SET")
    print("=" * 120)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
    ):
        print(
            comparison.to_string(
                index=False,
            )
        )

    print("\nSaved:")
    print(f"- {comparison_path}")
    print(f"- {metadata_path}")
    print(f"- {MODEL_DIRECTORY}")


if __name__ == "__main__":
    main()
