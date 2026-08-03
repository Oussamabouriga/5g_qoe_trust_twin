"""Calibrate final model probabilities using chronological data."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from qoe_twin.calibration import (
    ProbabilityCalibrator,
    calibration_table,
)
from qoe_twin.features import (
    get_cross_layer_feature_names,
    get_network_feature_names,
)
from qoe_twin.metrics import (
    classification_metrics,
    find_best_f1_threshold,
)


DATA_PATH = Path(
    "data/processed/qoe_features_final.parquet"
)

MODEL_DIRECTORY = Path("models/uncalibrated")
CALIBRATED_DIRECTORY = Path("models/calibrated")

TABLE_DIRECTORY = Path("results/tables")
METRIC_DIRECTORY = Path("results/metrics")
PREDICTION_DIRECTORY = Path("results/predictions")

TARGET_COLUMN = "future_poor_qoe"


def required_final_random_forest_artifact_path(
    model_directory: Path = MODEL_DIRECTORY,
) -> Path:
    """Return the required final RF path or fail closed."""
    artifact_path = (
        model_directory
        / "cross_layer_random_forest_final.joblib"
    )

    if not artifact_path.is_file():
        raise FileNotFoundError(
            "Required final Random Forest artifact is missing: "
            f"{artifact_path}"
        )

    return artifact_path


def split_validation_chronologically(
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split validation into calibration and threshold-selection periods.
    """
    timestamps = (
        validation["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    midpoint = timestamps.iloc[
        len(timestamps) // 2
    ]

    calibration = validation[
        validation["timestamp"] < midpoint
    ].copy()

    selection = validation[
        validation["timestamp"] >= midpoint
    ].copy()

    if calibration.empty or selection.empty:
        raise ValueError(
            "Calibration and selection periods "
            "must both contain observations."
        )

    if (
        calibration["timestamp"].max()
        >= selection["timestamp"].min()
    ):
        raise ValueError(
            "Calibration and threshold-selection "
            "periods overlap."
        )

    return calibration, selection


def get_feature_lists(
    frame: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
]:
    """Return network and cross-layer model feature lists."""
    network_features = [
        feature
        for feature in get_network_feature_names()
        if feature in frame.columns
    ]

    cross_features = [
        feature
        for feature in get_cross_layer_feature_names()
        if feature in frame.columns
    ]

    return network_features, cross_features


def evaluate_probability_variant(
    model_name: str,
    calibration_method: str,
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[dict, float, pd.DataFrame]:
    """Select a threshold and evaluate one probability variant."""
    threshold, threshold_curve = (
        find_best_f1_threshold(
            target,
            probability,
        )
    )

    metrics = classification_metrics(
        target,
        probability,
        threshold=threshold,
    )

    metrics["model"] = model_name
    metrics["calibration_method"] = (
        calibration_method
    )

    curve = pd.DataFrame(
        threshold_curve
    )

    curve["model"] = model_name
    curve["calibration_method"] = (
        calibration_method
    )

    return metrics, threshold, curve


def save_selection_predictions(
    selection: pd.DataFrame,
    model_name: str,
    calibration_method: str,
    probability: np.ndarray,
    threshold: float,
) -> None:
    """Save probabilities and decisions."""
    output = selection[
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
    output["calibration_method"] = (
        calibration_method
    )
    output["probability_poor_qoe"] = (
        probability
    )
    output["decision_threshold"] = threshold
    output["predicted_poor_qoe"] = (
        probability >= threshold
    ).astype(int)

    path = (
        PREDICTION_DIRECTORY
        / (
            f"{model_name}_{calibration_method}"
            "_final_selection_predictions.parquet"
        )
    )

    output.to_parquet(
        path,
        index=False,
    )


def main() -> None:
    """Calibrate and compare model probabilities."""
    final_random_forest_path = (
        required_final_random_forest_artifact_path(
            MODEL_DIRECTORY
        )
    )

    frame = pd.read_parquet(DATA_PATH)

    validation = frame[
        frame["split"].eq("validation")
        & frame[TARGET_COLUMN].notna()
    ].copy()

    validation[TARGET_COLUMN] = (
        validation[TARGET_COLUMN].astype(int)
    )

    calibration_frame, selection_frame = (
        split_validation_chronologically(
            validation
        )
    )

    (
        network_features,
        cross_features,
    ) = get_feature_lists(frame)

    models = {
        "network_logistic": {
            "path": (
                MODEL_DIRECTORY
                / "network_logistic_final.joblib"
            ),
            "features": network_features,
        },
        "cross_layer_random_forest": {
            "path": final_random_forest_path,
            "features": cross_features,
        },
    }

    CALIBRATED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    TABLE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    METRIC_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    PREDICTION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_calibration = calibration_frame[
        TARGET_COLUMN
    ].to_numpy()

    y_selection = selection_frame[
        TARGET_COLUMN
    ].to_numpy()

    result_rows: list[dict] = []
    threshold_rows: list[pd.DataFrame] = []
    reliability_rows: list[pd.DataFrame] = []

    selected_configuration: dict[
        str,
        dict[str, float | str],
    ] = {}

    print("CALIBRATION DATA")
    print("=" * 70)
    print(
        f"Calibration rows: "
        f"{len(calibration_frame):,}"
    )
    print(
        f"Selection rows: "
        f"{len(selection_frame):,}"
    )
    print(
        "Calibration period:",
        calibration_frame["timestamp"].min(),
        "to",
        calibration_frame["timestamp"].max(),
    )
    print(
        "Selection period:",
        selection_frame["timestamp"].min(),
        "to",
        selection_frame["timestamp"].max(),
    )

    for model_name, specification in models.items():
        print(f"\nProcessing {model_name}...")

        model = joblib.load(
            specification["path"]
        )

        features = specification["features"]

        raw_calibration_probability = (
            model.predict_proba(
                calibration_frame[features]
            )[:, 1]
        )

        raw_selection_probability = (
            model.predict_proba(
                selection_frame[features]
            )[:, 1]
        )

        probability_variants: dict[
            str,
            np.ndarray,
        ] = {
            "uncalibrated": (
                raw_selection_probability
            )
        }

        for method in [
            "sigmoid",
            "isotonic",
        ]:
            calibrator = ProbabilityCalibrator(
                method=method
            )

            calibrator.fit(
                raw_calibration_probability,
                y_calibration,
            )

            calibrated_probability = (
                calibrator.predict(
                    raw_selection_probability
                )
            )

            probability_variants[
                method
            ] = calibrated_probability

            calibrator_path = (
                CALIBRATED_DIRECTORY
                / f"{model_name}_{method}_final.joblib"
            )

            joblib.dump(
                calibrator,
                calibrator_path,
            )

        model_results = []

        for method, probability in (
            probability_variants.items()
        ):
            (
                metrics,
                threshold,
                threshold_curve,
            ) = evaluate_probability_variant(
                model_name=model_name,
                calibration_method=method,
                target=y_selection,
                probability=probability,
            )

            result_rows.append(metrics)
            model_results.append(metrics)
            threshold_rows.append(
                threshold_curve
            )

            reliability = calibration_table(
                y_selection,
                probability,
                number_of_bins=10,
            )

            reliability["model"] = model_name
            reliability[
                "calibration_method"
            ] = method

            reliability_rows.append(
                reliability
            )

            save_selection_predictions(
                selection=selection_frame,
                model_name=model_name,
                calibration_method=method,
                probability=probability,
                threshold=threshold,
            )

        best = min(
            model_results,
            key=lambda row: (
                row[
                    "expected_calibration_error"
                ],
                row["brier_score"],
                -row["f1"],
            ),
        )

        selected_configuration[
            model_name
        ] = {
            "calibration_method": (
                best["calibration_method"]
            ),
            "decision_threshold": (
                best["threshold"]
            ),
            "f1": best["f1"],
            "pr_auc": best["pr_auc"],
            "brier_score": (
                best["brier_score"]
            ),
            "expected_calibration_error": (
                best[
                    "expected_calibration_error"
                ]
            ),
        }

    comparison = pd.DataFrame(
        result_rows
    )

    comparison = comparison[
        [
            "model",
            "calibration_method",
            "threshold",
            "f1",
            "precision",
            "recall",
            "pr_auc",
            "brier_score",
            "expected_calibration_error",
            "false_alarm_rate",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        ]
    ].sort_values(
        [
            "expected_calibration_error",
            "brier_score",
        ]
    )

    comparison_path = (
        TABLE_DIRECTORY
        / "final_calibration_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    pd.concat(
        threshold_rows,
        ignore_index=True,
    ).to_csv(
        TABLE_DIRECTORY
        / "final_calibrated_threshold_curves.csv",
        index=False,
    )

    pd.concat(
        reliability_rows,
        ignore_index=True,
    ).to_csv(
        TABLE_DIRECTORY
        / "final_reliability_data.csv",
        index=False,
    )

    selection_path = (
        METRIC_DIRECTORY
        / "selected_calibration_configuration_final.json"
    )

    selection_path.write_text(
        json.dumps(
            selected_configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nCALIBRATION COMPARISON — "
        "SECOND VALIDATION PERIOD"
    )
    print("=" * 150)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(
            comparison.to_string(
                index=False
            )
        )

    print("\nSelected configurations:")
    print(
        json.dumps(
            selected_configuration,
            indent=2,
        )
    )

    print("\nSaved:")
    print(f"- {comparison_path}")
    print(f"- {selection_path}")


if __name__ == "__main__":
    main()
