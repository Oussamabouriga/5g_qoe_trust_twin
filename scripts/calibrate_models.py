"""Calibrate model probabilities using chronological validation data."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from qoe_twin.artifact_lineage import (
    load_resolved_configuration,
    require_feature_columns,
    validate_model_lineage,
    write_calibrator_lineage,
)
from qoe_twin.calibration import (
    ProbabilityCalibrator,
    calibration_table,
)
from qoe_twin.config import LoadedConfiguration
from qoe_twin.features import (
    get_cross_layer_feature_names,
    get_network_feature_names,
)
from qoe_twin.metrics import (
    classification_metrics,
    find_best_f1_threshold,
)
from qoe_twin.splitting import split_validation_chronologically
from qoe_twin.trust import (
    TrustPolicy,
    calculate_configured_reliability_score,
    calculate_data_quality,
    calculate_prediction_stability,
    select_validation_abstention_threshold,
)

DATA_PATH = Path(
    "data/processed/qoe_features_final.parquet"
)

MODEL_DIRECTORY = Path("models/uncalibrated")
CALIBRATED_DIRECTORY = Path("models/calibrated")

TABLE_DIRECTORY = Path("results/tables")
METRIC_DIRECTORY = Path("results/metrics")
PREDICTION_DIRECTORY = Path("results/predictions")
CONFIG_DIRECTORY = Path("configs")
RISK_COVERAGE_PATH = (
    TABLE_DIRECTORY / "final_rf_validation_risk_coverage.csv"
)

def save_final_random_forest_calibrator(
    calibrator: ProbabilityCalibrator,
    *,
    calibration_method: str,
    configuration: LoadedConfiguration,
    numeric_features: list[str],
    categorical_features: list[str],
    parent_model_sha256: str,
) -> Path:
    """Save one supported final RF calibrator and its lineage sidecar."""
    if calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError(
            "Final Random Forest calibrator method must be "
            "'sigmoid' or 'isotonic'."
        )
    calibrator_path = CALIBRATED_DIRECTORY / (
        "cross_layer_random_forest_"
        f"{calibration_method}_final.joblib"
    )
    joblib.dump(calibrator, calibrator_path)
    write_calibrator_lineage(
        calibrator_path,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        parent_model_sha256=parent_model_sha256,
    )
    return calibrator_path


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


def get_feature_lists(
    frame: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
]:
    """Return complete model feature lists or fail clearly."""
    network_features = list(get_network_feature_names())
    categorical_features = ["resolution"]
    cross_numeric_features = [
        feature
        for feature in get_cross_layer_feature_names()
        if feature not in categorical_features
    ]
    cross_features = [
        *cross_numeric_features,
        *categorical_features,
    ]

    require_feature_columns(
        frame.columns,
        network_features,
        context="final network calibration",
    )
    require_feature_columns(
        frame.columns,
        cross_features,
        context="final Random Forest calibration",
    )

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
    *,
    target_column: str,
) -> None:
    """Save probabilities and decisions."""
    output = selection[
        [
            "session_id",
            "user_id",
            "timestamp",
            "future_timestamp",
            "prediction_lead_seconds",
            target_column,
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


def build_validation_trust_scores(
    selection: pd.DataFrame,
    probability: np.ndarray,
    *,
    target_column: str,
    feature_names: list[str],
    decision_threshold: float,
    validation_f1: float,
    validation_ece: float,
    policy: TrustPolicy,
) -> pd.DataFrame:
    """Calculate threshold-independent trust scores on validation half two."""
    probabilities = np.asarray(probability, dtype=float).reshape(-1)
    if len(selection) != len(probabilities):
        raise ValueError(
            "Selection probabilities must align one-to-one with validation rows."
        )

    ordered = selection.copy()
    ordered["calibrated_probability"] = probabilities
    ordered = ordered.sort_values(
        ["timestamp", "session_id"],
        kind="stable",
    ).reset_index(drop=True)

    previous_probability_by_session: dict[str, list[float]] = defaultdict(list)
    records: list[dict[str, object]] = []

    for _, row in ordered.iterrows():
        session_id = str(row["session_id"])
        selected_probability = float(row["calibrated_probability"])
        previous_probabilities = previous_probability_by_session[session_id]
        stability = calculate_prediction_stability(
            selected_probability,
            previous_probabilities,
            window=5,
        )
        data_quality = calculate_data_quality(row, feature_names)
        trust_score = calculate_configured_reliability_score(
            probability=selected_probability,
            decision_threshold=decision_threshold,
            validation_f1=validation_f1,
            validation_ece=validation_ece,
            data_quality=data_quality,
            prediction_stability=stability,
            policy=policy,
        )
        records.append(
            {
                "session_id": session_id,
                "timestamp": row["timestamp"],
                "split": "validation",
                "target": int(row[target_column]),
                "prediction": int(
                    selected_probability >= decision_threshold
                ),
                "trust_score": trust_score,
            }
        )
        previous_probabilities.append(selected_probability)

    return pd.DataFrame(records)


def main() -> None:
    """Calibrate and compare model probabilities."""
    final_random_forest_path = (
        required_final_random_forest_artifact_path(
            MODEL_DIRECTORY
        )
    )

    configuration = load_resolved_configuration(
        CONFIG_DIRECTORY
    )
    categorical_features = ["resolution"]
    cross_numeric_features = [
        feature
        for feature in get_cross_layer_feature_names()
        if feature not in categorical_features
    ]
    model_lineage = validate_model_lineage(
        final_random_forest_path,
        configuration=configuration,
        numeric_features=cross_numeric_features,
        categorical_features=categorical_features,
    )
    model_configuration = configuration.values["model"]
    target_column = str(
        model_configuration["target"]["name"]
    )
    calibration_configuration = model_configuration[
        "calibration"
    ]
    calibration_methods = list(
        calibration_configuration["methods"]
    )
    calibration_fraction = float(
        calibration_configuration["fit_fraction"]
    )
    random_seed = int(
        model_configuration["experiment"]["random_seed"]
    )

    frame = pd.read_parquet(
        DATA_PATH,
        filters=[("split", "==", "validation")],
    )

    validation = frame[
        frame["split"].eq("validation")
        & frame[target_column].notna()
    ].copy()

    validation[target_column] = (
        validation[target_column].astype(int)
    )
    validation = validation.sort_values(
        ["timestamp", "session_id"],
        kind="stable",
    ).reset_index(drop=True)

    calibration_frame, selection_frame = (
        split_validation_chronologically(
            validation,
            calibration_fraction=calibration_fraction,
            target_column=target_column,
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
        target_column
    ].to_numpy()

    y_selection = selection_frame[
        target_column
    ].to_numpy()

    result_rows: list[dict] = []
    threshold_rows: list[pd.DataFrame] = []
    reliability_rows: list[pd.DataFrame] = []

    selected_configuration: dict[str, dict[str, object]] = {}
    selected_rf_probability: np.ndarray | None = None

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
        ] = {}

        for method in calibration_methods:
            calibrator = ProbabilityCalibrator(
                method=method,
                random_seed=random_seed,
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

            if model_name == "cross_layer_random_forest":
                save_final_random_forest_calibrator(
                    calibrator,
                    calibration_method=method,
                    configuration=configuration,
                    numeric_features=cross_numeric_features,
                    categorical_features=categorical_features,
                    parent_model_sha256=(
                        model_lineage.artifact_sha256
                    ),
                )
            else:
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
                target_column=target_column,
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

        if model_name == "cross_layer_random_forest":
            selected_rf_probability = probability_variants[
                str(best["calibration_method"])
            ]

    if selected_rf_probability is None:
        raise RuntimeError(
            "Final Random Forest validation probabilities were not produced."
        )

    selected_rf = selected_configuration["cross_layer_random_forest"]
    trust_policy = TrustPolicy.from_mapping(
        configuration.values["trust"]
    )
    validation_trust = build_validation_trust_scores(
        selection_frame,
        selected_rf_probability,
        target_column=target_column,
        feature_names=cross_features,
        decision_threshold=float(selected_rf["decision_threshold"]),
        validation_f1=float(selected_rf["f1"]),
        validation_ece=float(selected_rf["expected_calibration_error"]),
        policy=trust_policy,
    )
    abstention_threshold, risk_coverage = (
        select_validation_abstention_threshold(
            validation_trust,
            target_column="target",
            prediction_column="prediction",
            minimum_coverage=trust_policy.minimum_coverage,
        )
    )
    risk_coverage["selective_accuracy"] = (
        1.0 - risk_coverage["selective_risk"]
    )
    risk_coverage["minimum_coverage"] = trust_policy.minimum_coverage
    risk_coverage["selected"] = risk_coverage[
        "abstention_threshold"
    ].eq(abstention_threshold)
    selected_risk = risk_coverage.loc[
        risk_coverage["selected"]
    ].iloc[0]

    selected_rf.update(
        {
            "abstention_threshold": abstention_threshold,
            "abstention_minimum_coverage": trust_policy.minimum_coverage,
            "abstention_validation_coverage": float(
                selected_risk["coverage"]
            ),
            "abstention_validation_selective_accuracy": float(
                selected_risk["selective_accuracy"]
            ),
            "abstention_validation_selective_risk": float(
                selected_risk["selective_risk"]
            ),
        }
    )
    risk_coverage.to_csv(RISK_COVERAGE_PATH, index=False)

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

    selection_document = {
        "schema_version": 1,
        "configuration_sha256": configuration.sha256,
        "selection_policy": {
            "calibration_method": (
                "minimum expected_calibration_error, then minimum brier_score, "
                "then maximum f1"
            ),
            "decision_threshold": (
                "maximum validation-half-two F1, then precision, on the frozen "
                "0.05-to-0.95 grid with 0.01 step"
            ),
            "abstention_threshold": (
                "minimum selective risk subject to YAML minimum coverage; "
                "ties prefer higher coverage then higher threshold"
            ),
        },
        "validation_periods": {
            "calibrator_fit": {
                "rows": len(calibration_frame),
                "start_timestamp": calibration_frame["timestamp"].min().isoformat(),
                "end_timestamp": calibration_frame["timestamp"].max().isoformat(),
            },
            "selection": {
                "rows": len(selection_frame),
                "start_timestamp": selection_frame["timestamp"].min().isoformat(),
                "end_timestamp": selection_frame["timestamp"].max().isoformat(),
            },
        },
        **selected_configuration,
    }
    selection_path.write_text(
        json.dumps(
            selection_document,
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
            selection_document,
            indent=2,
        )
    )

    print("\nSaved:")
    print(f"- {comparison_path}")
    print(f"- {RISK_COVERAGE_PATH}")
    print(f"- {selection_path}")


if __name__ == "__main__":
    main()
