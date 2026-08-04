"""Apply trust scoring and abstention to calibrated predictions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

from qoe_twin.artifact_lineage import (
    load_resolved_configuration,
    load_selected_final_calibrator,
    require_feature_columns,
    validate_calibrator_lineage,
    validate_model_lineage,
    write_prediction_lineage,
)
from qoe_twin.features import (
    get_cross_layer_feature_names,
)
from qoe_twin.trust import (
    calculate_data_quality,
    calculate_prediction_stability,
    calculate_trust,
)

DATA_PATH = Path(
    "data/processed/qoe_features_final.parquet"
)

MODEL_PATH = Path(
    "models/uncalibrated/"
    "cross_layer_random_forest_final.joblib"
)

CALIBRATED_DIRECTORY = Path("models/calibrated")

CONFIGURATION_PATH = Path(
    "results/metrics/"
    "selected_calibration_configuration_final.json"
)

OUTPUT_PATH = Path(
    "results/predictions/"
    "final_trusted_predictions.parquet"
)

SUMMARY_PATH = Path(
    "results/tables/"
    "final_trust_summary.csv"
)
CONFIG_DIRECTORY = Path("configs")

TARGET_COLUMN = "future_poor_qoe"


def final_random_forest_feature_lists() -> tuple[list[str], list[str]]:
    """Return the exact ordered feature contract used by the final RF."""
    categorical_features = ["resolution"]
    numeric_features = [
        feature
        for feature in get_cross_layer_feature_names()
        if feature not in categorical_features
    ]
    return numeric_features, categorical_features


def main() -> None:
    """Generate calibrated predictions, trust scores and abstentions."""
    configuration_lineage = load_resolved_configuration(
        CONFIG_DIRECTORY
    )
    (
        calibration_method,
        calibrator_path,
        selected_configuration,
    ) = load_selected_final_calibrator(
        CONFIGURATION_PATH,
        CALIBRATED_DIRECTORY,
    )
    decision_threshold = float(
        selected_configuration["decision_threshold"]
    )
    validation_f1 = float(
        selected_configuration["f1"]
    )
    validation_ece = float(
        selected_configuration[
            "expected_calibration_error"
        ]
    )
    (
        numeric_features,
        categorical_features,
    ) = final_random_forest_feature_lists()
    feature_names = [
        *numeric_features,
        *categorical_features,
    ]
    model_lineage = validate_model_lineage(
        MODEL_PATH,
        configuration=configuration_lineage,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    calibrator_lineage = validate_calibrator_lineage(
        calibrator_path,
        configuration=configuration_lineage,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        expected_parent_model_sha256=(
            model_lineage.artifact_sha256
        ),
    )

    frame = pd.read_parquet(DATA_PATH)

    test = frame[
        frame["split"].eq("test")
        & frame[TARGET_COLUMN].notna()
    ].copy()

    test[TARGET_COLUMN] = test[
        TARGET_COLUMN
    ].astype(int)

    require_feature_columns(
        test.columns,
        feature_names,
        context="final trusted prediction inference",
    )

    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(calibrator_path)

    test = test.sort_values(
        ["session_id", "timestamp"]
    ).reset_index(drop=True)

    raw_probability = model.predict_proba(
        test[feature_names]
    )[:, 1]

    calibrated_probability = calibrator.predict(
        raw_probability
    )

    previous_probability_by_session: dict[
        str,
        list[float],
    ] = defaultdict(list)

    trust_records: list[dict] = []

    for index, row in test.iterrows():
        session_id = str(
            row["session_id"]
        )

        probability = float(
            calibrated_probability[index]
        )

        previous_probabilities = (
            previous_probability_by_session[
                session_id
            ]
        )

        stability = (
            calculate_prediction_stability(
                probability,
                previous_probabilities,
                window=5,
            )
        )

        data_quality = calculate_data_quality(
            row,
            feature_names,
        )

        trust = calculate_trust(
            probability=probability,
            validation_f1=validation_f1,
            validation_ece=validation_ece,
            data_quality=data_quality,
            prediction_stability=stability,
            abstention_threshold=0.55,
        )

        prediction = int(
            probability >= decision_threshold
        )

        record = {
            "session_id": session_id,
            "user_id": int(row["user_id"]),
            "timestamp": row["timestamp"],
            "future_timestamp": row[
                "future_timestamp"
            ],
            "actual_future_poor_qoe": int(
                row[TARGET_COLUMN]
            ),
            "raw_probability": float(
                raw_probability[index]
            ),
            "calibrated_probability": probability,
            "decision_threshold": (
                decision_threshold
            ),
            "predicted_poor_qoe": prediction,
            **trust.to_dict(),
        }

        trust_records.append(record)

        previous_probabilities.append(
            probability
        )

    trusted = pd.DataFrame(
        trust_records
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trusted.to_parquet(
        OUTPUT_PATH,
        index=False,
    )
    write_prediction_lineage(
        OUTPUT_PATH,
        configuration=configuration_lineage,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        parent_model_sha256=model_lineage.artifact_sha256,
        parent_calibrator_sha256=(
            calibrator_lineage.artifact_sha256
        ),
    )

    summary = (
        trusted.groupby(
            ["trust_level", "abstain"],
            dropna=False,
        )
        .agg(
            predictions=(
                "predicted_poor_qoe",
                "size",
            ),
            mean_trust_score=(
                "trust_score",
                "mean",
            ),
            mean_probability=(
                "calibrated_probability",
                "mean",
            ),
            actual_poor_qoe_rate=(
                "actual_future_poor_qoe",
                "mean",
            ),
            prediction_accuracy=(
                "predicted_poor_qoe",
                lambda prediction: (
                    prediction.to_numpy()
                    == trusted.loc[
                        prediction.index,
                        "actual_future_poor_qoe",
                    ].to_numpy()
                ).mean(),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    abstention_rate = float(
        trusted["abstain"].mean()
    )

    accepted = trusted[
        ~trusted["abstain"]
    ]

    accepted_accuracy = float(
        (
            accepted["predicted_poor_qoe"]
            == accepted[
                "actual_future_poor_qoe"
            ]
        ).mean()
    )

    overall_accuracy = float(
        (
            trusted["predicted_poor_qoe"]
            == trusted[
                "actual_future_poor_qoe"
            ]
        ).mean()
    )

    print("TRUST AND ABSTENTION SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print(
        f"\nTotal predictions: "
        f"{len(trusted):,}"
    )
    print(
        f"Abstention rate: "
        f"{abstention_rate:.2%}"
    )
    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.4f}"
    )
    print(
        f"Accepted-decision accuracy: "
        f"{accepted_accuracy:.4f}"
    )

    print("\nSaved:")
    print(f"- {OUTPUT_PATH}")
    print(f"- {SUMMARY_PATH}")
    print(f"- calibration method: {calibration_method}")


if __name__ == "__main__":
    main()
