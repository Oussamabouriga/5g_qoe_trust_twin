"""Apply trust scoring and abstention to calibrated predictions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

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

CALIBRATOR_PATH = Path(
    "models/calibrated/"
    "cross_layer_random_forest_isotonic_final.joblib"
)

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

TARGET_COLUMN = "future_poor_qoe"


def main() -> None:
    """Generate calibrated predictions, trust scores and abstentions."""
    frame = pd.read_parquet(DATA_PATH)

    test = frame[
        frame["split"].eq("test")
        & frame[TARGET_COLUMN].notna()
    ].copy()

    test[TARGET_COLUMN] = test[
        TARGET_COLUMN
    ].astype(int)

    feature_names = [
        feature
        for feature in get_cross_layer_feature_names()
        if feature in test.columns
    ]

    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)

    configuration = json.loads(
        CONFIGURATION_PATH.read_text(
            encoding="utf-8"
        )
    )["cross_layer_random_forest"]

    decision_threshold = float(
        configuration["decision_threshold"]
    )

    validation_f1 = float(
        configuration["f1"]
    )

    validation_ece = float(
        configuration[
            "expected_calibration_error"
        ]
    )

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

    print(f"\nSaved:")
    print(f"- {OUTPUT_PATH}")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
