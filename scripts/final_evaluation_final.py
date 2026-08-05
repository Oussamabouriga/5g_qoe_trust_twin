"""Complete scientific evaluation of the prototype digital twin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
)

from qoe_twin.artifact_lineage import (
    load_resolved_configuration,
    load_selected_final_calibrator,
    require_feature_columns,
    validate_prediction_chain,
)
from qoe_twin.evaluation import (
    build_reliability_table,
    evaluate_binary_predictions,
    evaluate_groups,
    evaluate_trust_levels,
    lead_time_statistics,
)
from qoe_twin.features import get_cross_layer_feature_names

FEATURE_PATH = Path(
    "data/processed/qoe_features_final.parquet"
)

TRUSTED_PREDICTION_PATH = Path(
    "results/predictions/"
    "final_trusted_predictions.parquet"
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
CONFIG_DIRECTORY = Path("configs")

TABLE_DIRECTORY = Path("results/tables")
FIGURE_DIRECTORY = Path("results/figures")
METRIC_DIRECTORY = Path("results/metrics")

TARGET_COLUMN = "actual_future_poor_qoe"
PREDICTION_COLUMN = "predicted_poor_qoe"
PROBABILITY_COLUMN = "calibrated_probability"


def final_random_forest_feature_lists() -> tuple[list[str], list[str]]:
    """Return the exact ordered feature contract used by the final RF."""
    categorical_features = ["resolution"]
    numeric_features = [
        feature
        for feature in get_cross_layer_feature_names()
        if feature not in categorical_features
    ]
    return numeric_features, categorical_features


def json_default(value: Any) -> Any:
    """Convert NumPy and pandas objects for JSON serialization."""
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    raise TypeError(
        f"Unsupported JSON object: {type(value)}"
    )


def load_evaluation_dataset() -> pd.DataFrame:
    """Merge trusted predictions with scenario and feature information."""
    print(
        f"Reading predictions: "
        f"{TRUSTED_PREDICTION_PATH}"
    )

    predictions = pd.read_parquet(
        TRUSTED_PREDICTION_PATH
    )
    require_feature_columns(
        predictions.columns,
        [
            "session_id",
            "user_id",
            "timestamp",
            TARGET_COLUMN,
            PREDICTION_COLUMN,
            PROBABILITY_COLUMN,
            "abstain",
            "trust_level",
        ],
        context="final evaluation predictions",
    )

    feature_columns = [
        "session_id",
        "user_id",
        "timestamp",
        "base_station",
        "prb",
        "bitrate_kbps",
        "resolution",
        "throughput_mbps",
        "current_mos",
        "plr_percent",
        "prediction_lead_seconds",
        "throughput_to_bitrate_ratio",
        "capacity_margin_mbps",
        "mos_change_1",
        "throughput_change_1",
    ]

    print(
        f"Reading features: "
        f"{FEATURE_PATH}"
    )

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=feature_columns,
    )
    require_feature_columns(
        features.columns,
        feature_columns,
        context="final evaluation features",
    )

    merged = predictions.merge(
        features,
        on=[
            "session_id",
            "user_id",
            "timestamp",
        ],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )

    if merged["base_station"].isna().any():
        missing = int(
            merged["base_station"].isna().sum()
        )

        raise ValueError(
            f"{missing} predictions could not be "
            "matched to feature rows."
        )

    if "prediction_lead_seconds_feature" in merged.columns:
        merged["prediction_lead_seconds"] = (
            merged[
                "prediction_lead_seconds_feature"
            ]
        )

        merged = merged.drop(
            columns=[
                "prediction_lead_seconds_feature"
            ]
        )

    return merged


def save_confusion_matrix(
    frame: pd.DataFrame,
) -> None:
    """Save confusion matrix as CSV and PNG."""
    matrix = confusion_matrix(
        frame[TARGET_COLUMN],
        frame[PREDICTION_COLUMN],
        labels=[0, 1],
    )

    matrix_frame = pd.DataFrame(
        matrix,
        index=[
            "actual_good_qoe",
            "actual_poor_qoe",
        ],
        columns=[
            "predicted_good_qoe",
            "predicted_poor_qoe",
        ],
    )

    matrix_frame.to_csv(
        TABLE_DIRECTORY
        / "final_confusion_matrix.csv"
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Good QoE",
            "Poor QoE",
        ],
    )

    display.plot(
        ax=axis,
        values_format=",d",
    )

    axis.set_title(
        "Cross-layer Random Forest confusion matrix"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_confusion_matrix.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def save_precision_recall_curve(
    frame: pd.DataFrame,
) -> None:
    """Save the precision-recall curve."""
    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    PrecisionRecallDisplay.from_predictions(
        y_true=frame[TARGET_COLUMN],
        y_pred=frame[PROBABILITY_COLUMN],
        name="Calibrated Random Forest",
        ax=axis,
    )

    axis.set_title(
        "Precision-recall curve"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_precision_recall_curve.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def save_reliability_diagram(
    reliability: pd.DataFrame,
) -> None:
    """Save the probability reliability diagram."""
    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.plot(
        reliability["mean_probability"],
        reliability["observed_frequency"],
        marker="o",
        label="Isotonic calibrated model",
    )

    axis.set_xlabel(
        "Mean predicted probability"
    )
    axis.set_ylabel(
        "Observed poor-QoE frequency"
    )
    axis.set_title(
        "Reliability diagram"
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_reliability_diagram.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_reliability_diagram.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def save_trust_distribution(
    trust_metrics: pd.DataFrame,
) -> None:
    """Save prediction counts by trust level."""
    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.bar(
        trust_metrics["trust_level"],
        trust_metrics["predictions"],
    )

    axis.set_xlabel("Trust level")
    axis.set_ylabel("Predictions")
    axis.set_title(
        "Prediction distribution by trust level"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_trust_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_trust_distribution.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def save_accuracy_by_trust(
    trust_metrics: pd.DataFrame,
) -> None:
    """Save decision accuracy by trust level."""
    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.bar(
        trust_metrics["trust_level"],
        trust_metrics["accuracy"],
    )

    axis.set_xlabel("Trust level")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.set_title(
        "Prediction accuracy by trust level"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_accuracy_by_trust.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_accuracy_by_trust.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def save_lead_time_histogram(
    frame: pd.DataFrame,
) -> None:
    """Save prediction lead-time distribution."""
    lead_time = pd.to_numeric(
        frame["prediction_lead_seconds"],
        errors="coerce",
    ).dropna()

    plot_limit = lead_time.quantile(0.99)
    visible = lead_time[
        lead_time <= plot_limit
    ]

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.hist(
        visible,
        bins=40,
    )

    axis.set_xlabel(
        "Prediction lead time (seconds)"
    )
    axis.set_ylabel("Observations")
    axis.set_title(
        "Prediction lead-time distribution"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_lead_time_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_lead_time_histogram.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def save_feature_importance() -> pd.DataFrame:
    """Extract and save Random Forest feature importance."""
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                classifier.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        TABLE_DIRECTORY
        / "final_feature_importance.csv",
        index=False,
    )

    top_features = importance.head(20).sort_values(
        "importance",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        top_features["feature"],
        top_features["importance"],
    )

    axis.set_xlabel("Feature importance")
    axis.set_ylabel("Feature")
    axis.set_title(
        "Top 20 Random Forest features"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / "final_feature_importance.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)

    return importance


def main() -> None:
    """Run and save the complete prototype evaluation."""
    configuration = load_resolved_configuration(
        CONFIG_DIRECTORY
    )
    (
        calibration_method,
        calibrator_path,
        _,
    ) = load_selected_final_calibrator(
        CONFIGURATION_PATH,
        CALIBRATED_DIRECTORY,
        configuration=configuration,
    )
    (
        numeric_features,
        categorical_features,
    ) = final_random_forest_feature_lists()
    validate_prediction_chain(
        model_path=MODEL_PATH,
        calibrator_path=calibrator_path,
        prediction_path=TRUSTED_PREDICTION_PATH,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    TABLE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = load_evaluation_dataset()

    print(
        f"Evaluation observations: "
        f"{len(frame):,}"
    )

    overall_metrics = evaluate_binary_predictions(
        target=frame[TARGET_COLUMN],
        prediction=frame[PREDICTION_COLUMN],
        probability=frame[
            PROBABILITY_COLUMN
        ],
    )

    accepted = frame[
        ~frame["abstain"]
    ].copy()

    accepted_metrics = (
        evaluate_binary_predictions(
            target=accepted[TARGET_COLUMN],
            prediction=accepted[
                PREDICTION_COLUMN
            ],
            probability=accepted[
                PROBABILITY_COLUMN
            ],
        )
    )

    trust_metrics = evaluate_trust_levels(
        frame=frame,
        target_column=TARGET_COLUMN,
        prediction_column=PREDICTION_COLUMN,
    )

    base_station_metrics = evaluate_groups(
        frame=frame,
        group_column="base_station",
        target_column=TARGET_COLUMN,
        prediction_column=PREDICTION_COLUMN,
        probability_column=PROBABILITY_COLUMN,
    )

    prb_metrics = evaluate_groups(
        frame=frame,
        group_column="prb",
        target_column=TARGET_COLUMN,
        prediction_column=PREDICTION_COLUMN,
        probability_column=PROBABILITY_COLUMN,
    )

    bitrate_metrics = evaluate_groups(
        frame=frame,
        group_column="bitrate_kbps",
        target_column=TARGET_COLUMN,
        prediction_column=PREDICTION_COLUMN,
        probability_column=PROBABILITY_COLUMN,
    )

    reliability = build_reliability_table(
        frame=frame,
        target_column=TARGET_COLUMN,
        probability_column=PROBABILITY_COLUMN,
    )

    lead_time = lead_time_statistics(
        frame["prediction_lead_seconds"]
    )

    trust_summary = {
        "total_predictions": int(len(frame)),
        "accepted_predictions": int(
            len(accepted)
        ),
        "abstained_predictions": int(
            frame["abstain"].sum()
        ),
        "abstention_rate": float(
            frame["abstain"].mean()
        ),
        "overall_accuracy": float(
            (
                frame[PREDICTION_COLUMN]
                == frame[TARGET_COLUMN]
            ).mean()
        ),
        "accepted_decision_accuracy": float(
            (
                accepted[PREDICTION_COLUMN]
                == accepted[TARGET_COLUMN]
            ).mean()
        ),
        "coverage": float(
            len(accepted) / len(frame)
        ),
    }

    overall_frame = pd.DataFrame(
        [
            {
                "evaluation_scope": "all_predictions",
                **overall_metrics,
            },
            {
                "evaluation_scope": "accepted_predictions",
                **accepted_metrics,
            },
        ]
    )

    overall_frame.to_csv(
        TABLE_DIRECTORY
        / "final_final_metrics.csv",
        index=False,
    )

    trust_metrics.to_csv(
        TABLE_DIRECTORY
        / "final_trust_metrics.csv",
        index=False,
    )

    base_station_metrics.to_csv(
        TABLE_DIRECTORY
        / "final_metrics_by_base_station.csv",
        index=False,
    )

    prb_metrics.to_csv(
        TABLE_DIRECTORY
        / "final_metrics_by_prb.csv",
        index=False,
    )

    bitrate_metrics.to_csv(
        TABLE_DIRECTORY
        / "final_metrics_by_bitrate.csv",
        index=False,
    )

    reliability.to_csv(
        TABLE_DIRECTORY
        / "final_final_reliability_table.csv",
        index=False,
    )

    pd.DataFrame(
        [lead_time]
    ).to_csv(
        TABLE_DIRECTORY
        / "final_lead_time_statistics.csv",
        index=False,
    )

    save_confusion_matrix(frame)
    save_precision_recall_curve(frame)
    save_reliability_diagram(
        reliability
    )
    save_trust_distribution(
        trust_metrics
    )
    save_accuracy_by_trust(
        trust_metrics
    )
    save_lead_time_histogram(frame)

    feature_importance = (
        save_feature_importance()
    )

    final_summary = {
        "dataset": "final_sample",
        "model": (
            "cross_layer_random_forest"
        ),
        "calibration": calibration_method,
        "overall_metrics": overall_metrics,
        "accepted_metrics": accepted_metrics,
        "trust_summary": trust_summary,
        "lead_time_statistics": lead_time,
        "top_20_features": (
            feature_importance.head(20)
            .to_dict(orient="records")
        ),
    }

    summary_path = (
        METRIC_DIRECTORY
        / "final_final_evaluation.json"
    )

    summary_path.write_text(
        json.dumps(
            final_summary,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print("\nFINAL SAMPLE EVALUATION")
    print("=" * 90)

    print("\nOverall metrics")
    print(
        overall_frame.to_string(
            index=False
        )
    )

    print("\nTrust metrics")
    print(
        trust_metrics.to_string(
            index=False
        )
    )

    print("\nTrust summary")
    for key, value in trust_summary.items():
        print(f"{key}: {value}")

    print("\nLead-time statistics")
    for key, value in lead_time.items():
        print(f"{key}: {value}")

    print("\nTop 10 features")
    print(
        feature_importance.head(10)
        .to_string(index=False)
    )

    print("\nSaved final evaluation:")
    print(f"- {summary_path}")
    print(f"- {TABLE_DIRECTORY}")
    print(f"- {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()
