"""Evaluate the frozen trust-aware QoE forecasting experiment."""

from __future__ import annotations

import json
import math
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
from qoe_twin.features import (
    get_cross_layer_feature_names,
    get_network_feature_names,
)

plt.switch_backend("Agg")

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
PERSISTENCE_MODEL_PATH = Path(
    "models/uncalibrated/persistence_final.joblib"
)
NETWORK_LOGISTIC_MODEL_PATH = Path(
    "models/uncalibrated/network_logistic_final.joblib"
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
COMPARISON_PREDICTION_PATH = Path(
    "results/predictions/final_model_comparison_predictions.parquet"
)
COMPARISON_TABLE_PATH = Path(
    "results/tables/final_model_comparison.csv"
)
COMPARISON_METRIC_PATH = Path(
    "results/metrics/final_model_comparison.json"
)

TARGET_COLUMN = "actual_future_poor_qoe"
PREDICTION_COLUMN = "predicted_poor_qoe"
PROBABILITY_COLUMN = "calibrated_probability"
FEATURE_TARGET_COLUMN = "future_poor_qoe"
ROW_KEY_COLUMNS = ["session_id", "user_id", "timestamp"]


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


def load_network_logistic_selection(
    selection_path: Path,
    calibrated_directory: Path,
    *,
    expected_configuration_sha256: str,
) -> tuple[str, Path, float]:
    """Resolve the frozen network-logistic calibrator and threshold."""
    if not selection_path.is_file():
        raise FileNotFoundError(
            "Selected calibration configuration is missing: "
            f"{selection_path}"
        )
    try:
        document = json.loads(
            selection_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Cannot read the selected calibration configuration: "
            f"{selection_path}"
        ) from exc

    if not isinstance(document, dict):
        raise ValueError(
            "Selected calibration configuration must contain an object."
        )
    if document.get("configuration_sha256") != expected_configuration_sha256:
        raise ValueError(
            "Network-logistic selection configuration SHA-256 does not "
            "match the resolved configuration."
        )

    selection = document.get("network_logistic")
    if not isinstance(selection, dict):
        raise ValueError(
            "Selected calibration configuration must contain a "
            "network_logistic object."
        )
    method = selection.get("calibration_method")
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(
            "Network-logistic calibration_method must be 'sigmoid' or "
            f"'isotonic', not {method!r}."
        )

    threshold_value = selection.get("decision_threshold")
    if (
        type(threshold_value) not in {int, float}
        or not math.isfinite(float(threshold_value))
        or not 0.0 < float(threshold_value) < 1.0
    ):
        raise ValueError(
            "Network-logistic decision_threshold must be a finite number "
            "strictly between zero and one."
        )

    calibrator_path = calibrated_directory / (
        f"network_logistic_{method}_final.joblib"
    )
    if not calibrator_path.is_file():
        raise FileNotFoundError(
            "Selected network-logistic calibrator is missing: "
            f"{calibrator_path}"
        )
    return method, calibrator_path, float(threshold_value)


def _validated_row_keys(
    frame: pd.DataFrame,
    *,
    context: str,
) -> pd.MultiIndex:
    """Return unique non-null row identities for exact test-row matching."""
    require_feature_columns(
        frame.columns,
        ROW_KEY_COLUMNS,
        context=context,
    )
    if frame[ROW_KEY_COLUMNS].isna().any(axis=None):
        raise ValueError(f"{context} contains null row-key values.")
    if frame.duplicated(ROW_KEY_COLUMNS).any():
        raise ValueError(f"{context} contains duplicate row identities.")
    return pd.MultiIndex.from_frame(frame[ROW_KEY_COLUMNS])


def _positive_probability(
    values: Any,
    *,
    observations: int,
    context: str,
) -> np.ndarray:
    """Validate one model's positive-class probability vector."""
    probability = np.asarray(values, dtype=float).reshape(-1)
    if len(probability) != observations:
        raise ValueError(
            f"{context} returned {len(probability)} probabilities for "
            f"{observations} observations."
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"{context} returned non-finite probabilities.")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError(f"{context} returned probabilities outside [0, 1].")
    return probability


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
            *ROW_KEY_COLUMNS,
            TARGET_COLUMN,
            PREDICTION_COLUMN,
            PROBABILITY_COLUMN,
            "decision_threshold",
            "abstain",
            "trust_level",
        ],
        context="final evaluation predictions",
    )
    prediction_keys = _validated_row_keys(
        predictions,
        context="final evaluation predictions",
    )

    feature_columns = list(dict.fromkeys([
        *ROW_KEY_COLUMNS,
        "split",
        FEATURE_TARGET_COLUMN,
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
        *get_network_feature_names(),
    ]))

    print(
        f"Reading features: "
        f"{FEATURE_PATH}"
    )

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=feature_columns,
        filters=[("split", "==", "test")],
    )
    require_feature_columns(
        features.columns,
        feature_columns,
        context="final evaluation features",
    )
    if not features["split"].eq("test").all():
        raise ValueError(
            "Final evaluation feature input contains non-test rows."
        )
    if features[FEATURE_TARGET_COLUMN].isna().any():
        raise ValueError(
            "Final evaluation test rows contain missing future targets."
        )
    feature_keys = _validated_row_keys(
        features,
        context="final evaluation test features",
    )
    missing_predictions = feature_keys.difference(prediction_keys)
    unexpected_predictions = prediction_keys.difference(feature_keys)
    if len(missing_predictions) or len(unexpected_predictions):
        raise ValueError(
            "Trusted predictions and sealed test features do not contain "
            "the same row identities."
        )

    merged = features.merge(
        predictions,
        on=ROW_KEY_COLUMNS,
        how="inner",
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

    feature_target = merged[FEATURE_TARGET_COLUMN].astype(int)
    prediction_target = merged[TARGET_COLUMN].astype(int)
    if not feature_target.equals(prediction_target):
        raise ValueError(
            "Trusted prediction targets do not match the sealed test "
            "feature targets."
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

    return merged.sort_values(
        ["timestamp", "session_id", "user_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_three_model_test_predictions(
    frame: pd.DataFrame,
    *,
    persistence_model: object,
    network_logistic_model: object,
    network_logistic_calibrator: object,
    network_logistic_method: str,
    network_logistic_threshold: float,
    random_forest_method: str,
    random_forest_threshold: float,
) -> pd.DataFrame:
    """Predict three models on one exact sealed-test row set."""
    if frame.empty:
        raise ValueError("Three-model comparison requires sealed test rows.")
    _validated_row_keys(frame, context="three-model sealed test frame")
    network_features = list(get_network_feature_names())
    require_feature_columns(
        frame.columns,
        [
            *network_features,
            "current_mos",
            TARGET_COLUMN,
            PROBABILITY_COLUMN,
            PREDICTION_COLUMN,
            "decision_threshold",
        ],
        context="three-model sealed test frame",
    )

    for label, threshold in {
        "network-logistic": network_logistic_threshold,
        "Random Forest": random_forest_threshold,
    }.items():
        if not math.isfinite(float(threshold)) or not 0.0 < float(threshold) < 1.0:
            raise ValueError(
                f"{label} decision threshold must be finite and strictly "
                "between zero and one."
            )

    observations = len(frame)
    persistence_output = np.asarray(
        persistence_model.predict_proba(frame[["current_mos"]]),
        dtype=float,
    )
    logistic_output = np.asarray(
        network_logistic_model.predict_proba(frame[network_features]),
        dtype=float,
    )
    if persistence_output.shape != (observations, 2):
        raise ValueError(
            "Persistence baseline must return a two-column probability matrix."
        )
    if logistic_output.shape != (observations, 2):
        raise ValueError(
            "Network logistic model must return a two-column probability matrix."
        )

    persistence_probability = _positive_probability(
        persistence_output[:, 1],
        observations=observations,
        context="persistence baseline",
    )
    logistic_probability = _positive_probability(
        network_logistic_calibrator.predict(logistic_output[:, 1]),
        observations=observations,
        context="network-logistic calibrator",
    )
    random_forest_probability = _positive_probability(
        frame[PROBABILITY_COLUMN],
        observations=observations,
        context="trusted Random Forest predictions",
    )

    recorded_rf_threshold = pd.to_numeric(
        frame["decision_threshold"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(recorded_rf_threshold).all()
        or not np.allclose(
            recorded_rf_threshold,
            float(random_forest_threshold),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Trusted Random Forest predictions do not use the frozen "
            "decision threshold."
        )
    expected_rf_prediction = (
        random_forest_probability >= float(random_forest_threshold)
    ).astype(int)
    recorded_rf_prediction = frame[PREDICTION_COLUMN].to_numpy(dtype=int)
    if not np.array_equal(recorded_rf_prediction, expected_rf_prediction):
        raise ValueError(
            "Trusted Random Forest decisions do not match their frozen "
            "threshold and probabilities."
        )

    identity = frame[
        [*ROW_KEY_COLUMNS, TARGET_COLUMN]
    ].reset_index(drop=True)

    def variant(
        *,
        model: str,
        calibration_method: str,
        threshold: float,
        probability: np.ndarray,
    ) -> pd.DataFrame:
        output = identity.copy()
        output["model"] = model
        output["calibration_method"] = calibration_method
        output["decision_threshold"] = float(threshold)
        output["probability_poor_qoe"] = probability
        output["predicted_poor_qoe"] = (
            probability >= float(threshold)
        ).astype(int)
        return output

    return pd.concat(
        [
            variant(
                model="persistence",
                calibration_method="uncalibrated",
                threshold=0.5,
                probability=persistence_probability,
            ),
            variant(
                model="network_logistic",
                calibration_method=network_logistic_method,
                threshold=network_logistic_threshold,
                probability=logistic_probability,
            ),
            variant(
                model="cross_layer_random_forest",
                calibration_method=random_forest_method,
                threshold=random_forest_threshold,
                probability=random_forest_probability,
            ),
        ],
        ignore_index=True,
    )


def evaluate_three_model_test_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate each model on the same recorded test observations."""
    required_columns = [
        *ROW_KEY_COLUMNS,
        TARGET_COLUMN,
        "model",
        "calibration_method",
        "decision_threshold",
        "probability_poor_qoe",
        "predicted_poor_qoe",
    ]
    require_feature_columns(
        predictions.columns,
        required_columns,
        context="three-model comparison predictions",
    )
    models = [
        "persistence",
        "network_logistic",
        "cross_layer_random_forest",
    ]
    if set(predictions["model"]) != set(models):
        raise ValueError(
            "Three-model comparison must contain persistence, network "
            "logistic and cross-layer Random Forest predictions."
        )

    reference = None
    rows: list[dict[str, Any]] = []
    for model in models:
        model_predictions = predictions.loc[
            predictions["model"].eq(model)
        ].copy()
        keys = _validated_row_keys(
            model_predictions,
            context=f"{model} comparison predictions",
        )
        keyed_target = pd.Series(
            model_predictions[TARGET_COLUMN].to_numpy(dtype=int),
            index=keys,
        ).sort_index()
        if reference is None:
            reference = keyed_target
        elif not keyed_target.equals(reference):
            raise ValueError(
                "The three compared models do not use identical test rows "
                "and targets."
            )

        methods = model_predictions["calibration_method"].unique()
        thresholds = model_predictions["decision_threshold"].unique()
        if len(methods) != 1 or len(thresholds) != 1:
            raise ValueError(
                f"{model} comparison metadata is not frozen to one method "
                "and threshold."
            )
        metrics = evaluate_binary_predictions(
            target=model_predictions[TARGET_COLUMN],
            prediction=model_predictions["predicted_poor_qoe"],
            probability=model_predictions["probability_poor_qoe"],
        )
        rows.append(
            {
                "model": model,
                "calibration_method": methods[0],
                "decision_threshold": float(thresholds[0]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


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
    calibration_method: str,
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
        label=f"{calibration_method.capitalize()} calibrated model",
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
    """Run and save the complete frozen evaluation."""
    configuration = load_resolved_configuration(
        CONFIG_DIRECTORY
    )
    (
        calibration_method,
        calibrator_path,
        selected_random_forest,
    ) = load_selected_final_calibrator(
        CONFIGURATION_PATH,
        CALIBRATED_DIRECTORY,
        configuration=configuration,
    )
    random_forest_threshold_value = selected_random_forest.get(
        "decision_threshold"
    )
    if (
        type(random_forest_threshold_value) not in {int, float}
        or not math.isfinite(float(random_forest_threshold_value))
        or not 0.0 < float(random_forest_threshold_value) < 1.0
    ):
        raise ValueError(
            "Final Random Forest decision_threshold must be a finite number "
            "strictly between zero and one."
        )
    random_forest_threshold = float(random_forest_threshold_value)
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
    (
        network_logistic_method,
        network_logistic_calibrator_path,
        network_logistic_threshold,
    ) = load_network_logistic_selection(
        CONFIGURATION_PATH,
        CALIBRATED_DIRECTORY,
        expected_configuration_sha256=configuration.sha256,
    )
    for model_path, label in (
        (PERSISTENCE_MODEL_PATH, "persistence baseline"),
        (NETWORK_LOGISTIC_MODEL_PATH, "network logistic model"),
    ):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Required final {label} artifact is missing: {model_path}"
            )

    persistence_model = joblib.load(PERSISTENCE_MODEL_PATH)
    network_logistic_model = joblib.load(NETWORK_LOGISTIC_MODEL_PATH)
    network_logistic_calibrator = joblib.load(
        network_logistic_calibrator_path
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

    COMPARISON_PREDICTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = load_evaluation_dataset()

    print(
        f"Evaluation observations: "
        f"{len(frame):,}"
    )

    comparison_predictions = build_three_model_test_predictions(
        frame,
        persistence_model=persistence_model,
        network_logistic_model=network_logistic_model,
        network_logistic_calibrator=network_logistic_calibrator,
        network_logistic_method=network_logistic_method,
        network_logistic_threshold=network_logistic_threshold,
        random_forest_method=calibration_method,
        random_forest_threshold=random_forest_threshold,
    )
    comparison_metrics = evaluate_three_model_test_predictions(
        comparison_predictions
    )
    comparison_predictions.to_parquet(
        COMPARISON_PREDICTION_PATH,
        index=False,
    )
    comparison_metrics.to_csv(
        COMPARISON_TABLE_PATH,
        index=False,
    )
    comparison_document = {
        "configuration_sha256": configuration.sha256,
        "sealed_test_rows": len(frame),
        "models": comparison_metrics.to_dict(orient="records"),
    }
    COMPARISON_METRIC_PATH.write_text(
        json.dumps(
            comparison_document,
            indent=2,
            default=json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
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
        reliability,
        calibration_method,
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
        "three_model_test_metrics": comparison_metrics.to_dict(
            orient="records"
        ),
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

    print("\nThree-model sealed-test comparison")
    print(
        comparison_metrics.to_string(index=False)
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
    print(f"- {COMPARISON_PREDICTION_PATH}")
    print(f"- {COMPARISON_TABLE_PATH}")
    print(f"- {COMPARISON_METRIC_PATH}")
    print(f"- {TABLE_DIRECTORY}")
    print(f"- {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()
