"""Scientific evaluation utilities for the QoE digital twin."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from qoe_twin.calibration import (
    calibration_table,
    expected_calibration_error,
)


def evaluate_binary_predictions(
    target: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Calculate classification and probability-quality metrics."""
    y_true = np.asarray(target, dtype=int)
    y_pred = np.asarray(prediction, dtype=int)
    y_probability = np.asarray(probability, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics: dict[str, Any] = {
        "observations": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_probability,
            )
        ),
        "expected_calibration_error": float(
            expected_calibration_error(
                y_true,
                y_probability,
                number_of_bins=10,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_alarm_rate": float(
            fp / (fp + tn)
            if fp + tn > 0
            else 0.0
        ),
        "miss_rate": float(
            fn / (fn + tp)
            if fn + tp > 0
            else 0.0
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        )
    else:
        metrics["roc_auc"] = None

    return metrics


def evaluate_groups(
    frame: pd.DataFrame,
    group_column: str,
    target_column: str,
    prediction_column: str,
    probability_column: str,
) -> pd.DataFrame:
    """Evaluate the model independently for each scenario group."""
    rows: list[dict[str, Any]] = []

    for group_value, group in frame.groupby(
        group_column,
        dropna=False,
    ):
        if group.empty:
            continue

        metrics = evaluate_binary_predictions(
            target=group[target_column],
            prediction=group[prediction_column],
            probability=group[probability_column],
        )

        metrics[group_column] = group_value
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    ordered_columns = [
        group_column,
        *[
            column
            for column in result.columns
            if column != group_column
        ],
    ]

    return result[ordered_columns]


def evaluate_trust_levels(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    """Calculate accuracy and coverage for every trust level."""
    total = len(frame)
    rows: list[dict[str, Any]] = []

    for trust_level, group in frame.groupby(
        "trust_level",
        dropna=False,
    ):
        accuracy = float(
            (
                group[prediction_column]
                == group[target_column]
            ).mean()
        )

        rows.append(
            {
                "trust_level": trust_level,
                "predictions": int(len(group)),
                "coverage": float(
                    len(group) / total
                ),
                "mean_trust_score": float(
                    group["trust_score"].mean()
                ),
                "mean_probability": float(
                    group[
                        "calibrated_probability"
                    ].mean()
                ),
                "actual_poor_qoe_rate": float(
                    group[target_column].mean()
                ),
                "accuracy": accuracy,
                "abstention_rate": float(
                    group["abstain"].mean()
                ),
            }
        )

    order = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    result = pd.DataFrame(rows)

    result["_order"] = (
        result["trust_level"]
        .map(order)
        .fillna(99)
    )

    return (
        result.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def lead_time_statistics(
    lead_time: pd.Series,
) -> dict[str, float | int]:
    """Summarize the warning lead-time distribution."""
    values = pd.to_numeric(
        lead_time,
        errors="coerce",
    ).dropna()

    return {
        "count": int(len(values)),
        "minimum_seconds": float(values.min()),
        "p25_seconds": float(
            values.quantile(0.25)
        ),
        "median_seconds": float(
            values.median()
        ),
        "mean_seconds": float(
            values.mean()
        ),
        "p75_seconds": float(
            values.quantile(0.75)
        ),
        "p90_seconds": float(
            values.quantile(0.90)
        ),
        "p95_seconds": float(
            values.quantile(0.95)
        ),
        "maximum_seconds": float(
            values.max()
        ),
    }


def build_reliability_table(
    frame: pd.DataFrame,
    target_column: str,
    probability_column: str,
) -> pd.DataFrame:
    """Create the reliability-diagram data table."""
    return calibration_table(
        target=frame[target_column].to_numpy(),
        probability=frame[
            probability_column
        ].to_numpy(),
        number_of_bins=10,
    )
