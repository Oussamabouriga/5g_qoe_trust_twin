"""Classification and probability-quality metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from qoe_twin.calibration import expected_calibration_error


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate classification and probability metrics."""
    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),
        "expected_calibration_error": float(
            expected_calibration_error(
                y_true,
                probability,
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
    }


def find_best_f1_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    minimum_threshold: float = 0.05,
    maximum_threshold: float = 0.95,
    step: float = 0.01,
) -> tuple[float, list[dict[str, float]]]:
    """Find the validation threshold that maximizes F1-score."""
    thresholds = np.arange(
        minimum_threshold,
        maximum_threshold + step,
        step,
    )

    results: list[dict[str, float]] = []

    for threshold in thresholds:
        prediction = (
            probability >= threshold
        ).astype(int)

        results.append(
            {
                "threshold": float(threshold),
                "f1": float(
                    f1_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                ),
                "precision": float(
                    precision_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                ),
            }
        )

    best = max(
        results,
        key=lambda item: (
            item["f1"],
            item["precision"],
        ),
    )

    return best["threshold"], results
