"""Tests for complete model evaluation."""

import numpy as np
import pandas as pd

from qoe_twin.evaluation import (
    evaluate_binary_predictions,
    evaluate_groups,
    evaluate_trust_levels,
    lead_time_statistics,
)


def create_evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "prediction": [0, 1, 1, 1],
            "probability": [
                0.1,
                0.7,
                0.8,
                0.9,
            ],
            "base_station": [
                1,
                1,
                2,
                2,
            ],
            "trust_level": [
                "high",
                "low",
                "high",
                "medium",
            ],
            "trust_score": [
                0.9,
                0.4,
                0.95,
                0.7,
            ],
            "calibrated_probability": [
                0.1,
                0.7,
                0.8,
                0.9,
            ],
            "abstain": [
                False,
                True,
                False,
                False,
            ],
            "lead_time": [
                5.0,
                10.0,
                15.0,
                20.0,
            ],
        }
    )


def test_binary_evaluation_returns_metrics() -> None:
    frame = create_evaluation_frame()

    metrics = evaluate_binary_predictions(
        target=frame["target"],
        prediction=frame["prediction"],
        probability=frame["probability"],
    )

    assert metrics["observations"] == 4
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["brier_score"] <= 1


def test_group_evaluation_returns_each_group() -> None:
    frame = create_evaluation_frame()

    result = evaluate_groups(
        frame=frame,
        group_column="base_station",
        target_column="target",
        prediction_column="prediction",
        probability_column="probability",
    )

    assert set(
        result["base_station"]
    ) == {1, 2}


def test_trust_evaluation_contains_levels() -> None:
    frame = create_evaluation_frame()

    result = evaluate_trust_levels(
        frame=frame,
        target_column="target",
        prediction_column="prediction",
    )

    assert set(
        result["trust_level"]
    ) == {
        "high",
        "medium",
        "low",
    }


def test_lead_time_statistics() -> None:
    statistics = lead_time_statistics(
        pd.Series(
            [5.0, 10.0, 15.0, 20.0]
        )
    )

    assert statistics["count"] == 4
    assert statistics["median_seconds"] == 12.5
    assert statistics["minimum_seconds"] == 5.0
    assert statistics["maximum_seconds"] == 20.0


def test_binary_metrics_with_numpy_arrays() -> None:
    metrics = evaluate_binary_predictions(
        target=np.array([0, 1]),
        prediction=np.array([0, 1]),
        probability=np.array([0.1, 0.9]),
    )

    assert metrics["accuracy"] == 1.0
