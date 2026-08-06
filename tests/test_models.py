"""Tests for model pipelines and evaluation metrics."""

import numpy as np
import pandas as pd

from qoe_twin.metrics import (
    classification_metrics,
    find_best_f1_threshold,
)
from qoe_twin.models import (
    build_cross_layer_random_forest,
    build_network_logistic_regression,
)


def create_training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        {
            "throughput_mbps": [
                1.0,
                1.5,
                2.0,
                5.0,
                6.0,
                7.0,
                1.2,
                6.5,
            ],
            "prb": [
                5,
                5,
                10,
                20,
                25,
                30,
                5,
                30,
            ],
            "current_mos": [
                1.5,
                2.0,
                2.5,
                4.0,
                4.5,
                4.8,
                1.8,
                4.7,
            ],
            "resolution": [
                "1080p",
                "1080p",
                "720p",
                "720p",
                "1080p",
                "1080p",
                "720p",
                "1080p",
            ],
        }
    )

    target = np.array(
        [1, 1, 1, 0, 0, 0, 1, 0]
    )

    return frame, target


def test_network_logistic_predicts_probabilities() -> None:
    frame, target = create_training_frame()

    model = build_network_logistic_regression(
        numeric_features=[
            "throughput_mbps",
            "prb",
        ]
    )

    model.fit(
        frame[
            [
                "throughput_mbps",
                "prb",
            ]
        ],
        target,
    )

    probability = model.predict_proba(
        frame[
            [
                "throughput_mbps",
                "prb",
            ]
        ]
    )

    assert probability.shape == (8, 2)


def test_cross_layer_random_forest_predicts_probabilities() -> None:
    frame, target = create_training_frame()

    model = build_cross_layer_random_forest(
        numeric_features=[
            "throughput_mbps",
            "prb",
            "current_mos",
        ],
        categorical_features=[
            "resolution",
        ],
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=1,
        n_jobs=1,
    )

    model.fit(
        frame,
        target,
    )

    probability = model.predict_proba(frame)

    assert probability.shape == (8, 2)


def test_metrics_are_bounded() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = classification_metrics(
        target,
        probability,
        threshold=0.5,
    )

    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_threshold_search_returns_valid_threshold() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])

    threshold, results = find_best_f1_threshold(
        target,
        probability,
    )

    assert 0.05 <= threshold <= 0.95
    assert len(results) > 0
