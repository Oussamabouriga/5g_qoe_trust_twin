"""Tests for prediction trust scoring."""

import pandas as pd

from qoe_twin.trust import (
    calculate_data_quality,
    calculate_prediction_stability,
    calculate_trust,
    probability_confidence,
)


def test_probability_confidence() -> None:
    assert probability_confidence(0.5) == 0.0
    assert probability_confidence(0.0) == 1.0
    assert probability_confidence(1.0) == 1.0


def test_data_quality() -> None:
    row = pd.Series(
        {
            "a": 1.0,
            "b": None,
            "c": 2.0,
        }
    )

    score = calculate_data_quality(
        row,
        ["a", "b", "c"],
    )

    assert score == 2 / 3


def test_stability_is_high_for_constant_probabilities() -> None:
    stability = calculate_prediction_stability(
        probability=0.8,
        previous_probabilities=[
            0.8,
            0.8,
            0.8,
        ],
    )

    assert stability == 1.0


def test_low_confidence_can_abstain() -> None:
    result = calculate_trust(
        probability=0.5,
        validation_f1=0.8,
        validation_ece=0.05,
        data_quality=1.0,
        prediction_stability=1.0,
        abstention_threshold=0.55,
    )

    assert result.abstain is True


def test_high_confidence_is_trusted() -> None:
    result = calculate_trust(
        probability=0.95,
        validation_f1=0.85,
        validation_ece=0.03,
        data_quality=1.0,
        prediction_stability=1.0,
        abstention_threshold=0.55,
    )

    assert result.abstain is False
    assert result.trust_level in {
        "medium",
        "high",
    }
