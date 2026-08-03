"""Tests for prediction trust scoring."""

import pandas as pd
import pytest

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


def test_probability_confidence_is_symmetric() -> None:
    assert probability_confidence(0.25) == 0.5
    assert probability_confidence(0.75) == 0.5


@pytest.mark.parametrize("probability", [-1.0, 2.0])
def test_probability_confidence_clips_out_of_range_values(
    probability: float,
) -> None:
    assert probability_confidence(probability) == 1.0


def test_probability_confidence_rejects_non_numeric_input() -> None:
    with pytest.raises(ValueError):
        probability_confidence("invalid")  # type: ignore[arg-type]


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


def test_data_quality_with_no_required_features_is_complete() -> None:
    assert calculate_data_quality(pd.Series(dtype=float), []) == 1.0


def test_data_quality_counts_absent_features_as_unusable() -> None:
    row = pd.Series({"available": 1.0})

    score = calculate_data_quality(
        row,
        ["available", "absent"],
    )

    assert score == 0.5


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


def test_stability_without_history_is_complete() -> None:
    assert calculate_prediction_stability(0.8, []) == 1.0


def test_stability_uses_only_the_configured_recent_window() -> None:
    stability = calculate_prediction_stability(
        probability=0.8,
        previous_probabilities=[0.0, 0.8, 0.8, 0.8],
        window=3,
    )

    assert stability == 1.0


def test_trust_matches_hand_calculated_weighted_score() -> None:
    result = calculate_trust(
        probability=0.9,
        validation_f1=0.8,
        validation_ece=0.1,
        data_quality=0.5,
        prediction_stability=0.75,
        confidence_weight=0.25,
        validation_weight=0.25,
        data_quality_weight=0.25,
        stability_weight=0.25,
        abstention_threshold=0.55,
    )

    # confidence = 0.8; validation quality = 0.7*0.8 + 0.3*0.9 = 0.83
    # trust = 0.25 * (0.8 + 0.83 + 0.5 + 0.75) = 0.72
    assert result.model_confidence == pytest.approx(0.8)
    assert result.validation_quality == pytest.approx(0.83)
    assert result.trust_score == pytest.approx(0.72)
    assert result.trust_level == "medium"
    assert result.abstain is False


def test_trust_clips_quality_inputs_for_score_calculation() -> None:
    result = calculate_trust(
        probability=1.0,
        validation_f1=2.0,
        validation_ece=-0.5,
        data_quality=2.0,
        prediction_stability=-1.0,
        confidence_weight=0.25,
        validation_weight=0.25,
        data_quality_weight=0.25,
        stability_weight=0.25,
        abstention_threshold=0.55,
    )

    # Clipped components are confidence=1, validation=1, data=1, stability=0.
    assert result.validation_quality == pytest.approx(1.0)
    assert result.trust_score == pytest.approx(0.75)
    assert result.trust_level == "medium"
    assert result.abstain is False


@pytest.mark.parametrize(
    ("probability", "expected_level", "expected_abstention"),
    [
        (0.9, "high", False),
        (0.775, "medium", False),
        (0.7745, "low", True),
    ],
)
def test_trust_level_boundaries(
    probability: float,
    expected_level: str,
    expected_abstention: bool,
) -> None:
    result = calculate_trust(
        probability=probability,
        validation_f1=0.0,
        validation_ece=1.0,
        data_quality=0.0,
        prediction_stability=0.0,
        confidence_weight=1.0,
        validation_weight=0.0,
        data_quality_weight=0.0,
        stability_weight=0.0,
        abstention_threshold=0.55,
    )

    assert result.trust_level == expected_level
    assert result.abstain is expected_abstention


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
