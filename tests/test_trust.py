"""Tests for prediction trust scoring."""

import pandas as pd
import pytest

from qoe_twin.trust import (
    TrustPolicy,
    calculate_configured_trust,
    calculate_data_quality,
    calculate_prediction_stability,
    calculate_trust,
    probability_confidence,
    require_selected_abstention_threshold,
    risk_coverage_table,
    select_validation_abstention_threshold,
)


def test_probability_confidence() -> None:
    assert probability_confidence(0.5, 0.5) == 0.0
    assert probability_confidence(0.0, 0.5) == 1.0
    assert probability_confidence(1.0, 0.5) == 1.0


def test_probability_confidence_is_symmetric() -> None:
    assert probability_confidence(0.25, 0.5) == 0.5
    assert probability_confidence(0.75, 0.5) == 0.5


def test_probability_confidence_uses_selected_decision_threshold() -> None:
    assert probability_confidence(0.3, 0.3) == 0.0
    assert probability_confidence(0.0, 0.3) == 1.0
    assert probability_confidence(1.0, 0.3) == 1.0
    assert probability_confidence(0.65, 0.3) == pytest.approx(0.5)
    assert probability_confidence(0.65, 0.5) == pytest.approx(0.3)


@pytest.mark.parametrize("probability", [-1.0, 2.0])
def test_probability_confidence_clips_out_of_range_values(
    probability: float,
) -> None:
    assert probability_confidence(probability, 0.5) == 1.0


def test_probability_confidence_rejects_non_numeric_input() -> None:
    with pytest.raises(ValueError):
        probability_confidence("invalid", 0.5)  # type: ignore[arg-type]


def test_probability_confidence_rejects_nonfinite_probability() -> None:
    with pytest.raises(ValueError, match="probability must be finite"):
        probability_confidence(float("nan"), 0.5)


def _configured_policy() -> TrustPolicy:
    return TrustPolicy.from_mapping(
        {
            "trust": {
                "confidence_weight": 0.4,
                "validation_performance_weight": 0.3,
                "calibration_weight": 0.2,
                "data_quality_weight": 0.1,
            },
            "levels": {"high": 0.8, "medium": 0.55},
            "abstention": {"enabled": True, "minimum_coverage": 0.9},
        }
    )


def test_configured_trust_uses_exact_yaml_terms_without_stability() -> None:
    policy = _configured_policy()
    low_stability = calculate_configured_trust(
        probability=0.9,
        decision_threshold=0.5,
        validation_f1=0.8,
        validation_ece=0.1,
        data_quality=0.5,
        prediction_stability=0.0,
        policy=policy,
        abstention_threshold=0.7,
    )
    high_stability = calculate_configured_trust(
        probability=0.9,
        decision_threshold=0.5,
        validation_f1=0.8,
        validation_ece=0.1,
        data_quality=0.5,
        prediction_stability=1.0,
        policy=policy,
        abstention_threshold=0.7,
    )

    # 0.4*0.8 confidence + 0.3*0.8 F1 + 0.2*0.9 calibration
    # quality + 0.1*0.5 data quality = 0.79.
    assert low_stability.trust_score == pytest.approx(0.79)
    assert high_stability.trust_score == pytest.approx(0.79)
    assert low_stability.validation_performance == pytest.approx(0.8)
    assert low_stability.calibration_quality == pytest.approx(0.9)
    assert low_stability.trust_level == "medium"
    assert low_stability.abstain is False


@pytest.mark.parametrize(
    "nonfinite_field",
    [
        "probability",
        "validation_f1",
        "validation_ece",
        "data_quality",
        "prediction_stability",
    ],
)
def test_configured_trust_rejects_nonfinite_inputs(
    nonfinite_field: str,
) -> None:
    inputs = {
        "probability": 0.8,
        "decision_threshold": 0.5,
        "validation_f1": 0.8,
        "validation_ece": 0.1,
        "data_quality": 1.0,
        "prediction_stability": 1.0,
        "policy": _configured_policy(),
        "abstention_threshold": 0.6,
    }
    inputs[nonfinite_field] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        calculate_configured_trust(**inputs)  # type: ignore[arg-type]


def test_cp8_abstention_threshold_has_no_implicit_fallback() -> None:
    with pytest.raises(ValueError, match="CP8 validation-selected"):
        require_selected_abstention_threshold({})

    assert require_selected_abstention_threshold(
        {"abstention_threshold": 0.62}
    ) == pytest.approx(0.62)


def _validation_selection_frame() -> pd.DataFrame:
    target = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    prediction = target.copy()
    prediction[0] = 1
    return pd.DataFrame(
        {
            "split": ["validation"] * 10,
            "target": target,
            "prediction": prediction,
            "trust_score": [index / 10 for index in range(1, 11)],
        }
    )


def test_risk_coverage_table_has_exact_counts_and_risk() -> None:
    frame = _validation_selection_frame()
    table = risk_coverage_table(
        target=frame["target"],
        prediction=frame["prediction"],
        trust_score=frame["trust_score"],
    )

    full_coverage = table.loc[
        table["abstention_threshold"].eq(0.1)
    ].iloc[0]
    ninety_percent = table.loc[
        table["abstention_threshold"].eq(0.2)
    ].iloc[0]
    assert full_coverage["accepted_predictions"] == 10
    assert full_coverage["coverage"] == pytest.approx(1.0)
    assert full_coverage["selective_risk"] == pytest.approx(0.1)
    assert ninety_percent["accepted_predictions"] == 9
    assert ninety_percent["abstained_predictions"] == 1
    assert ninety_percent["coverage"] == pytest.approx(0.9)
    assert ninety_percent["selective_risk"] == pytest.approx(0.0)


def test_risk_coverage_table_aggregates_tied_scores_once() -> None:
    table = risk_coverage_table(
        target=[0, 1, 0, 1],
        prediction=[1, 1, 0, 1],
        trust_score=[0.1, 0.2, 0.2, 0.3],
    )

    assert table["abstention_threshold"].tolist() == [0.1, 0.2, 0.3]
    assert table["accepted_predictions"].tolist() == [4, 3, 1]
    assert table["coverage"].tolist() == pytest.approx([1.0, 0.75, 0.25])
    assert table["selective_risk"].tolist() == pytest.approx([0.25, 0.0, 0.0])


def test_abstention_selector_minimizes_validation_risk_at_90_coverage() -> None:
    threshold, table = select_validation_abstention_threshold(
        _validation_selection_frame(),
        target_column="target",
        prediction_column="prediction",
        minimum_coverage=0.9,
    )

    assert threshold == pytest.approx(0.2)
    selected = table.loc[
        table["abstention_threshold"].eq(threshold)
    ].iloc[0]
    assert selected["coverage"] == pytest.approx(0.9)
    assert selected["selective_risk"] == pytest.approx(0.0)


def test_abstention_selector_rejects_test_rows_and_lower_coverage() -> None:
    frame = _validation_selection_frame()
    frame.loc[0, "split"] = "test"

    with pytest.raises(ValueError, match="validation rows only"):
        select_validation_abstention_threshold(
            frame,
            target_column="target",
            prediction_column="prediction",
            minimum_coverage=0.9,
        )
    with pytest.raises(ValueError, match="between 0.90 and 1.0"):
        select_validation_abstention_threshold(
            _validation_selection_frame(),
            target_column="target",
            prediction_column="prediction",
            minimum_coverage=0.89,
        )


def test_abstention_selector_rejects_missing_split_labels() -> None:
    frame = _validation_selection_frame()
    frame["split"] = frame["split"].astype("string")
    frame.loc[0, "split"] = pd.NA

    with pytest.raises(ValueError, match="validation rows only"):
        select_validation_abstention_threshold(
            frame,
            target_column="target",
            prediction_column="prediction",
            minimum_coverage=0.9,
        )


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
