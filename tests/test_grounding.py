"""Focused synthetic tests for deterministic LLM grounding checks."""

import pytest

from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator


def _evidence(decision: str = "future_poor_qoe") -> dict[str, object]:
    return {
        "prediction": decision,
        "prediction_probability": 0.81,
        "trust_score": 0.72,
        "trust_level": "medium",
        "current_mos": 2.1,
        "throughput_mbps": 5.3,
        "bitrate_kbps": 6000,
        "capacity_margin_mbps": -0.7,
        "throughput_to_bitrate_ratio": 0.88,
        "plr_percent": 2.4,
    }


def _explanation(**overrides: object) -> Explanation:
    payload: dict[str, object] = {
        "prediction": "future_poor_qoe",
        "confidence": "medium",
        "summary": "The model predicts poor QoE at the next observation.",
        "likely_causes": [
            {
                "statement": "Packet loss is 2.4 percent.",
                "evidence_keys": ["plr_percent"],
            }
        ],
        "recommended_operator_checks": ["Verify the 6000 kbps bitrate."],
        "limitations": "Only supplied evidence was considered.",
    }
    payload.update(overrides)
    return Explanation.model_validate(payload)


def test_grounding_accepts_supported_cited_claims() -> None:
    explanation = _explanation(
        summary="Throughput is 5.3 Mbps and current MOS is 2.1.",
        likely_causes=[
            {
                "statement": "Packet loss is 2.4 percent.",
                "evidence_keys": ["plr_percent"],
            },
            {
                "statement": "Throughput-to-bitrate ratio is 0.88.",
                "evidence_keys": ["throughput_to_bitrate_ratio"],
            },
        ],
    )

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is True
    assert errors == []


@pytest.mark.parametrize(
    ("decision", "summary", "expected_error"),
    [
        (
            "future_acceptable_qoe",
            "The model predicts poor QoE.",
            "described as a degradation forecast",
        ),
        (
            "future_poor_qoe",
            "The model predicts acceptable QoE.",
            "described as acceptable QoE",
        ),
        (
            "future_acceptable_qoe",
            "Poor QoE is likely.",
            "described as a degradation forecast",
        ),
        (
            "future_poor_qoe",
            "QoE should remain acceptable.",
            "described as acceptable QoE",
        ),
    ],
)
def test_grounding_rejects_prediction_narrative_contradictions(
    decision: str,
    summary: str,
    expected_error: str,
) -> None:
    explanation = _explanation(prediction=decision, summary=summary)

    valid, errors = GroundingValidator().validate(
        explanation,
        _evidence(decision),
    )

    assert valid is False
    assert any(expected_error in error for error in errors)


@pytest.mark.parametrize(
    ("decision", "summary"),
    [
        ("future_acceptable_qoe", "Poor QoE is not likely."),
        ("future_poor_qoe", "QoE should not remain acceptable."),
        ("future_acceptable_qoe", "The model does not predict poor QoE."),
        ("future_acceptable_qoe", "The model cannot predict poor QoE."),
        ("future_poor_qoe", "The model does not predict acceptable QoE."),
        ("future_poor_qoe", "The model cannot predict acceptable QoE."),
    ],
)
def test_grounding_preserves_immediate_negations(
    decision: str,
    summary: str,
) -> None:
    explanation = _explanation(prediction=decision, summary=summary)

    valid, errors = GroundingValidator().validate(
        explanation,
        _evidence(decision),
    )

    assert valid is True
    assert errors == []


def test_grounding_rejects_prediction_field_mismatch() -> None:
    explanation = _explanation(prediction="future_acceptable_qoe")

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is False
    assert any("contradicts evidence prediction" in error for error in errors)


def test_grounding_rejects_confident_abstention_with_causes() -> None:
    explanation = _explanation(
        prediction="abstain",
        confidence="high",
        summary="Insufficient evidence supports a prediction.",
    )

    valid, errors = GroundingValidator().validate(
        explanation,
        _evidence("abstain"),
    )

    assert valid is False
    assert "Abstention explanation confidence must be low." in errors
    assert "Abstention explanation must not contain likely causes." in errors


def test_grounding_accepts_abstention_without_causes() -> None:
    explanation = _explanation(
        prediction="abstain",
        confidence="low",
        summary="The system abstained because reliability was insufficient.",
        likely_causes=[],
        limitations="No cause is stated because of insufficient evidence.",
    )

    valid, errors = GroundingValidator().validate(
        explanation,
        _evidence("abstain"),
    )

    assert valid is True
    assert errors == []


def test_grounding_rejects_unknown_citation() -> None:
    explanation = _explanation(
        likely_causes=[
            {
                "statement": "A service measurement may be relevant.",
                "evidence_keys": ["invented_measurement"],
            }
        ]
    )

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is False
    assert any("unknown evidence key" in error for error in errors)


def test_grounding_rejects_unsupported_number() -> None:
    explanation = _explanation(summary="Current MOS is 9.9.")

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is False
    assert "Unsupported number: 9.9" in errors


def test_grounding_rejects_wrong_field_attribution() -> None:
    explanation = _explanation(
        likely_causes=[
            {
                "statement": "Current MOS is 5.3 Mbps.",
                "evidence_keys": ["throughput_mbps"],
            }
        ]
    )

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is False
    assert any("value of current_mos" in error for error in errors)
    assert any("without citing it" in error for error in errors)


def test_grounding_rejects_wrong_unit() -> None:
    explanation = _explanation(
        likely_causes=[
            {
                "statement": "Throughput is 5.3 kbps.",
                "evidence_keys": ["throughput_mbps"],
            }
        ]
    )

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is False
    assert any("Wrong unit for throughput_mbps" in error for error in errors)


def test_grounding_rejects_recognized_incompatible_unit() -> None:
    explanation = _explanation(summary="Throughput is 5.3 Gbps.")

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is False
    assert "Wrong unit for throughput_mbps: Gbps" in errors


def test_grounding_rejects_wrong_field_with_measured_wording() -> None:
    explanation = _explanation(summary="Throughput measured 2.1 Mbps.")

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is False
    assert "Claimed value 2.1 is not the value of throughput_mbps." in errors


def test_grounding_rejects_confidence_trust_level_mismatch() -> None:
    explanation = _explanation(confidence="high")

    valid, errors = GroundingValidator().validate(explanation, _evidence())

    assert valid is False
    assert any("contradicts evidence trust_level" in error for error in errors)
