"""Synthetic tests for the currently implemented grounding contract."""

import pytest
from pydantic import ValidationError

from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator


def build_explanation(**overrides: object) -> Explanation:
    payload: dict[str, object] = {
        "prediction": "Poor QoE predicted",
        "confidence": "high",
        "summary": "The supplied measurements support the prediction.",
        "likely_causes": ["Inspect the supplied service measurements."],
        "recommended_operator_checks": ["Verify the measured values."],
        "limitations": "Only the supplied evidence was considered.",
    }
    payload.update(overrides)
    return Explanation.model_validate(payload)


def test_grounding_accepts_supported_quantitative_claims() -> None:
    explanation = build_explanation(
        summary="Throughput is 5.3 Mbps and current MOS is 2.1.",
        likely_causes=["Packet loss is 2.4 percent."],
        recommended_operator_checks=["Verify the 6000 kbps bitrate."],
    )
    evidence = {
        "throughput_mbps": 5.3,
        "current_mos": 2.1,
        "plr_percent": 2.4,
        "bitrate_kbps": 6000,
    }

    valid, errors = GroundingValidator().validate(
        explanation,
        evidence,
    )

    assert valid is True
    assert errors == []


def test_grounding_rejects_unsupported_quantitative_claim() -> None:
    explanation = build_explanation(
        summary="The current MOS is 9.9.",
    )

    valid, errors = GroundingValidator().validate(
        explanation,
        {"current_mos": 2.1},
    )

    assert valid is False
    assert errors == ["Unsupported number: 9.9"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prediction", " "),
        ("summary", " "),
        ("likely_causes", []),
        ("recommended_operator_checks", []),
        ("limitations", " "),
    ],
)
def test_explanation_schema_rejects_missing_required_content(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        build_explanation(**{field: value})


def test_explanation_schema_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        build_explanation(confidence="very_high")


def test_validator_reports_missing_fields_if_schema_is_bypassed() -> None:
    explanation = Explanation.model_construct(
        prediction="",
        confidence="",
        summary="",
        likely_causes=["Qualitative cause."],
        recommended_operator_checks=["Qualitative check."],
        limitations="Evidence only.",
    )

    valid, errors = GroundingValidator().validate(
        explanation,
        {},
    )

    assert valid is False
    assert set(errors) == {
        "Missing summary",
        "Missing confidence",
        "Missing prediction",
    }
