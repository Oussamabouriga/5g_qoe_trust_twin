"""Tests for the strict explanation schema."""

import pytest
from pydantic import ValidationError

from qoe_twin.explanation_schema import Explanation


def _payload() -> dict[str, object]:
    return {
        "prediction": "future_poor_qoe",
        "confidence": "high",
        "summary": "The model predicts poor QoE.",
        "likely_causes": [
            {
                "statement": "Throughput is 5.3 Mbps.",
                "evidence_keys": ["throughput_mbps"],
            }
        ],
        "recommended_operator_checks": ["Inspect throughput."],
        "limitations": "Only supplied evidence was considered.",
    }


def test_schema_accepts_structured_cause_citations() -> None:
    explanation = Explanation.model_validate(_payload())

    assert explanation.likely_causes[0].evidence_keys == ["throughput_mbps"]


def test_schema_requires_likely_causes_field() -> None:
    payload = _payload()
    del payload["likely_causes"]

    with pytest.raises(ValidationError):
        Explanation.model_validate(payload)


def test_schema_allows_explicit_insufficient_evidence_with_no_causes() -> None:
    payload = _payload()
    payload["likely_causes"] = []
    payload["limitations"] = "Insufficient evidence supports a specific cause."

    explanation = Explanation.model_validate(payload)

    assert explanation.likely_causes == []


def test_schema_rejects_empty_causes_without_insufficient_evidence() -> None:
    payload = _payload()
    payload["likely_causes"] = []

    with pytest.raises(ValidationError, match="insufficient evidence"):
        Explanation.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prediction", "poor"),
        ("confidence", "very high"),
        ("recommended_operator_checks", []),
    ],
)
def test_schema_rejects_invalid_required_content(
    field: str,
    value: object,
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        Explanation.model_validate(payload)


def test_schema_rejects_empty_or_duplicate_citations() -> None:
    for evidence_keys in ([], ["throughput_mbps", "throughput_mbps"]):
        payload = _payload()
        payload["likely_causes"] = [
            {
                "statement": "Throughput is relevant.",
                "evidence_keys": evidence_keys,
            }
        ]

        with pytest.raises(ValidationError):
            Explanation.model_validate(payload)


def test_schema_rejects_extra_fields() -> None:
    payload = _payload()
    payload["invented_field"] = "not allowed"

    with pytest.raises(ValidationError):
        Explanation.model_validate(payload)
