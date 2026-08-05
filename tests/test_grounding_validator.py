"""Regression tests for citation-level numerical grounding."""

from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator


def test_cause_number_must_come_from_its_cited_field() -> None:
    explanation = Explanation(
        prediction="future_poor_qoe",
        confidence="high",
        summary="The model predicts poor QoE.",
        likely_causes=[
            {
                "statement": "Packet loss is 5.3 percent.",
                "evidence_keys": ["plr_percent"],
            }
        ],
        recommended_operator_checks=["Inspect packet loss."],
        limitations="Only supplied evidence was considered.",
    )
    evidence = {
        "prediction": "future_poor_qoe",
        "trust_level": "high",
        "throughput_mbps": 5.3,
        "plr_percent": 2.4,
    }

    valid, errors = GroundingValidator().validate(explanation, evidence)

    assert valid is False
    assert any("not supported by its cited evidence keys" in error for error in errors)


def test_model_metadata_cannot_be_cited_as_a_cause() -> None:
    explanation = Explanation(
        prediction="future_poor_qoe",
        confidence="high",
        summary="The model predicts poor QoE.",
        likely_causes=[
            {
                "statement": "The prediction probability is 0.81.",
                "evidence_keys": ["prediction_probability"],
            }
        ],
        recommended_operator_checks=["Inspect service measurements."],
        limitations="Only supplied evidence was considered.",
    )
    evidence = {
        "prediction": "future_poor_qoe",
        "trust_level": "high",
        "prediction_probability": 0.81,
    }

    valid, errors = GroundingValidator().validate(explanation, evidence)

    assert valid is False
    assert any("non-causal evidence key" in error for error in errors)
