"""Tests for explanation generation without external API calls."""

import pytest

from qoe_twin.explanation_generator import ExplanationGenerator
from qoe_twin.explanation_schema import Explanation


class FakeExplanationClient:
    def __init__(self, explanation: Explanation) -> None:
        self.explanation = explanation
        self.prompt: str | None = None

    def generate_json(self, prompt: str) -> Explanation:
        self.prompt = prompt
        return self.explanation


def _explanation(evidence_key: str = "current_mos") -> Explanation:
    return Explanation(
        prediction="future_poor_qoe",
        confidence="medium",
        summary="The model predicts poor QoE.",
        likely_causes=[
            {
                "statement": "Current MOS is 2.1.",
                "evidence_keys": [evidence_key],
            }
        ],
        recommended_operator_checks=["Inspect the current MOS."],
        limitations="Only supplied evidence was considered.",
    )


def _row() -> dict[str, object]:
    return {
        "prediction": "future_poor_qoe",
        "prediction_probability": 0.81,
        "trust_score": 0.72,
        "trust_level": "medium",
        "current_mos": 2.1,
        "prediction_lead_seconds": 10.0,
    }


def test_generator_uses_fake_client_and_enforces_grounding() -> None:
    client = FakeExplanationClient(_explanation())
    generator = ExplanationGenerator(client=client)

    result = generator.explain(_row())

    assert result.prediction == "future_poor_qoe"
    assert client.prompt is not None
    assert "prediction_lead_seconds" not in client.prompt


def test_generator_rejects_invalid_grounding() -> None:
    client = FakeExplanationClient(_explanation("unknown_field"))
    generator = ExplanationGenerator(client=client)

    with pytest.raises(ValueError, match="failed grounding validation"):
        generator.explain(_row())
