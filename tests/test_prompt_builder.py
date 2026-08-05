"""Tests for decision-specific LLM prompts."""

import json

import pytest

from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator
from qoe_twin.prompt_builder import build_prompt


@pytest.mark.parametrize(
    ("decision", "expected_instruction"),
    [
        ("future_poor_qoe", "poor-QoE forecast"),
        ("future_acceptable_qoe", "acceptable-QoE forecast"),
        ("abstain", "withheld a prediction"),
    ],
)
def test_build_prompt_dispatches_by_decision(
    decision: str,
    expected_instruction: str,
) -> None:
    evidence = {
        "prediction": decision,
        "prediction_probability": 0.31,
        "trust_score": 0.72,
        "trust_level": "medium",
        "current_mos": 2.4,
    }
    prompt = build_prompt(evidence)

    assert expected_instruction in prompt
    assert f'"prediction": "{decision}"' in prompt
    assert "calibrated probability of poor QoE" in prompt
    assert "heuristic reliability score" in prompt
    assert "not a probability of correctness" in prompt
    assert '"likely_causes": []' in prompt
    assert "{statement, evidence_keys}" in prompt
    if decision == "abstain":
        assert '"confidence": "low"' in prompt
    else:
        assert '"confidence": "medium"' in prompt

    assert "low | medium | high" not in prompt
    assert "exact_evidence_key" not in prompt
    assert '"recommended_operator_checks": [""]' not in prompt

    example_json = prompt.split(
        "Return ONLY this JSON schema:\n\n",
        maxsplit=1,
    )[1]
    explanation = Explanation.model_validate(json.loads(example_json))
    valid, errors = GroundingValidator().validate(explanation, evidence)

    assert valid is True
    assert errors == []


@pytest.mark.parametrize("decision", [None, "unknown"])
def test_build_prompt_rejects_missing_or_unknown_decision(
    decision: str | None,
) -> None:
    evidence = {} if decision is None else {"prediction": decision}

    with pytest.raises(ValueError, match="evidence.prediction"):
        build_prompt(evidence)


def test_non_abstained_prompt_requires_trust_level() -> None:
    with pytest.raises(ValueError, match="requires trust_level"):
        build_prompt(
            {
                "prediction": "future_poor_qoe",
                "current_mos": 2.4,
            }
        )
