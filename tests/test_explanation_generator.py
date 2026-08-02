from qoe_twin.evidence_builder import build_evidence
from qoe_twin.prompt_builder import build_prompt


def test_prompt_generation_pipeline():

    row = {
        "prediction_probability": 0.9,
        "trust_score": 0.8,
    }

    evidence = build_evidence(row)

    prompt = build_prompt(
        evidence
    )

    assert "prediction_probability" in prompt
    assert "trust_score" in prompt
