from qoe_twin.prompt_builder import build_prompt


def test_prompt_contains_evidence():

    prompt = build_prompt(
        {
            "prediction_probability": 0.91
        }
    )

    assert "prediction_probability" in prompt
    assert "Return ONLY this JSON schema" in prompt
