from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator


def test_grounding_passes() -> None:
    explanation = Explanation(
        prediction="Poor QoE",
        confidence="high",
        summary="MOS is 2.1",
        likely_causes=[
            "The current MOS is 2.1."
        ],
        recommended_operator_checks=[
            "Inspect the current MOS."
        ],
        limitations="Only uses supplied evidence.",
    )

    evidence = {
        "current_mos": 2.1
    }

    validator = GroundingValidator()

    valid, errors = validator.validate(
        explanation,
        evidence,
    )

    assert valid
    assert errors == []


def test_grounding_rejects_fake_number() -> None:
    explanation = Explanation(
        prediction="Poor QoE",
        confidence="high",
        summary="MOS is 9.9",
        likely_causes=[
            "The current MOS is 9.9."
        ],
        recommended_operator_checks=[
            "Inspect the current MOS."
        ],
        limitations="Only uses supplied evidence.",
    )

    evidence = {
        "current_mos": 2.1
    }

    validator = GroundingValidator()

    valid, errors = validator.validate(
        explanation,
        evidence,
    )

    assert not valid
    assert any(
        "Unsupported number: 9.9" in error
        for error in errors
    )
