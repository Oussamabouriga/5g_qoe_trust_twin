from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator

explanation = Explanation(
    prediction="Poor QoE",
    confidence="high",
    summary="Current MOS is 2.1",
    likely_causes=[],
    recommended_operator_checks=[],
    limitations="Evidence only.",
)

evidence = {
    "current_mos": 2.1
}

validator = GroundingValidator()

valid, errors = validator.validate(
    explanation,
    evidence,
)

print("VALID:", valid)
print("ERRORS:", errors)
