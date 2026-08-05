from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator

explanation = Explanation(
    prediction="future_poor_qoe",
    confidence="high",
    summary="The model forecasts poor QoE.",
    likely_causes=[
        {
            "statement": "Current MOS is 2.1.",
            "evidence_keys": ["current_mos"],
        }
    ],
    recommended_operator_checks=["Inspect the current MOS."],
    limitations="Only supplied evidence was considered.",
)

evidence = {
    "prediction": "future_poor_qoe",
    "trust_level": "high",
    "current_mos": 2.1,
}

validator = GroundingValidator()

valid, errors = validator.validate(
    explanation,
    evidence,
)

print("VALID:", valid)
print("ERRORS:", errors)
