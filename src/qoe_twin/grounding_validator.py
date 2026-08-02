from __future__ import annotations

import re

from qoe_twin.explanation_schema import Explanation


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


class GroundingValidator:

    def validate(
        self,
        explanation: Explanation,
        evidence: dict,
    ) -> tuple[bool, list[str]]:

        errors: list[str] = []

        text = (
            explanation.summary
            + " "
            + " ".join(explanation.likely_causes)
            + " "
            + " ".join(explanation.recommended_operator_checks)
            + " "
            + explanation.limitations
        )

        evidence_numbers = {
            str(v)
            for v in evidence.values()
            if isinstance(v, (int, float))
        }

        generated_numbers = NUMBER_PATTERN.findall(text)

        for number in generated_numbers:

            if number not in evidence_numbers:
                errors.append(
                    f"Unsupported number: {number}"
                )

        if not explanation.summary.strip():
            errors.append("Missing summary")

        if not explanation.confidence.strip():
            errors.append("Missing confidence")

        if not explanation.prediction.strip():
            errors.append("Missing prediction")

        return len(errors) == 0, errors
