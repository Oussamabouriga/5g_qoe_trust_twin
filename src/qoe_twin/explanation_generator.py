"""Generate grounded explanations from Digital Twin evidence."""

from __future__ import annotations

from qoe_twin.evidence_builder import build_evidence
from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator
from qoe_twin.openai_client import OpenAIExplanationClient
from qoe_twin.prompt_builder import build_prompt


class ExplanationGenerator:

    def __init__(
        self,
        client: OpenAIExplanationClient | None = None,
        validator: GroundingValidator | None = None,
    ) -> None:
        self.client = client or OpenAIExplanationClient()
        self.validator = validator or GroundingValidator()

    def explain(
        self,
        row: dict,
    ) -> Explanation:

        evidence = build_evidence(row)

        prompt = build_prompt(
            evidence
        )

        explanation = self.client.generate_json(
            prompt
        )

        valid, errors = self.validator.validate(
            explanation,
            evidence,
        )

        if not valid:
            raise ValueError(
                "Explanation failed grounding validation: "
                + "; ".join(errors)
            )

        return explanation
