"""Generate grounded explanations from Digital Twin evidence."""

from __future__ import annotations

from qoe_twin.evidence_builder import build_evidence
from qoe_twin.openai_client import OpenAIExplanationClient
from qoe_twin.prompt_builder import build_prompt
from qoe_twin.grounding_validator import GroundingValidator


class ExplanationGenerator:

    def __init__(self) -> None:
        self.client = OpenAIExplanationClient()
        self.validator = GroundingValidator()

    def explain(
        self,
        row: dict,
    ) -> dict:

        evidence = build_evidence(row)

        prompt = build_prompt(
            evidence
        )

        explanation = self.client.generate_json(
            prompt
        )

        return explanation
