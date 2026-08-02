"""Prompt builder for QoE explanations."""

from __future__ import annotations

import json


SYSTEM_PROMPT = """
You are a senior 5G network engineer.

Your task is to explain a predicted future QoE degradation.

Rules:

- Use ONLY the evidence provided.
- Never invent numbers.
- Never invent causes.
- Never mention fields that are not present.
- Return ONLY valid JSON.
"""


def build_prompt(
    evidence: dict,
) -> str:

    return (
        SYSTEM_PROMPT
        + "\n\nEvidence:\n"
        + json.dumps(
            evidence,
            indent=2,
        )
        + """

Return ONLY this JSON schema:

{
  "prediction": "",
  "confidence": "",
  "summary": "",
  "likely_causes": [],
  "recommended_operator_checks": [],
  "limitations": ""
}
"""
    )
