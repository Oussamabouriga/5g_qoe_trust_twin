"""Decision-specific prompt builder for QoE explanations."""

from __future__ import annotations

import json
from typing import Any

DECISION_INSTRUCTIONS = {
    "future_poor_qoe": (
        "Explain the model's next-observation poor-QoE forecast. Do not claim "
        "that degradation has already occurred or that its onset time is known."
    ),
    "future_acceptable_qoe": (
        "Explain the model's next-observation acceptable-QoE forecast. Do not "
        "describe this decision as a degradation or poor-QoE forecast."
    ),
    "abstain": (
        "Explain why the system withheld a prediction. Set confidence to low and "
        "likely_causes to an empty list; do not present a confident causal diagnosis."
    ),
}

COMMON_RULES = """You are a senior 5G network engineer.

Rules:
- Use only the supplied evidence.
- prediction_probability is the calibrated probability of poor QoE at the next
  observation. It is not an observed outcome.
- trust_score is a heuristic reliability score, not a probability of correctness.
- Never invent numbers, fields, units, causes, future measurements, or timing.
- Every likely cause must cite one or more exact evidence keys in evidence_keys.
- Optional supported-cause item shape: {statement, evidence_keys}.
- For a non-abstained decision, confidence must equal the supplied trust_level.
- If the evidence supports no cause, use an empty likely_causes list and write
  "insufficient evidence" in the summary or limitations.
- Return only valid JSON.
"""


def build_prompt(
    evidence: dict[str, Any],
) -> str:
    """Build the prompt for one canonical prediction decision."""
    decision = evidence.get("prediction")

    if decision not in DECISION_INSTRUCTIONS:
        allowed = ", ".join(DECISION_INSTRUCTIONS)
        raise ValueError(
            "evidence.prediction must be one of: "
            f"{allowed}; received {decision!r}."
        )

    trust_level = evidence.get("trust_level")

    if decision != "abstain" and trust_level not in {"low", "medium", "high"}:
        raise ValueError(
            "Non-abstained evidence requires trust_level low, medium or high."
        )

    confidence_schema = "low" if decision == "abstain" else trust_level
    summary_schema = {
        "future_poor_qoe": "The model forecasts poor QoE at the next observation.",
        "future_acceptable_qoe": (
            "The model forecasts acceptable QoE at the next observation."
        ),
        "abstain": "The system abstains because reliability is insufficient.",
    }[decision]

    return (
        COMMON_RULES
        + "\nDecision-specific task:\n"
        + DECISION_INSTRUCTIONS[decision]
        + "\n\nEvidence:\n"
        + json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
        )
        + f"""

Return ONLY this JSON schema:

{{
  "prediction": "{decision}",
  "confidence": "{confidence_schema}",
  "summary": "{summary_schema}",
  "likely_causes": [],
  "recommended_operator_checks": ["Review the cited operational evidence."],
  "limitations": "Insufficient evidence supports a specific cause."
}}
"""
    )
