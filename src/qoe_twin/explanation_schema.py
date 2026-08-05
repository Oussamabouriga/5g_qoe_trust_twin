"""Strict schema for generated QoE explanations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

PredictionDecision = Literal[
    "future_poor_qoe",
    "future_acceptable_qoe",
    "abstain",
]


class LikelyCause(BaseModel):
    """One proposed cause with explicit operational evidence citations."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    statement: str = Field(
        min_length=1,
        max_length=400,
    )

    evidence_keys: list[str] = Field(
        min_length=1,
        max_length=5,
    )

    @field_validator("evidence_keys")
    @classmethod
    def validate_evidence_keys(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]

        if not cleaned:
            raise ValueError("At least one evidence key is required.")

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Evidence keys must be unique within a cause.")

        return cleaned


class Explanation(BaseModel):
    """Structured explanation accepted by the Digital Twin."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    prediction: PredictionDecision

    confidence: Literal[
        "low",
        "medium",
        "high",
    ]

    summary: str = Field(
        min_length=1,
        max_length=800,
    )

    likely_causes: list[LikelyCause] = Field(
        ...,
        max_length=5,
    )

    recommended_operator_checks: list[str] = Field(
        min_length=1,
        max_length=5,
    )

    limitations: str = Field(
        min_length=1,
        max_length=500,
    )

    @field_validator("recommended_operator_checks")
    @classmethod
    def validate_nonempty_items(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]

        if not cleaned:
            raise ValueError("At least one non-empty item is required.")

        return cleaned

    @model_validator(mode="after")
    def require_explicit_insufficient_evidence_for_no_causes(self) -> Self:
        if self.likely_causes:
            return self

        narrative = f"{self.summary} {self.limitations}".lower()

        if "insufficient evidence" not in narrative:
            raise ValueError(
                "Empty likely_causes requires an explicit "
                "'insufficient evidence' statement."
            )

        return self
