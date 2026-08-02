"""Strict schema for generated QoE explanations."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class Explanation(BaseModel):
    """Structured explanation accepted by the Digital Twin."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    prediction: str = Field(
        min_length=1,
        max_length=160,
    )

    confidence: Literal[
        "low",
        "medium",
        "high",
    ]

    summary: str = Field(
        min_length=1,
        max_length=800,
    )

    likely_causes: list[str] = Field(
        min_length=1,
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

    @field_validator(
        "likely_causes",
        "recommended_operator_checks",
    )
    @classmethod
    def validate_nonempty_items(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned = [
            value.strip()
            for value in values
            if value.strip()
        ]

        if not cleaned:
            raise ValueError(
                "At least one non-empty item is required."
            )

        return cleaned
