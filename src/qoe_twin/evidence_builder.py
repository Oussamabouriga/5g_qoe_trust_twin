"""Create structured evidence sent to the LLM."""

from __future__ import annotations

from typing import Any

FIELDS = (
    "prediction",
    "prediction_probability",
    "trust_score",
    "trust_level",
    "current_mos",
    "throughput_mbps",
    "bitrate_kbps",
    "capacity_margin_mbps",
    "throughput_to_bitrate_ratio",
    "plr_percent",
)


def build_evidence(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Return only operational, present-time evidence fields."""

    evidence: dict[str, Any] = {}

    for field in FIELDS:
        if field in row:
            evidence[field] = row[field]

    return evidence
