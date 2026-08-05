"""Deterministic checks for obvious explanation-grounding failures."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any

from qoe_twin.explanation_schema import Explanation, LikelyCause

NUMBER_SOURCE = r"-?\d+(?:\.\d+)?"
UNIT_SOURCE = (
    r"percentage points?|percentage|percent|mbps|kbps|gbps|bps|"
    r"milliseconds?|seconds?|ms|dbm?|%"
)
NUMBER_PATTERN = re.compile(
    rf"(?<![\w.])(?P<number>{NUMBER_SOURCE})(?!\w)"
)
NUMBER_WITH_UNIT_PATTERN = re.compile(
    rf"(?<![\w.])(?P<number>{NUMBER_SOURCE})(?![\w.])"
    rf"\s*(?P<unit>{UNIT_SOURCE})(?!\w)",
    re.IGNORECASE,
)

CAUSAL_EVIDENCE_KEYS = frozenset(
    {
        "current_mos",
        "throughput_mbps",
        "bitrate_kbps",
        "capacity_margin_mbps",
        "throughput_to_bitrate_ratio",
        "plr_percent",
    }
)

FIELD_ALIASES = {
    "current_mos": r"\b(?:current\s+)?mos\b",
    "throughput_mbps": (
        r"\bthroughput\b"
        r"(?![ _-]to[ _-]bitrate[ _-]ratio|"
        r"[ _-]bitrate[ _-]ratio|[ _-]ratio)"
    ),
    "bitrate_kbps": r"\bbitrate\b(?![ _-]ratio)",
    "capacity_margin_mbps": r"\bcapacity[ _-]margin\b",
    "throughput_to_bitrate_ratio": (
        r"\bthroughput(?:[ _-]to[ _-]bitrate|[ _-]bitrate)[ _-]ratio\b"
    ),
    "plr_percent": r"\b(?:packet[ _-]loss|plr)\b",
}

FIELD_MENTION_PATTERNS = {
    key: re.compile(pattern, re.IGNORECASE)
    for key, pattern in FIELD_ALIASES.items()
}

FIELD_CLAIM_PATTERNS = {
    key: re.compile(
        rf"{pattern}\s*"
        rf"(?:is|was|=|of|at|measured(?:\s+at)?|reported(?:\s+as)?)?\s*"
        rf"(?P<number>{NUMBER_SOURCE})"
        rf"(?:\s*(?P<unit>{UNIT_SOURCE})(?!\w))?",
        re.IGNORECASE,
    )
    for key, pattern in FIELD_ALIASES.items()
}

EXPECTED_UNITS = {
    "current_mos": frozenset(),
    "throughput_mbps": frozenset({"mbps"}),
    "bitrate_kbps": frozenset({"kbps"}),
    "capacity_margin_mbps": frozenset({"mbps"}),
    "throughput_to_bitrate_ratio": frozenset(),
    "plr_percent": frozenset(
        {"%", "percent", "percentage", "percentage point", "percentage points"}
    ),
}

ACCEPTABLE_CONTRADICTION = re.compile(
    r"\b(?:predicts?|forecasts?|expects?)\s+(?:future\s+)?"
    r"(?:poor|degraded|unacceptable)\s+qoe\b|"
    r"\b(?:poor\s+qoe|qoe\s+degradation)\s+(?:is\s+)?"
    r"(?:predicted|expected|forecast|likely)\b|"
    r"\bqoe\s+(?:is\s+expected\s+to|will)\s+"
    r"(?:degrade|deteriorate)\b|"
    r"\bqoe\s+is\s+(?:poor|degraded|unacceptable)\b",
    re.IGNORECASE,
)

POOR_CONTRADICTION = re.compile(
    r"\b(?:predicts?|forecasts?|expects?)\s+(?:future\s+)?"
    r"(?:acceptable|good)\s+qoe\b|"
    r"\b(?:acceptable|good)\s+qoe\s+(?:is\s+)?"
    r"(?:predicted|expected|forecast)\b|"
    r"\bno\s+(?:qoe\s+)?degradation\s+(?:is\s+)?"
    r"(?:predicted|expected|forecast)\b|"
    r"\bqoe\s+(?:(?:is|will|should)\s+)?(?:remain\s+)?"
    r"(?:acceptable|good)\b",
    re.IGNORECASE,
)

ACTIVE_NEGATED_PREDICTION = re.compile(
    r"\b(?:(?:do|does|did)\s+not|(?:do|does|did)n['’]t|"
    r"cannot|can['’]t)\s+(?:predict|forecast|expect)\s+"
    r"(?:future\s+)?(?:(?:poor|degraded|unacceptable|acceptable|good)\s+qoe|"
    r"qoe\s+degradation)\b",
    re.IGNORECASE,
)


def _as_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None

    try:
        decimal = Decimal(str(value))
    except InvalidOperation:
        return None

    return decimal if decimal.is_finite() else None


def _normalize_unit(unit: str) -> str:
    return " ".join(unit.lower().split())


def _narrative(explanation: Explanation) -> str:
    parts = [
        explanation.summary,
        *(cause.statement for cause in explanation.likely_causes),
        *explanation.recommended_operator_checks,
        explanation.limitations,
    ]
    return " ".join(parts)


class GroundingValidator:
    """Reject narrow, mechanically detectable grounding contradictions."""

    def validate(
        self,
        explanation: Explanation,
        evidence: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []
        text = _narrative(explanation)

        self._validate_required_content(explanation, errors)
        self._validate_decision(explanation, evidence, text, errors)
        self._validate_numbers(evidence, text, errors)
        self._validate_field_claims(evidence, text, errors)

        for index, cause in enumerate(explanation.likely_causes, start=1):
            self._validate_cause(index, cause, evidence, errors)

        return len(errors) == 0, errors

    @staticmethod
    def _validate_required_content(
        explanation: Explanation,
        errors: list[str],
    ) -> None:
        if not explanation.summary.strip():
            errors.append("Missing summary")

        if not explanation.confidence.strip():
            errors.append("Missing confidence")

        if not explanation.prediction.strip():
            errors.append("Missing prediction")

        if (
            not explanation.likely_causes
            and "insufficient evidence"
            not in f"{explanation.summary} {explanation.limitations}".lower()
        ):
            errors.append(
                "Empty likely_causes requires an explicit insufficient-evidence "
                "statement."
            )

    @staticmethod
    def _validate_decision(
        explanation: Explanation,
        evidence: dict[str, Any],
        text: str,
        errors: list[str],
    ) -> None:
        decision = evidence.get("prediction")

        if decision not in {
            "future_poor_qoe",
            "future_acceptable_qoe",
            "abstain",
        }:
            errors.append(f"Invalid or missing evidence prediction: {decision!r}")
            return

        if explanation.prediction != decision:
            errors.append(
                "Explanation prediction contradicts evidence prediction: "
                f"{explanation.prediction!r} != {decision!r}."
            )

        polarity_text = ACTIVE_NEGATED_PREDICTION.sub("", text)

        trust_level = evidence.get("trust_level")

        if decision != "abstain" and trust_level not in {"low", "medium", "high"}:
            errors.append(f"Invalid or missing evidence trust_level: {trust_level!r}")
        elif decision != "abstain" and explanation.confidence != trust_level:
            errors.append(
                "Explanation confidence contradicts evidence trust_level: "
                f"{explanation.confidence!r} != {trust_level!r}."
            )

        if decision == "future_acceptable_qoe" and ACCEPTABLE_CONTRADICTION.search(
            polarity_text
        ):
            errors.append(
                "Acceptable-QoE prediction is described as a degradation forecast."
            )

        if decision == "future_poor_qoe" and POOR_CONTRADICTION.search(
            polarity_text
        ):
            errors.append("Poor-QoE prediction is described as acceptable QoE.")

        if decision != "abstain":
            return

        if explanation.confidence != "low":
            errors.append("Abstention explanation confidence must be low.")

        if explanation.likely_causes:
            errors.append("Abstention explanation must not contain likely causes.")

    @staticmethod
    def _validate_numbers(
        evidence: dict[str, Any],
        text: str,
        errors: list[str],
    ) -> None:
        evidence_numbers = {
            decimal
            for value in evidence.values()
            if (decimal := _as_decimal(value)) is not None
        }
        unsupported: set[Decimal] = set()

        for match in NUMBER_PATTERN.finditer(text):
            number = Decimal(match.group("number"))

            if number not in evidence_numbers:
                unsupported.add(number)

        for number in sorted(unsupported):
            errors.append(f"Unsupported number: {number}")

    @staticmethod
    def _validate_field_claims(
        evidence: dict[str, Any],
        text: str,
        errors: list[str],
    ) -> None:
        for key, pattern in FIELD_CLAIM_PATTERNS.items():
            for match in pattern.finditer(text):
                expected_value = _as_decimal(evidence.get(key))
                claimed_value = Decimal(match.group("number"))

                if expected_value is None:
                    errors.append(f"Claim cites unavailable evidence field: {key}")
                elif claimed_value != expected_value:
                    errors.append(
                        f"Claimed value {claimed_value} is not the value of {key}."
                    )

                unit = match.group("unit")

                if (
                    unit is not None
                    and _normalize_unit(unit) not in EXPECTED_UNITS[key]
                ):
                    errors.append(f"Wrong unit for {key}: {unit}")

    @staticmethod
    def _validate_cause(
        index: int,
        cause: LikelyCause,
        evidence: dict[str, Any],
        errors: list[str],
    ) -> None:
        cited_keys = set(cause.evidence_keys)
        unknown_keys = sorted(cited_keys - evidence.keys())

        for key in unknown_keys:
            errors.append(f"Cause {index} cites unknown evidence key: {key}")

        noncausal_keys = sorted(cited_keys - CAUSAL_EVIDENCE_KEYS)

        for key in noncausal_keys:
            errors.append(f"Cause {index} cites non-causal evidence key: {key}")

        for key, pattern in FIELD_MENTION_PATTERNS.items():
            if pattern.search(cause.statement) and key not in cited_keys:
                errors.append(
                    f"Cause {index} attributes a claim to {key} without citing it."
                )

        cited_values = {
            key: decimal
            for key in cited_keys
            if (decimal := _as_decimal(evidence.get(key))) is not None
        }

        for match in NUMBER_PATTERN.finditer(cause.statement):
            number = Decimal(match.group("number"))

            if number not in cited_values.values():
                errors.append(
                    f"Cause {index} number {number} is not supported by its cited "
                    "evidence keys."
                )

        for match in NUMBER_WITH_UNIT_PATTERN.finditer(cause.statement):
            number = Decimal(match.group("number"))
            unit = _normalize_unit(match.group("unit"))
            matching_keys = {
                key for key, value in cited_values.items() if value == number
            }

            if matching_keys and not any(
                unit in EXPECTED_UNITS.get(key, frozenset()) for key in matching_keys
            ):
                errors.append(
                    f"Cause {index} uses unit {match.group('unit')} for incompatible "
                    "cited evidence."
                )
