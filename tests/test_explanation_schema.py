import pytest
from pydantic import ValidationError

from qoe_twin.explanation_schema import Explanation


def test_schema() -> None:
    explanation = Explanation(
        prediction="Poor QoE predicted",
        confidence="high",
        summary="Example explanation.",
        likely_causes=[
            "Throughput is below the requested bitrate."
        ],
        recommended_operator_checks=[
            "Inspect throughput and bitrate allocation."
        ],
        limitations=(
            "Only based on the supplied evidence."
        ),
    )

    assert explanation.prediction.startswith(
        "Poor"
    )


def test_schema_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        Explanation(
            prediction="Poor QoE predicted",
            confidence="very high",
            summary="Example explanation.",
            likely_causes=["Cause"],
            recommended_operator_checks=["Check"],
            limitations="Evidence only.",
        )


def test_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Explanation(
            prediction="Poor QoE predicted",
            confidence="high",
            summary="Example explanation.",
            likely_causes=["Cause"],
            recommended_operator_checks=["Check"],
            limitations="Evidence only.",
            invented_field="not allowed",
        )
