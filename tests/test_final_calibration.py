"""Synthetic calibration and abstention-integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qoe_twin.trust import (
    TrustPolicy,
    calculate_configured_reliability_score,
    calculate_configured_trust,
    select_validation_abstention_threshold,
)
from scripts.calibrate_models import build_validation_trust_scores


def _confidence_only_policy() -> TrustPolicy:
    return TrustPolicy.from_mapping(
        {
            "trust": {
                "confidence_weight": 1.0,
                "validation_performance_weight": 0.0,
                "calibration_weight": 0.0,
                "data_quality_weight": 0.0,
            },
            "levels": {"high": 0.8, "medium": 0.55},
            "abstention": {"enabled": True, "minimum_coverage": 0.9},
        }
    )


def test_score_only_calculation_matches_final_trust_score() -> None:
    inputs = {
        "probability": 0.8,
        "decision_threshold": 0.5,
        "validation_f1": 0.75,
        "validation_ece": 0.1,
        "data_quality": 1.0,
        "prediction_stability": 0.6,
        "policy": _confidence_only_policy(),
    }

    score = calculate_configured_reliability_score(**inputs)
    final = calculate_configured_trust(
        **inputs,
        abstention_threshold=0.7,
    )

    assert score == pytest.approx(0.6)
    assert score == final.trust_score


def test_validation_trust_selects_minimum_risk_at_ninety_percent_coverage() -> None:
    timestamps = pd.date_range("2026-01-01", periods=10, freq="s", tz="UTC")
    probabilities = np.array(
        [0.45, 0.40, 0.35, 0.30, 0.25, 0.80, 0.85, 0.90, 0.95, 1.00]
    )
    targets = [1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ordered = pd.DataFrame(
        {
            "session_id": ["session"] * 10,
            "timestamp": timestamps,
            "future_poor_qoe": targets,
            "feature": [1.0] * 10,
        }
    )
    reverse_order = np.arange(9, -1, -1)

    trust = build_validation_trust_scores(
        ordered.iloc[reverse_order].reset_index(drop=True),
        probabilities[reverse_order],
        target_column="future_poor_qoe",
        feature_names=["feature"],
        decision_threshold=0.5,
        validation_f1=0.8,
        validation_ece=0.1,
        policy=_confidence_only_policy(),
    )
    threshold, table = select_validation_abstention_threshold(
        trust,
        target_column="target",
        prediction_column="prediction",
        minimum_coverage=0.9,
    )

    assert trust["timestamp"].tolist() == timestamps.tolist()
    assert trust["trust_score"].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    assert threshold == pytest.approx(0.2)
    selected = table.loc[table["abstention_threshold"].eq(threshold)].iloc[0]
    assert selected["coverage"] == pytest.approx(0.9)
    assert selected["selective_risk"] == pytest.approx(0.0)


def test_validation_trust_rejects_misaligned_probabilities() -> None:
    selection = pd.DataFrame(
        {
            "session_id": ["session"],
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
            "future_poor_qoe": [0],
            "feature": [1.0],
        }
    )

    with pytest.raises(ValueError, match="align one-to-one"):
        build_validation_trust_scores(
            selection,
            np.array([]),
            target_column="future_poor_qoe",
            feature_names=["feature"],
            decision_threshold=0.5,
            validation_f1=0.8,
            validation_ece=0.1,
            policy=_confidence_only_policy(),
        )
