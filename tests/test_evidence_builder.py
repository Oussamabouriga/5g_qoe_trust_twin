"""Tests for the operational LLM evidence allowlist."""

from qoe_twin.evidence_builder import build_evidence


def test_build_evidence_excludes_future_derived_and_unknown_fields() -> None:
    row = {
        "prediction": "future_poor_qoe",
        "prediction_probability": 0.91,
        "trust_score": 0.88,
        "trust_level": "high",
        "current_mos": 2.1,
        "throughput_mbps": 5.3,
        "bitrate_kbps": 6000,
        "capacity_margin_mbps": -0.7,
        "throughput_to_bitrate_ratio": 0.88,
        "plr_percent": 2.4,
        "lead_time_seconds": 10.0,
        "prediction_lead_seconds": 10.0,
        "actual_future_poor_qoe": 1,
        "unused": 999,
    }

    evidence = build_evidence(row)

    assert evidence["prediction"] == "future_poor_qoe"
    assert evidence["prediction_probability"] == 0.91
    assert evidence["trust_level"] == "high"
    assert "lead_time_seconds" not in evidence
    assert "prediction_lead_seconds" not in evidence
    assert "actual_future_poor_qoe" not in evidence
    assert "unused" not in evidence
