from qoe_twin.evidence_builder import build_evidence


def test_build_evidence():

    row = {
        "prediction": 1,
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
        "unused": 999,
    }

    evidence = build_evidence(row)

    assert "unused" not in evidence
    assert evidence["trust_level"] == "high"
    assert evidence["prediction_probability"] == 0.91
