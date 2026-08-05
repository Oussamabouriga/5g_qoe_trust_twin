from qoe_twin.evidence_builder import build_evidence

example = {
    "prediction": "future_poor_qoe",
    "prediction_probability": 0.95,
    "trust_score": 0.92,
    "trust_level": "high",
    "current_mos": 2.4,
    "throughput_mbps": 4.8,
    "bitrate_kbps": 6000,
    "capacity_margin_mbps": -1.2,
    "throughput_to_bitrate_ratio": 0.80,
    "plr_percent": 3.7,
}

print(build_evidence(example))
