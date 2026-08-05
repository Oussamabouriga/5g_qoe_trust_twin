from qoe_twin.evidence_builder import build_evidence
from qoe_twin.prompt_builder import build_prompt

row = {
    "prediction": "future_poor_qoe",
    "prediction_probability": 0.93,
    "trust_score": 0.91,
    "trust_level": "high",
    "current_mos": 2.2,
    "throughput_mbps": 4.7,
    "bitrate_kbps": 6000,
    "capacity_margin_mbps": -1.3,
    "throughput_to_bitrate_ratio": 0.78,
    "plr_percent": 3.9,
}

evidence = build_evidence(row)

prompt = build_prompt(evidence)

print(prompt)
