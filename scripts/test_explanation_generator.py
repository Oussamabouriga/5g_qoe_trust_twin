from dotenv import load_dotenv

from qoe_twin.explanation_generator import (
    ExplanationGenerator,
)

load_dotenv()

row = {
    "prediction": "future_poor_qoe",
    "prediction_probability": 0.94,
    "trust_score": 0.93,
    "trust_level": "high",
    "current_mos": 2.1,
    "throughput_mbps": 4.2,
    "bitrate_kbps": 6000,
    "capacity_margin_mbps": -1.8,
    "throughput_to_bitrate_ratio": 0.70,
    "plr_percent": 4.5,
}

generator = ExplanationGenerator()

result = generator.explain(row)

print(result)
