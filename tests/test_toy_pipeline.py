"""Artifact-independent smoke test for the replay-to-evaluation pipeline.

Later phases intentionally own the following extensions:
- threshold-aware confidence around the learned decision threshold: CP3;
- semantic LLM field, unit, and contradiction validation: CP6;
- global interleaved-session replay and batch/replay parity: CP9.
"""

import numpy as np
import pandas as pd
import pytest

from qoe_twin.evaluation import evaluate_binary_predictions
from qoe_twin.replay import TraceDrivenQoETwin
from qoe_twin.trust import calculate_trust


class DeterministicProbabilityModel:
    """Return probabilities supplied by the synthetic replay rows."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = features["toy_probability"].to_numpy(dtype=float)
        return np.column_stack((1.0 - probability, probability))


def create_toy_replay_rows() -> pd.DataFrame:
    """Create a complete replay frame with one unevaluable session end."""
    return pd.DataFrame(
        {
            "session_id": ["toy-session"] * 5,
            "user_id": [1] * 5,
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:08Z",
                    "2026-01-01T00:00:16Z",
                    "2026-01-01T00:00:24Z",
                    "2026-01-01T00:00:32Z",
                ]
            ),
            "base_station": [1] * 5,
            "prb": [10] * 5,
            "latitude": [48.0] * 5,
            "longitude": [2.0] * 5,
            "throughput_mbps": [8.0, 6.0, 3.0, 2.0, 7.0],
            "bitrate_kbps": [4000] * 5,
            "resolution": ["1080p"] * 5,
            "plr_percent": [0.0, 2.0, 10.0, 15.0, 1.0],
            "current_mos": [4.0, 3.5, 2.5, 2.0, 4.0],
            "session_step": list(range(5)),
            "elapsed_session_seconds": [0.0, 8.0, 16.0, 24.0, 32.0],
            "future_poor_qoe": [0, 0, 1, 1, pd.NA],
            "prediction_lead_seconds": [8.0, 8.0, 8.0, 8.0, pd.NA],
            "toy_probability": [0.1, 0.7, 0.8, 0.4, 0.9],
        }
    )


def test_toy_replay_trust_and_evaluation_pipeline() -> None:
    decision_threshold = 0.5
    abstention_threshold = 0.55
    twin = TraceDrivenQoETwin(
        model=DeterministicProbabilityModel(),
        feature_names=["toy_probability"],
        decision_threshold=decision_threshold,
    )

    predictions = twin.replay(create_toy_replay_rows())

    trust_results = [
        calculate_trust(
            probability=probability,
            validation_f1=0.8,
            validation_ece=0.1,
            data_quality=1.0,
            prediction_stability=0.75,
            confidence_weight=0.25,
            validation_weight=0.25,
            data_quality_weight=0.25,
            stability_weight=0.25,
            abstention_threshold=abstention_threshold,
        )
        for probability in predictions["probability_poor_qoe"]
    ]
    predictions["trust_score"] = [
        result.trust_score for result in trust_results
    ]
    predictions["abstain"] = [result.abstain for result in trust_results]

    evaluable = predictions.loc[
        predictions["actual_future_poor_qoe"].notna()
    ]
    metrics = evaluate_binary_predictions(
        target=evaluable["actual_future_poor_qoe"],
        prediction=evaluable["predicted_poor_qoe"],
        probability=evaluable["probability_poor_qoe"],
    )

    assert predictions["probability_poor_qoe"].tolist() == [
        0.1,
        0.7,
        0.8,
        0.4,
        0.9,
    ]
    assert predictions["trust_score"].tolist() == pytest.approx(
        [0.845, 0.745, 0.795, 0.695, 0.845]
    )
    assert predictions["abstain"].tolist() == [False] * 5
    assert len(predictions) == 5
    assert len(evaluable) == 4
    assert metrics["observations"] == 4
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
