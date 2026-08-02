"""Trust scoring and abstention rules for QoE predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrustResult:
    """Trust information associated with one prediction."""

    trust_score: float
    trust_level: str
    abstain: bool

    model_confidence: float
    validation_quality: float
    data_quality: float
    prediction_stability: float

    def to_dict(self) -> dict[str, float | str | bool]:
        """Return a serializable trust record."""
        return asdict(self)


def probability_confidence(
    probability: float,
) -> float:
    """
    Convert probability distance from 0.5 into confidence.

    A probability of 0.5 gives confidence 0.
    Probabilities of 0 or 1 give confidence 1.
    """
    confidence = 2.0 * abs(float(probability) - 0.5)

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )


def calculate_data_quality(
    row: pd.Series,
    required_features: list[str],
) -> float:
    """Calculate the fraction of required features that are usable."""
    if not required_features:
        return 1.0

    valid_count = 0

    for feature in required_features:
        if feature not in row.index:
            continue

        value = row[feature]

        if pd.notna(value):
            valid_count += 1

    return float(
        valid_count / len(required_features)
    )


def calculate_prediction_stability(
    probability: float,
    previous_probabilities: list[float],
    window: int = 5,
) -> float:
    """
    Estimate prediction stability from recent probabilities.

    Stable recent probabilities receive a higher score.
    """
    recent = previous_probabilities[-window:]

    if not recent:
        return 1.0

    values = np.asarray(
        [*recent, float(probability)],
        dtype=float,
    )

    variation = float(
        np.std(values)
    )

    stability = 1.0 - min(
        variation / 0.5,
        1.0,
    )

    return float(
        np.clip(
            stability,
            0.0,
            1.0,
        )
    )


def calculate_trust(
    probability: float,
    validation_f1: float,
    validation_ece: float,
    data_quality: float,
    prediction_stability: float,
    confidence_weight: float = 0.55,
    validation_weight: float = 0.25,
    data_quality_weight: float = 0.10,
    stability_weight: float = 0.10,
    abstention_threshold: float = 0.55,
) -> TrustResult:
    """Combine reliability indicators into one trust score."""
    confidence = probability_confidence(
        probability
    )

    calibration_quality = 1.0 - np.clip(
        validation_ece,
        0.0,
        1.0,
    )

    validation_quality = (
        0.70 * np.clip(
            validation_f1,
            0.0,
            1.0,
        )
        + 0.30 * calibration_quality
    )

    trust_score = (
        confidence_weight * confidence
        + validation_weight * validation_quality
        + data_quality_weight * np.clip(
            data_quality,
            0.0,
            1.0,
        )
        + stability_weight * np.clip(
            prediction_stability,
            0.0,
            1.0,
        )
    )

    trust_score = float(
        np.clip(
            trust_score,
            0.0,
            1.0,
        )
    )

    if trust_score >= 0.80:
        trust_level = "high"
    elif trust_score >= abstention_threshold:
        trust_level = "medium"
    else:
        trust_level = "low"

    return TrustResult(
        trust_score=trust_score,
        trust_level=trust_level,
        abstain=trust_score < abstention_threshold,
        model_confidence=confidence,
        validation_quality=float(validation_quality),
        data_quality=float(data_quality),
        prediction_stability=float(
            prediction_stability
        ),
    )
