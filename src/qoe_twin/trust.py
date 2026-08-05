"""Trust scoring and abstention rules for QoE predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrustResult:
    """Historical prototype trust result retained for CP1 compatibility."""

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


@dataclass(frozen=True)
class TrustPolicy:
    """YAML-backed policy for the final heuristic reliability score."""

    confidence_weight: float
    validation_performance_weight: float
    calibration_weight: float
    data_quality_weight: float
    high_level: float
    medium_level: float
    abstention_enabled: bool
    minimum_coverage: float

    @classmethod
    def from_mapping(cls, configuration: Mapping[str, Any]) -> TrustPolicy:
        """Build a policy from one strictly validated ``trust.yaml`` value."""
        weights = _required_mapping(configuration, "trust")
        levels = _required_mapping(configuration, "levels")
        abstention = _required_mapping(configuration, "abstention")

        policy = cls(
            confidence_weight=_unit_interval_value(
                weights,
                "confidence_weight",
            ),
            validation_performance_weight=_unit_interval_value(
                weights,
                "validation_performance_weight",
            ),
            calibration_weight=_unit_interval_value(
                weights,
                "calibration_weight",
            ),
            data_quality_weight=_unit_interval_value(
                weights,
                "data_quality_weight",
            ),
            high_level=_unit_interval_value(levels, "high"),
            medium_level=_unit_interval_value(levels, "medium"),
            abstention_enabled=_boolean_value(abstention, "enabled"),
            minimum_coverage=_unit_interval_value(
                abstention,
                "minimum_coverage",
            ),
        )

        weight_sum = (
            policy.confidence_weight
            + policy.validation_performance_weight
            + policy.calibration_weight
            + policy.data_quality_weight
        )
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Trust policy weights must sum to 1.0.")
        if policy.high_level <= policy.medium_level:
            raise ValueError("High trust level must exceed medium trust level.")
        if policy.minimum_coverage < 0.90:
            raise ValueError("Minimum abstention coverage must be at least 0.90.")

        return policy


@dataclass(frozen=True)
class ConfiguredTrustResult:
    """Final heuristic reliability result; not correctness probability."""

    trust_score: float
    trust_level: str
    abstain: bool
    model_confidence: float
    validation_performance: float
    calibration_quality: float
    data_quality: float
    prediction_stability: float

    def to_dict(self) -> dict[str, float | str | bool]:
        """Return a serializable reliability record."""
        return asdict(self)


def _required_mapping(
    configuration: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = configuration.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Trust policy {key!r} must be a mapping.")
    return value


def _unit_interval_value(
    configuration: Mapping[str, Any],
    key: str,
) -> float:
    value = configuration.get(key)
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"Trust policy {key!r} must be a finite number.")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"Trust policy {key!r} must be between 0 and 1.")
    return result


def _boolean_value(
    configuration: Mapping[str, Any],
    key: str,
) -> bool:
    value = configuration.get(key)
    if type(value) is not bool:
        raise ValueError(f"Trust policy {key!r} must be a boolean.")
    return value


def probability_confidence(
    probability: float,
    decision_threshold: float,
) -> float:
    """
    Normalize distance from the selected model decision threshold.

    The selected threshold has confidence zero. Each probability endpoint has
    confidence one on its corresponding decision side.
    """
    threshold = float(decision_threshold)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("decision_threshold must be strictly between 0 and 1.")

    raw_probability = float(probability)
    if not math.isfinite(raw_probability):
        raise ValueError("probability must be finite.")
    clipped_probability = float(np.clip(raw_probability, 0.0, 1.0))
    if clipped_probability < threshold:
        confidence = (
            threshold - clipped_probability
        ) / threshold
    else:
        confidence = (
            clipped_probability - threshold
        ) / (1.0 - threshold)

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )


def calculate_configured_trust(
    *,
    probability: float,
    decision_threshold: float,
    validation_f1: float,
    validation_ece: float,
    data_quality: float,
    prediction_stability: float,
    policy: TrustPolicy,
    abstention_threshold: float,
) -> ConfiguredTrustResult:
    """Calculate the YAML-defined heuristic reliability score.

    Prediction stability is retained as a diagnostic field but is deliberately
    excluded from the score because it has no weight in ``trust.yaml``.
    """
    threshold = _unit_interval_value(
        {"abstention_threshold": abstention_threshold},
        "abstention_threshold",
    )
    confidence = probability_confidence(
        probability,
        decision_threshold,
    )
    validation_performance = _finite_clipped_unit_value(
        validation_f1,
        "validation_f1",
    )
    calibration_quality = float(
        1.0
        - _finite_clipped_unit_value(
            validation_ece,
            "validation_ece",
        )
    )
    usable_data = _finite_clipped_unit_value(
        data_quality,
        "data_quality",
    )
    stability = _finite_clipped_unit_value(
        prediction_stability,
        "prediction_stability",
    )

    trust_score = float(
        np.clip(
            policy.confidence_weight * confidence
            + policy.validation_performance_weight
            * validation_performance
            + policy.calibration_weight * calibration_quality
            + policy.data_quality_weight * usable_data,
            0.0,
            1.0,
        )
    )

    if trust_score >= policy.high_level:
        trust_level = "high"
    elif trust_score >= policy.medium_level:
        trust_level = "medium"
    else:
        trust_level = "low"

    return ConfiguredTrustResult(
        trust_score=trust_score,
        trust_level=trust_level,
        abstain=(
            policy.abstention_enabled
            and trust_score < threshold
        ),
        model_confidence=confidence,
        validation_performance=validation_performance,
        calibration_quality=calibration_quality,
        data_quality=usable_data,
        prediction_stability=stability,
    )


def require_selected_abstention_threshold(
    selected_configuration: Mapping[str, Any],
) -> float:
    """Require the validation-selected CP8 threshold without a fallback."""
    if "abstention_threshold" not in selected_configuration:
        raise ValueError(
            "The CP8 validation-selected abstention_threshold is missing; "
            "final trust inference is blocked."
        )
    return _unit_interval_value(
        selected_configuration,
        "abstention_threshold",
    )


def _finite_clipped_unit_value(value: float, label: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{label} must be finite.")
    return float(np.clip(numeric_value, 0.0, 1.0))


def risk_coverage_table(
    target: Sequence[int] | pd.Series | np.ndarray,
    prediction: Sequence[int] | pd.Series | np.ndarray,
    trust_score: Sequence[float] | pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Build a deterministic selective-risk versus coverage table."""
    target_values = np.asarray(target)
    prediction_values = np.asarray(prediction)
    trust_values = np.asarray(trust_score, dtype=float)

    arrays = {
        "target": target_values,
        "prediction": prediction_values,
        "trust_score": trust_values,
    }
    for label, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional.")
    lengths = {len(values) for values in arrays.values()}
    if lengths != {len(target_values)} or not target_values.size:
        raise ValueError("Risk/coverage inputs must be nonempty and equal-length.")
    if not np.isfinite(trust_values).all():
        raise ValueError("trust_score must contain only finite values.")
    if ((trust_values < 0.0) | (trust_values > 1.0)).any():
        raise ValueError("trust_score must be between 0 and 1.")
    if not set(np.unique(target_values)).issubset({0, 1}):
        raise ValueError("target must contain only binary values.")
    if not set(np.unique(prediction_values)).issubset({0, 1}):
        raise ValueError("prediction must contain only binary values.")

    errors = prediction_values != target_values
    total = len(target_values)
    rows: list[dict[str, float | int]] = []

    descending_order = np.argsort(-trust_values, kind="stable")
    descending_scores = trust_values[descending_order]
    descending_errors = errors[descending_order].astype(int)
    cumulative_errors = np.cumsum(descending_errors)
    tied_score_ends = np.flatnonzero(
        np.r_[
            descending_scores[:-1] != descending_scores[1:],
            True,
        ]
    )

    for tied_score_end in tied_score_ends[::-1]:
        accepted_count = int(tied_score_end + 1)
        threshold = float(descending_scores[tied_score_end])
        rows.append(
            {
                "abstention_threshold": threshold,
                "observations": total,
                "accepted_predictions": accepted_count,
                "abstained_predictions": total - accepted_count,
                "coverage": float(accepted_count / total),
                "selective_risk": float(
                    cumulative_errors[tied_score_end] / accepted_count
                ),
            }
        )

    return pd.DataFrame(rows)


def select_validation_abstention_threshold(
    validation_frame: pd.DataFrame,
    *,
    target_column: str,
    prediction_column: str,
    minimum_coverage: float,
    trust_score_column: str = "trust_score",
    split_column: str = "split",
) -> tuple[float, pd.DataFrame]:
    """Minimize validation selective risk subject to at least 90% coverage."""
    if not 0.90 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0.90 and 1.0.")

    required_columns = {
        target_column,
        prediction_column,
        trust_score_column,
        split_column,
    }
    missing = sorted(required_columns - set(validation_frame.columns))
    if missing:
        raise ValueError(
            "Validation abstention selection is missing columns: "
            + ", ".join(missing)
        )
    if validation_frame.empty:
        raise ValueError("Validation abstention selection requires observations.")
    validation_rows = validation_frame[split_column].eq("validation")
    if not validation_rows.fillna(False).all():
        raise ValueError(
            "Abstention threshold selection accepts validation rows only."
        )

    table = risk_coverage_table(
        target=validation_frame[target_column],
        prediction=validation_frame[prediction_column],
        trust_score=validation_frame[trust_score_column],
    )
    eligible = table.loc[
        table["coverage"] >= minimum_coverage
    ].copy()
    if eligible.empty:
        raise ValueError(
            "No abstention threshold satisfies the minimum coverage."
        )

    best = eligible.sort_values(
        [
            "selective_risk",
            "coverage",
            "abstention_threshold",
        ],
        ascending=[True, False, False],
        kind="stable",
    ).iloc[0]
    return float(best["abstention_threshold"]), table


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
    """Calculate the historical prototype score retained for CP1 only."""
    confidence = probability_confidence(
        probability,
        decision_threshold=0.5,
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
