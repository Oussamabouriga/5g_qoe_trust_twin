"""Probability calibration and reliability metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    """Calibrate probabilities using sigmoid or isotonic regression."""

    method: str
    model: object | None = None
    random_seed: int = 42

    def fit(
        self,
        raw_probability: np.ndarray,
        target: np.ndarray,
    ) -> ProbabilityCalibrator:
        """Fit the selected calibration method."""
        probability = np.asarray(
            raw_probability,
            dtype=float,
        ).reshape(-1)

        target = np.asarray(
            target,
            dtype=int,
        ).reshape(-1)

        if self.method == "sigmoid":
            model = LogisticRegression(
                solver="lbfgs",
                random_state=self.random_seed,
            )

            model.fit(
                probability.reshape(-1, 1),
                target,
            )

            self.model = model

        elif self.method == "isotonic":
            model = IsotonicRegression(
                out_of_bounds="clip",
            )

            model.fit(
                probability,
                target,
            )

            self.model = model

        else:
            raise ValueError(
                "Calibration method must be "
                "'sigmoid' or 'isotonic'."
            )

        return self

    def predict(
        self,
        raw_probability: np.ndarray,
    ) -> np.ndarray:
        """Return calibrated probabilities."""
        if self.model is None:
            raise RuntimeError(
                "The calibrator must be fitted first."
            )

        probability = np.asarray(
            raw_probability,
            dtype=float,
        ).reshape(-1)

        if self.method == "sigmoid":
            calibrated = self.model.predict_proba(
                probability.reshape(-1, 1)
            )[:, 1]

        else:
            calibrated = self.model.predict(
                probability
            )

        return np.clip(
            calibrated,
            0.0,
            1.0,
        )


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    number_of_bins: int = 10,
) -> float:
    """Calculate expected calibration error."""
    target = np.asarray(
        target,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    bin_ids = np.digitize(
        probability,
        bin_edges[1:-1],
        right=True,
    )

    error = 0.0
    total = len(target)

    for bin_id in range(number_of_bins):
        mask = bin_ids == bin_id
        bin_count = int(mask.sum())

        if bin_count == 0:
            continue

        average_confidence = probability[
            mask
        ].mean()

        observed_frequency = target[
            mask
        ].mean()

        error += (
            bin_count / total
        ) * abs(
            observed_frequency
            - average_confidence
        )

    return float(error)


def calibration_table(
    target: np.ndarray,
    probability: np.ndarray,
    number_of_bins: int = 10,
) -> pd.DataFrame:
    """Create data for a reliability diagram."""
    target = np.asarray(
        target,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    bin_ids = np.digitize(
        probability,
        bin_edges[1:-1],
        right=True,
    )

    rows: list[dict[str, float | int]] = []

    for bin_id in range(number_of_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())

        if count == 0:
            continue

        rows.append(
            {
                "bin": bin_id + 1,
                "lower_bound": float(
                    bin_edges[bin_id]
                ),
                "upper_bound": float(
                    bin_edges[bin_id + 1]
                ),
                "count": count,
                "mean_probability": float(
                    probability[mask].mean()
                ),
                "observed_frequency": float(
                    target[mask].mean()
                ),
            }
        )

    return pd.DataFrame(rows)
