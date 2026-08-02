"""Baseline prediction models for the QoE project."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CurrentQoEPersistenceBaseline:
    """
    Predict future poor QoE using the current QoE state.

    This is the persistence baseline required by the assignment.
    """

    def __init__(
        self,
        poor_mos_threshold: float = 3.0,
    ) -> None:
        self.poor_mos_threshold = poor_mos_threshold

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Return deterministic persistence probabilities."""
        if "current_mos" not in features.columns:
            raise ValueError(
                "Persistence baseline requires current_mos."
            )

        poor_probability = (
            features["current_mos"]
            < self.poor_mos_threshold
        ).astype(float)

        good_probability = 1.0 - poor_probability

        return np.column_stack(
            [
                good_probability.to_numpy(),
                poor_probability.to_numpy(),
            ]
        )
