"""Chronological replay engine for the 5G QoE digital twin."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


class ProbabilityModel(Protocol):
    """Interface required from a classification model."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""
        ...


@dataclass
class TwinState:
    """Current state of one 5G video-service trajectory."""

    session_id: str
    user_id: int
    timestamp: pd.Timestamp

    base_station: int
    prb: int

    latitude: float
    longitude: float
    throughput_mbps: float

    bitrate_kbps: int
    resolution: str
    plr_percent: float
    current_mos: float

    session_step: int
    elapsed_session_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a serializable dictionary."""
        state = asdict(self)
        state["timestamp"] = self.timestamp.isoformat()
        return state


@dataclass
class TwinPrediction:
    """Prediction generated at one replay step."""

    session_id: str
    user_id: int
    timestamp: pd.Timestamp
    predicted_poor_qoe: int
    probability_poor_qoe: float
    actual_future_poor_qoe: int | None
    prediction_lead_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to a dictionary."""
        return asdict(self)


class TraceDrivenQoETwin:
    """
    Trace-driven digital twin that processes observations chronologically.

    The twin never reads future rows while processing the current state.
    """

    def __init__(
        self,
        model: ProbabilityModel | None = None,
        feature_names: list[str] | None = None,
        decision_threshold: float = 0.5,
        history_size: int = 5,
    ) -> None:
        self.model = model
        self.feature_names = feature_names or []
        self.decision_threshold = decision_threshold
        self._history_size = history_size

        self.current_state: TwinState | None = None
        self.history: deque[TwinState] = deque(
            maxlen=history_size
        )
        self._state_by_session: dict[str, TwinState] = {}
        self._history_by_session: dict[
            str,
            deque[TwinState],
        ] = {}

        self.predictions: list[TwinPrediction] = []
        self.current_session_id: str | None = None

    def reset_session(self) -> None:
        """Clear all session state and the active-session view."""
        self.current_state = None
        self.history.clear()
        self.current_session_id = None
        for session_history in self._history_by_session.values():
            session_history.clear()
        self._state_by_session.clear()
        self._history_by_session.clear()

    def update_state(self, row: pd.Series) -> TwinState:
        """Update the twin using the current chronological observation."""
        session_id = str(row["session_id"])

        state = TwinState(
            session_id=session_id,
            user_id=int(row["user_id"]),
            timestamp=pd.Timestamp(row["timestamp"]),
            base_station=int(row["base_station"]),
            prb=int(row["prb"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            throughput_mbps=float(row["throughput_mbps"]),
            bitrate_kbps=int(row["bitrate_kbps"]),
            resolution=str(row["resolution"]),
            plr_percent=float(row["plr_percent"]),
            current_mos=float(row["current_mos"]),
            session_step=int(row["session_step"]),
            elapsed_session_seconds=float(
                row["elapsed_session_seconds"]
            ),
        )

        session_history = self._history_by_session.setdefault(
            session_id,
            deque(maxlen=self._history_size),
        )
        session_history.append(state)

        self._state_by_session[session_id] = state
        self.current_state = state
        self.current_session_id = session_id
        self.history = session_history

        return state

    def predict_current_row(
        self,
        row: pd.Series,
    ) -> TwinPrediction | None:
        """
        Generate a prediction from features already available at time t.

        The target columns are used only after prediction for evaluation.
        """
        if self.model is None:
            return None

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in row.index
        ]

        if missing_features:
            raise ValueError(
                f"Missing replay features: {missing_features}"
            )

        feature_frame = pd.DataFrame(
            [
                {
                    feature: row[feature]
                    for feature in self.feature_names
                }
            ]
        )

        probability = float(
            self.model.predict_proba(feature_frame)[0, 1]
        )

        predicted_class = int(
            probability >= self.decision_threshold
        )

        actual_target: int | None

        if pd.isna(row.get("future_poor_qoe")):
            actual_target = None
        else:
            actual_target = int(row["future_poor_qoe"])

        lead_time = row.get("prediction_lead_seconds")

        prediction = TwinPrediction(
            session_id=str(row["session_id"]),
            user_id=int(row["user_id"]),
            timestamp=pd.Timestamp(row["timestamp"]),
            predicted_poor_qoe=predicted_class,
            probability_poor_qoe=probability,
            actual_future_poor_qoe=actual_target,
            prediction_lead_seconds=(
                None
                if pd.isna(lead_time)
                else float(lead_time)
            ),
        )

        self.predictions.append(prediction)

        return prediction

    def replay(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Replay a dataset one row at a time in chronological order."""
        self.predictions.clear()
        self.reset_session()

        required_columns = {
            "session_id",
            "user_id",
            "timestamp",
            "base_station",
            "prb",
            "latitude",
            "longitude",
            "throughput_mbps",
            "bitrate_kbps",
            "resolution",
            "plr_percent",
            "current_mos",
            "session_step",
            "elapsed_session_seconds",
        }

        missing = required_columns - set(frame.columns)

        if missing:
            raise ValueError(
                f"Replay dataset is missing columns: {sorted(missing)}"
            )

        ordered = frame.sort_values(
            [
                "timestamp",
                "session_id",
            ],
            kind="stable",
        ).reset_index(drop=True)

        previous_timestamp_by_session: dict[
            str,
            pd.Timestamp,
        ] = {}

        for _, row in ordered.iterrows():
            session_id = str(row["session_id"])
            timestamp = pd.Timestamp(row["timestamp"])

            previous_timestamp = previous_timestamp_by_session.get(
                session_id
            )

            if (
                previous_timestamp is not None
                and timestamp < previous_timestamp
            ):
                raise ValueError(
                    "Chronological replay order was violated."
                )

            self.update_state(row)
            self.predict_current_row(row)

            previous_timestamp_by_session[
                session_id
            ] = timestamp

        return pd.DataFrame(
            prediction.to_dict()
            for prediction in self.predictions
        )
