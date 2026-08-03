"""Semantic tests for the trace-driven digital twin."""

import pandas as pd

from qoe_twin.baselines import (
    CurrentQoEPersistenceBaseline,
)
from qoe_twin.replay import TraceDrivenQoETwin


class HistoryRecordingTwin(TraceDrivenQoETwin):
    """Twin test double that records history visible at prediction time."""

    def __init__(self) -> None:
        super().__init__()
        self.history_snapshots: list[tuple[str, ...]] = []

    def predict_current_row(self, row: pd.Series) -> None:
        self.history_snapshots.append(
            tuple(state.session_id for state in self.history)
        )
        super().predict_current_row(row)


def create_replay_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1"],
            "user_id": [1, 1, 1],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:08Z",
                    "2026-01-01T00:00:16Z",
                ]
            ),
            "base_station": [1, 1, 1],
            "prb": [10, 10, 10],
            "latitude": [48.0, 48.0001, 48.0002],
            "longitude": [2.0, 2.0001, 2.0002],
            "throughput_mbps": [2.0, 3.0, 4.0],
            "bitrate_kbps": [4000, 4000, 4000],
            "resolution": ["1080p", "1080p", "1080p"],
            "plr_percent": [20.0, 10.0, 0.0],
            "current_mos": [2.0, 2.5, 4.0],
            "session_step": [0, 1, 2],
            "elapsed_session_seconds": [0.0, 8.0, 16.0],
            "future_poor_qoe": [1, 0, pd.NA],
            "prediction_lead_seconds": [
                8.0,
                8.0,
                float("nan"),
            ],
        }
    )


def create_second_session() -> pd.DataFrame:
    frame = create_replay_frame().copy()
    frame["session_id"] = "s2"
    frame["user_id"] = 2
    return frame


def test_replay_processes_all_rows() -> None:
    model = CurrentQoEPersistenceBaseline()
    twin = TraceDrivenQoETwin(
        model=model,
        feature_names=["current_mos"],
    )

    predictions = twin.replay(create_replay_frame())

    assert len(predictions) == 3


def test_persistence_baseline_predictions() -> None:
    model = CurrentQoEPersistenceBaseline()
    twin = TraceDrivenQoETwin(
        model=model,
        feature_names=["current_mos"],
    )

    predictions = twin.replay(create_replay_frame())

    assert predictions.loc[0, "predicted_poor_qoe"] == 1
    assert predictions.loc[2, "predicted_poor_qoe"] == 0


def test_replay_clears_stale_state_before_processing() -> None:
    twin = HistoryRecordingTwin()
    stale_row = create_replay_frame().iloc[0].copy()
    stale_row["session_id"] = "stale"
    twin.update_state(stale_row)

    twin.replay(create_replay_frame().iloc[:2])

    assert twin.history_snapshots == [
        ("s1",),
        ("s1", "s1"),
    ]
    assert twin.current_session_id == "s1"
    assert all(
        "stale" not in snapshot
        for snapshot in twin.history_snapshots
    )


def test_update_state_isolates_history_when_session_changes() -> None:
    twin = TraceDrivenQoETwin()
    first_session = create_replay_frame().iloc[:2]
    second_session_row = create_second_session().iloc[0]

    twin.update_state(first_session.iloc[0])
    twin.update_state(first_session.iloc[1])
    assert len(twin.history) == 2

    twin.update_state(second_session_row)

    assert twin.current_session_id == "s2"
    assert {
        state.session_id for state in twin.history
    } == {"s2"}
    assert len(twin.history) == 1


def test_replay_order_is_deterministic_and_chronological_per_session() -> None:
    combined = pd.concat(
        [create_replay_frame(), create_second_session()],
        ignore_index=True,
    )
    first_order = combined.iloc[
        [4, 2, 5, 0, 3, 1]
    ].reset_index(drop=True)
    second_order = combined.iloc[
        [1, 5, 0, 3, 2, 4]
    ].reset_index(drop=True)
    twin = TraceDrivenQoETwin(
        model=CurrentQoEPersistenceBaseline(),
        feature_names=["current_mos"],
    )

    first_predictions = twin.replay(first_order)
    second_predictions = twin.replay(second_order)

    pd.testing.assert_frame_equal(
        first_predictions,
        second_predictions,
    )

    for _, session_predictions in first_predictions.groupby(
        "session_id"
    ):
        assert session_predictions[
            "timestamp"
        ].is_monotonic_increasing
