"""Tests for the trace-driven digital twin."""

import pandas as pd

from qoe_twin.baselines import (
    CurrentQoEPersistenceBaseline,
)
from qoe_twin.replay import TraceDrivenQoETwin


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
            "prediction_lead_seconds": [8.0, 8.0, float("nan")],
        }
    )


def test_replay_processes_all_rows() -> None:
    model = CurrentQoEPersistenceBaseline()

    twin = TraceDrivenQoETwin(
        model=model,
        feature_names=["current_mos"],
    )

    predictions = twin.replay(
        create_replay_frame()
    )

    assert len(predictions) == 3


def test_persistence_baseline_predictions() -> None:
    model = CurrentQoEPersistenceBaseline()

    twin = TraceDrivenQoETwin(
        model=model,
        feature_names=["current_mos"],
    )

    predictions = twin.replay(
        create_replay_frame()
    )

    assert predictions.loc[
        0,
        "predicted_poor_qoe",
    ] == 1

    assert predictions.loc[
        2,
        "predicted_poor_qoe",
    ] == 0


def test_replay_resets_for_new_session() -> None:
    frame = create_replay_frame()

    second_session = frame.iloc[[0]].copy()
    second_session["session_id"] = "s2"
    second_session["user_id"] = 2

    combined = pd.concat(
        [frame, second_session],
        ignore_index=True,
    )

    twin = TraceDrivenQoETwin()

    twin.replay(combined)

    assert twin.current_session_id == "s2"
    assert len(twin.history) == 1
