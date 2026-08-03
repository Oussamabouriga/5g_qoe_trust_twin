"""Synthetic tests for session and future-target contracts."""

import pandas as pd

from qoe_twin.target import add_future_target, add_sessions


def _trajectory_frame(
    timestamps: list[str],
    mos_values: list[float],
) -> pd.DataFrame:
    row_count = len(timestamps)

    return pd.DataFrame(
        {
            "base_station": [1] * row_count,
            "prb": [10] * row_count,
            "bitrate_kbps": [4000] * row_count,
            "user_id": [7] * row_count,
            "timestamp": pd.to_datetime(
                timestamps,
                utc=True,
            ),
            "current_mos": mos_values,
        }
    )


def test_add_sessions_orders_trajectory_and_splits_after_gap() -> None:
    frame = _trajectory_frame(
        timestamps=[
            "2026-01-01T00:01:01Z",
            "2026-01-01T00:00:30Z",
            "2026-01-01T00:00:00Z",
        ],
        mos_values=[2.0, 3.0, 4.0],
    )

    result = add_sessions(
        frame,
        gap_threshold_seconds=30.0,
    )

    assert result["timestamp"].tolist() == list(
        pd.to_datetime(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:30Z",
                "2026-01-01T00:01:01Z",
            ],
            utc=True,
        )
    )
    assert result["gap_seconds"].iloc[1:].tolist() == [
        30.0,
        31.0,
    ]
    assert result["is_session_start"].tolist() == [
        True,
        False,
        True,
    ]
    assert result["session_number"].tolist() == [1, 1, 2]
    assert result["session_id"].tolist() == [
        "1_10_4000_7_1",
        "1_10_4000_7_1",
        "1_10_4000_7_2",
    ]


def test_one_step_target_uses_the_next_observed_mos() -> None:
    frame = _trajectory_frame(
        timestamps=[
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:08Z",
            "2026-01-01T00:00:20Z",
        ],
        mos_values=[2.0, 4.0, 2.5],
    )
    frame["session_id"] = "session-1"

    result = add_future_target(
        frame,
        poor_mos_threshold=3.0,
        horizon_steps=1,
    )

    pd.testing.assert_series_equal(
        result["future_mos"],
        pd.Series(
            [4.0, 2.5, float("nan")],
            name="future_mos",
        ),
    )
    pd.testing.assert_series_equal(
        result["future_poor_qoe"],
        pd.Series(
            [0, 1, pd.NA],
            dtype="Int8",
            name="future_poor_qoe",
        ),
    )
    assert result["prediction_lead_seconds"].tolist()[:2] == [
        8.0,
        12.0,
    ]


def test_poor_qoe_uses_strict_mos_threshold_boundary() -> None:
    frame = _trajectory_frame(
        timestamps=[
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:00:20Z",
            "2026-01-01T00:00:30Z",
        ],
        mos_values=[4.0, 2.999, 3.000, 3.001],
    )
    frame["session_id"] = "session-1"

    result = add_future_target(
        frame,
        poor_mos_threshold=3.0,
    )

    pd.testing.assert_series_equal(
        result["future_mos"],
        pd.Series(
            [2.999, 3.000, 3.001, float("nan")],
            name="future_mos",
        ),
    )
    pd.testing.assert_series_equal(
        result["future_poor_qoe"],
        pd.Series(
            [1, 0, 0, pd.NA],
            dtype="Int8",
            name="future_poor_qoe",
        ),
    )


def test_session_end_has_no_target_and_does_not_cross_boundary() -> None:
    frame = _trajectory_frame(
        timestamps=[
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:01:10Z",
        ],
        mos_values=[4.0, 2.0, 1.0, 4.5],
    )
    frame["session_id"] = ["first", "first", "second", "second"]

    result = add_future_target(frame)

    assert result.loc[0, "future_mos"] == 2.0
    assert pd.isna(result.loc[1, "future_mos"])
    assert pd.isna(result.loc[1, "future_timestamp"])
    assert pd.isna(result.loc[1, "future_poor_qoe"])
    assert result.loc[2, "future_mos"] == 4.5
    assert pd.isna(result.loc[3, "future_mos"])
    assert pd.isna(result.loc[3, "future_poor_qoe"])


def test_horizon_steps_selects_exact_row_within_each_session() -> None:
    frame = _trajectory_frame(
        timestamps=[
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:00:20Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:01:08Z",
            "2026-01-01T00:01:16Z",
        ],
        mos_values=[4.5, 4.0, 2.0, 2.0, 2.5, 4.5],
    )
    frame["session_id"] = [
        "first",
        "first",
        "first",
        "second",
        "second",
        "second",
    ]

    result = add_future_target(
        frame,
        horizon_steps=2,
    )

    assert result.loc[0, "future_mos"] == 2.0
    assert result.loc[0, "prediction_lead_seconds"] == 20.0
    assert result.loc[0, "future_poor_qoe"] == 1
    assert pd.isna(result.loc[1, "future_poor_qoe"])
    assert pd.isna(result.loc[2, "future_poor_qoe"])
    assert result.loc[3, "future_mos"] == 4.5
    assert result.loc[3, "prediction_lead_seconds"] == 16.0
    assert result.loc[3, "future_poor_qoe"] == 0
    assert pd.isna(result.loc[4, "future_poor_qoe"])
    assert pd.isna(result.loc[5, "future_poor_qoe"])


def test_target_does_not_peek_beyond_configured_horizon() -> None:
    frame = _trajectory_frame(
        timestamps=[
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:00:20Z",
        ],
        mos_values=[4.5, 4.0, 2.0],
    )
    frame["session_id"] = "session-1"
    changed_later_observation = frame.copy()
    changed_later_observation.loc[2, "current_mos"] = 5.0

    original = add_future_target(frame, horizon_steps=1)
    changed = add_future_target(
        changed_later_observation,
        horizon_steps=1,
    )

    assert original.loc[0, "future_poor_qoe"] == 0
    assert changed.loc[0, "future_poor_qoe"] == 0
    assert original.loc[1, "future_poor_qoe"] == 1
    assert changed.loc[1, "future_poor_qoe"] == 0
