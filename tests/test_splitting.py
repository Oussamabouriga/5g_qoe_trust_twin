"""Semantic tests for leakage-safe chronological splitting."""

import pandas as pd

from qoe_twin.splitting import (
    SplitBoundaries,
    assign_chronological_split,
    calculate_time_boundaries,
    remove_cross_boundary_targets,
    split_validation_chronologically,
)


def create_frame() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=10,
        freq="10s",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "future_timestamp": list(timestamps[1:]) + [pd.NaT],
            "future_poor_qoe": [0] * 9 + [pd.NA],
        }
    )


def test_boundaries_and_split_labels_use_exact_chronological_cutoffs() -> None:
    frame = create_frame()
    timestamps = frame["timestamp"]

    boundaries = calculate_time_boundaries(
        frame,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assert boundaries == SplitBoundaries(
        train_end=timestamps.iloc[6],
        validation_end=timestamps.iloc[8],
    )

    assigned = assign_chronological_split(frame, boundaries)

    assert assigned["split"].tolist() == (
        ["train"] * 6
        + ["validation"] * 2
        + ["test"] * 2
    )
    assert assigned.loc[6, "split"] == "validation"
    assert assigned.loc[8, "split"] == "test"


def test_cross_boundary_horizons_are_removed_and_isolated() -> None:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=9,
        freq="10s",
        tz="UTC",
    )
    boundaries = SplitBoundaries(
        train_end=timestamps[3],
        validation_end=timestamps[6],
    )
    frame = pd.DataFrame(
        {
            "case": [
                "train_inside",
                "train_at_boundary",
                "train_beyond_boundary",
                "validation_inside",
                "validation_at_boundary",
                "validation_beyond_boundary",
                "test_with_target",
                "missing_future",
                "missing_target",
            ],
            "timestamp": timestamps,
            "future_timestamp": [
                timestamps[2],
                timestamps[3],
                timestamps[4],
                timestamps[5],
                timestamps[6],
                timestamps[7],
                timestamps[8],
                pd.NaT,
                timestamps[8],
            ],
            "future_poor_qoe": [
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                pd.NA,
            ],
        }
    )

    assigned = assign_chronological_split(frame, boundaries)
    cleaned = remove_cross_boundary_targets(assigned, boundaries)

    assert cleaned["case"].tolist() == [
        "train_inside",
        "validation_inside",
        "test_with_target",
    ]

    retained_train = cleaned[cleaned["split"] == "train"]
    retained_validation = cleaned[
        cleaned["split"] == "validation"
    ]

    assert (
        retained_train["future_timestamp"]
        < boundaries.train_end
    ).all()
    assert (
        retained_validation["future_timestamp"]
        < boundaries.validation_end
    ).all()


def test_global_split_keeps_tied_timestamps_together() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:00:20Z",
            "2026-01-01T00:00:20Z",
            "2026-01-01T00:00:30Z",
            "2026-01-01T00:00:30Z",
            "2026-01-01T00:00:40Z",
            "2026-01-01T00:00:40Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "session_id": ["a", "b"] * 5,
            "timestamp": timestamps,
        }
    )

    boundaries = calculate_time_boundaries(
        frame,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    assigned = assign_chronological_split(frame, boundaries)

    assert assigned.groupby("timestamp")["split"].nunique().eq(1).all()


def test_validation_halves_remove_labels_crossing_the_midpoint() -> None:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=6,
        freq="10s",
        tz="UTC",
    )
    validation = pd.DataFrame(
        {
            "case": [
                "fit_inside_1",
                "fit_inside_2",
                "fit_crossing",
                "select_1",
                "select_2",
                "select_3",
            ],
            "timestamp": timestamps,
            "future_timestamp": [
                timestamps[1],
                timestamps[2],
                timestamps[3],
                timestamps[4],
                timestamps[5],
                timestamps[5] + pd.Timedelta(seconds=10),
            ],
            "future_poor_qoe": [0, 1, 1, 0, 1, 0],
        }
    )

    calibration, selection = split_validation_chronologically(
        validation,
        calibration_fraction=0.5,
    )

    assert calibration["case"].tolist() == [
        "fit_inside_1",
        "fit_inside_2",
    ]
    assert selection["case"].tolist() == [
        "select_1",
        "select_2",
        "select_3",
    ]
    assert (
        calibration["future_timestamp"]
        < selection["timestamp"].min()
    ).all()


def test_validation_halves_require_two_nonempty_periods() -> None:
    timestamp = pd.Timestamp("2026-01-01", tz="UTC")
    validation = pd.DataFrame(
        {
            "timestamp": [timestamp],
            "future_timestamp": [timestamp + pd.Timedelta(seconds=10)],
            "future_poor_qoe": [0],
        }
    )

    try:
        split_validation_chronologically(
            validation,
            calibration_fraction=0.5,
        )
    except ValueError as exc:
        assert "each contain timestamps" in str(exc)
    else:
        raise AssertionError("A one-timestamp validation period must fail.")
