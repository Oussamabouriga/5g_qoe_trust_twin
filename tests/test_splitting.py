"""Tests for chronological splitting."""

import pandas as pd

from qoe_twin.splitting import (
    assign_chronological_split,
    calculate_time_boundaries,
    remove_cross_boundary_targets,
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


def test_chronological_split_assignment() -> None:
    frame = create_frame()

    boundaries = calculate_time_boundaries(
        frame,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assigned = assign_chronological_split(
        frame,
        boundaries,
    )

    assert assigned.iloc[0]["split"] == "train"
    assert assigned.iloc[-1]["split"] == "test"


def test_cross_boundary_labels_are_removed() -> None:
    frame = create_frame()

    boundaries = calculate_time_boundaries(
        frame,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assigned = assign_chronological_split(
        frame,
        boundaries,
    )

    cleaned = remove_cross_boundary_targets(
        assigned,
        boundaries,
    )

    assert cleaned["future_timestamp"].notna().all()
