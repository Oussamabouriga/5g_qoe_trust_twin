"""Leakage-safe chronological splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitBoundaries:
    """Timestamp boundaries used for chronological splitting."""

    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def calculate_time_boundaries(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> SplitBoundaries:
    """Calculate global timestamp boundaries."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError(
            "Split fractions must be positive."
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "Training and validation fractions must sum to less than 1."
        )

    unique_timestamps = (
        frame["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    train_index = min(
        int(len(unique_timestamps) * train_fraction),
        len(unique_timestamps) - 1,
    )

    validation_index = min(
        int(
            len(unique_timestamps)
            * (train_fraction + validation_fraction)
        ),
        len(unique_timestamps) - 1,
    )

    return SplitBoundaries(
        train_end=unique_timestamps.iloc[train_index],
        validation_end=unique_timestamps.iloc[validation_index],
    )


def assign_chronological_split(
    frame: pd.DataFrame,
    boundaries: SplitBoundaries,
) -> pd.DataFrame:
    """
    Assign rows to train, validation and test periods.

    Rows whose future target crosses a split boundary are removed later.
    """
    result = frame.copy()

    result["split"] = "test"

    result.loc[
        result["timestamp"] < boundaries.validation_end,
        "split",
    ] = "validation"

    result.loc[
        result["timestamp"] < boundaries.train_end,
        "split",
    ] = "train"

    return result


def remove_cross_boundary_targets(
    frame: pd.DataFrame,
    boundaries: SplitBoundaries,
) -> pd.DataFrame:
    """Remove labels whose future observation belongs to another split."""
    result = frame.copy()

    train_crossing = (
        result["split"].eq("train")
        & (
            result["future_timestamp"]
            >= boundaries.train_end
        )
    )

    validation_crossing = (
        result["split"].eq("validation")
        & (
            result["future_timestamp"]
            >= boundaries.validation_end
        )
    )

    invalid_target = (
        result["future_timestamp"].isna()
        | result["future_poor_qoe"].isna()
        | train_crossing
        | validation_crossing
    )

    result = result.loc[
        ~invalid_target
    ].copy()

    return result


def validate_split_order(
    frame: pd.DataFrame,
) -> None:
    """Raise an error when chronological ordering is violated."""
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Every chronological split must contain observations."
        )

    if train["timestamp"].max() >= validation["timestamp"].min():
        raise ValueError(
            "Training and validation periods overlap."
        )

    if validation["timestamp"].max() >= test["timestamp"].min():
        raise ValueError(
            "Validation and test periods overlap."
        )
