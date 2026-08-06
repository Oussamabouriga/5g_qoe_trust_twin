"""Session and future-target construction."""

from __future__ import annotations

import pandas as pd

TRAJECTORY_COLUMNS = [
    "base_station",
    "prb",
    "bitrate_kbps",
    "user_id",
]


def add_sessions(
    frame: pd.DataFrame,
    gap_threshold_seconds: float = 30.0,
) -> pd.DataFrame:
    """Create temporal sessions when gaps exceed the threshold."""
    result = frame.sort_values(
        TRAJECTORY_COLUMNS + ["timestamp"]
    ).reset_index(drop=True)

    result["gap_seconds"] = (
        result.groupby(TRAJECTORY_COLUMNS)["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    result["is_session_start"] = (
        result["gap_seconds"].isna()
        | (result["gap_seconds"] > gap_threshold_seconds)
    )

    result["session_number"] = (
        result.groupby(TRAJECTORY_COLUMNS)[
            "is_session_start"
        ]
        .cumsum()
        .astype("int32")
    )

    result["session_id"] = (
        result["base_station"].astype(str)
        + "_"
        + result["prb"].astype(str)
        + "_"
        + result["bitrate_kbps"].astype(str)
        + "_"
        + result["user_id"].astype(str)
        + "_"
        + result["session_number"].astype(str)
    )

    return result


def add_future_target(
    frame: pd.DataFrame,
    poor_mos_threshold: float = 3.0,
    horizon_steps: int = 1,
) -> pd.DataFrame:
    """Predict poor QoE at the next observation within the same session."""
    result = frame.copy()

    result["future_mos"] = (
        result.groupby("session_id")["current_mos"]
        .shift(-horizon_steps)
    )

    result["future_timestamp"] = (
        result.groupby("session_id")["timestamp"]
        .shift(-horizon_steps)
    )

    result["prediction_lead_seconds"] = (
        result["future_timestamp"] - result["timestamp"]
    ).dt.total_seconds()

    result["future_poor_qoe"] = (
        result["future_mos"] < poor_mos_threshold
    ).astype("Int8")

    # Last observations of sessions have no valid future target.
    result.loc[
        result["future_mos"].isna(),
        "future_poor_qoe",
    ] = pd.NA

    return result
