"""Preprocessing utilities for 5G-QoERA."""

from __future__ import annotations

import pandas as pd

BITRATE_TO_RESOLUTION = {
    2000: "720p",
    4000: "1080p",
    6000: "1080p",
    8000: "1080p",
}


def convert_selected_bitrates_to_long(
    frame: pd.DataFrame,
    selected_bitrates: list[int],
) -> pd.DataFrame:
    """Convert selected bitrate-specific PLR and MOS columns to long form."""
    long_frames: list[pd.DataFrame] = []

    common_columns = [
        "User_ID",
        "Timestamp",
        "Tput",
        "Latitude",
        "Longitude",
        "base_station",
        "mobility",
        "prb",
    ]

    for bitrate in selected_bitrates:
        resolution = BITRATE_TO_RESOLUTION[bitrate]
        plr_column = f"{resolution}_{bitrate}kbps"
        mos_column = f"MOS_{bitrate}kbps"

        subset = frame[common_columns].copy()

        subset["bitrate_kbps"] = bitrate
        subset["resolution"] = resolution
        subset["plr_percent"] = frame[plr_column].astype(float)
        subset["current_mos"] = frame[mos_column].astype(float)

        long_frames.append(subset)

    result = pd.concat(
        long_frames,
        ignore_index=True,
    )

    result = result.rename(
        columns={
            "User_ID": "user_id",
            "Timestamp": "timestamp",
            "Tput": "throughput_mbps",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )

    return result
