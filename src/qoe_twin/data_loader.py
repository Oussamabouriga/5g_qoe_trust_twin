"""Loading utilities for the 5G-QoERA dataset."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"MOS_BS_(?P<base_station>\d+)_"
    r"(?P<mobility>[A-Za-z]+)_"
    r"PRB_(?P<prb>\d+)\.tsv"
)


def parse_filename_metadata(path: Path) -> dict[str, int | str]:
    """Extract base station, mobility type and PRB from a filename."""
    match = FILENAME_PATTERN.fullmatch(path.name)

    if match is None:
        raise ValueError(f"Unexpected dataset filename: {path.name}")

    values = match.groupdict()

    return {
        "base_station": int(values["base_station"]),
        "mobility": values["mobility"],
        "prb": int(values["prb"]),
    }


def load_scenario_file(
    path: Path,
    selected_bitrates: list[int],
) -> pd.DataFrame:
    """Load one TSV scenario with only required columns."""
    required_columns = [
        "User_ID",
        "Timestamp",
        "Tput",
        "Latitude",
        "Longitude",
    ]

    for bitrate in selected_bitrates:
        required_columns.extend(
            [
                f"MOS_{bitrate}kbps",
            ]
        )

        resolution = {
            2000: "720p",
            4000: "1080p",
            6000: "1080p",
            8000: "1080p",
        }[bitrate]

        required_columns.append(
            f"{resolution}_{bitrate}kbps"
        )

    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=required_columns,
    )

    frame["Timestamp"] = pd.to_datetime(
        frame["Timestamp"],
        errors="raise",
        utc=True,
    )

    metadata = parse_filename_metadata(path)

    for key, value in metadata.items():
        frame[key] = value

    return frame
