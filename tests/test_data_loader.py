"""Artifact-independent tests for dataset loading contracts."""

from pathlib import Path

import pandas as pd
import pytest

from qoe_twin.data_loader import (
    load_scenario_file,
    parse_filename_metadata,
)


def test_parse_filename_metadata_for_supported_scenario_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "MOS_BS_12_Driver_PRB_25.tsv"

    assert parse_filename_metadata(path) == {
        "base_station": 12,
        "mobility": "Driver",
        "prb": 25,
    }


@pytest.mark.parametrize(
    "filename",
    [
        "MOS_BS_1_Driver_PRB_10.csv",
        "MOS_BS_Driver_PRB_10.tsv",
        "unexpected.tsv",
    ],
)
def test_parse_filename_metadata_rejects_unsupported_name(
    filename: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unexpected dataset filename",
    ):
        parse_filename_metadata(Path(filename))


def test_load_scenario_file_reads_selected_tsv_data_and_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "MOS_BS_3_Driver_PRB_15.tsv"
    source = pd.DataFrame(
        {
            "User_ID": [7, 8],
            "Timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:10Z",
            ],
            "Tput": [12.5, 8.0],
            "Latitude": [48.0, 48.1],
            "Longitude": [2.0, 2.1],
            "MOS_2000kbps": [4.2, 3.8],
            "720p_2000kbps": [0.1, 0.2],
            "MOS_4000kbps": [3.9, 3.4],
            "1080p_4000kbps": [0.3, 0.4],
            "MOS_6000kbps": [3.2, 2.7],
            "1080p_6000kbps": [0.5, 0.6],
            "MOS_8000kbps": [2.8, 2.2],
            "1080p_8000kbps": [0.7, 0.8],
            "unused_column": ["not", "loaded"],
        }
    )
    source.to_csv(path, sep="\t", index=False)

    loaded = load_scenario_file(
        path,
        selected_bitrates=[2000, 8000],
    )

    expected_source_columns = {
        "User_ID",
        "Timestamp",
        "Tput",
        "Latitude",
        "Longitude",
        "MOS_2000kbps",
        "720p_2000kbps",
        "MOS_8000kbps",
        "1080p_8000kbps",
    }
    assert set(loaded.columns) == expected_source_columns | {
        "base_station",
        "mobility",
        "prb",
    }
    assert "unused_column" not in loaded
    assert "MOS_4000kbps" not in loaded
    assert "1080p_4000kbps" not in loaded
    assert "MOS_6000kbps" not in loaded
    assert "1080p_6000kbps" not in loaded
    assert "MOS_2000kbps" in loaded
    assert "720p_2000kbps" in loaded
    assert "MOS_8000kbps" in loaded
    assert "1080p_8000kbps" in loaded
    assert loaded["Timestamp"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:10Z"),
    ]
    assert str(loaded["Timestamp"].dtype) == (
        "datetime64[ns, UTC]"
    )
    assert loaded["base_station"].tolist() == [3, 3]
    assert loaded["mobility"].tolist() == [
        "Driver",
        "Driver",
    ]
    assert loaded["prb"].tolist() == [15, 15]
    assert loaded["MOS_8000kbps"].tolist() == [
        2.8,
        2.2,
    ]


def test_load_scenario_file_rejects_missing_required_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "MOS_BS_3_Driver_PRB_15.tsv"
    source_missing_throughput = pd.DataFrame(
        {
            "User_ID": [7],
            "Timestamp": ["2026-01-01T00:00:00Z"],
            "Latitude": [48.0],
            "Longitude": [2.0],
            "MOS_2000kbps": [4.2],
            "720p_2000kbps": [0.1],
        }
    )
    source_missing_throughput.to_csv(
        path,
        sep="\t",
        index=False,
    )

    with pytest.raises(ValueError):
        load_scenario_file(
            path,
            selected_bitrates=[2000],
        )
