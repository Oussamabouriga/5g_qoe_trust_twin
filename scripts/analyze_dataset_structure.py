"""Analyze the semantic and temporal structure of the 5G-QoERA dataset."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIRECTORY = Path("data/external/5G-QoERA/5G-QoERA")

FILE_PATTERN = re.compile(
    r"MOS_BS_(?P<base_station>\d+)_Driver_PRB_(?P<prb>\d+)\.tsv"
)


def parse_metadata(path: Path) -> tuple[int, int]:
    match = FILE_PATTERN.fullmatch(path.name)

    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")

    return (
        int(match.group("base_station")),
        int(match.group("prb")),
    )


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    return {
        "identity": [
            column
            for column in columns
            if column in {"User_ID", "Timestamp"}
        ],
        "network": [
            column
            for column in columns
            if column in {"Tput", "Latitude", "Longitude"}
        ],
        "application": [
            column
            for column in columns
            if not column.startswith("MOS_")
            and column
            not in {
                "User_ID",
                "Timestamp",
                "Tput",
                "Latitude",
                "Longitude",
            }
        ],
        "mos": [
            column
            for column in columns
            if column.startswith("MOS_")
        ],
    }


def timestamp_statistics(frame: pd.DataFrame) -> dict[str, float]:
    ordered = frame.sort_values(["User_ID", "Timestamp"]).copy()

    ordered["Timestamp"] = pd.to_datetime(
        ordered["Timestamp"],
        errors="coerce",
        utc=True,
    )

    gaps = (
        ordered.groupby("User_ID")["Timestamp"]
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    return {
        "minimum_gap_seconds": float(gaps.min()),
        "median_gap_seconds": float(gaps.median()),
        "mean_gap_seconds": float(gaps.mean()),
        "p90_gap_seconds": float(gaps.quantile(0.90)),
        "p95_gap_seconds": float(gaps.quantile(0.95)),
        "maximum_gap_seconds": float(gaps.max()),
    }


def compare_prb_files(
    first_path: Path,
    second_path: Path,
) -> dict[str, object]:
    first = pd.read_csv(first_path, sep="\t")
    second = pd.read_csv(second_path, sep="\t")

    keys = ["User_ID", "Timestamp"]

    same_number_of_rows = len(first) == len(second)
    same_keys = first[keys].equals(second[keys])

    same_coordinates = first[
        ["Latitude", "Longitude"]
    ].equals(
        second[["Latitude", "Longitude"]]
    )

    throughput_equal_fraction = float(
        np.isclose(
            first["Tput"].to_numpy(),
            second["Tput"].to_numpy(),
            equal_nan=True,
        ).mean()
    )

    mos_columns = [
        column
        for column in first.columns
        if column.startswith("MOS_")
    ]

    mos_equal_fraction = float(
        np.isclose(
            first[mos_columns].to_numpy(),
            second[mos_columns].to_numpy(),
            equal_nan=True,
        ).mean()
    )

    return {
        "first_file": first_path.name,
        "second_file": second_path.name,
        "same_number_of_rows": same_number_of_rows,
        "same_user_timestamp_keys": same_keys,
        "same_coordinates": same_coordinates,
        "throughput_equal_fraction": throughput_equal_fraction,
        "mos_equal_fraction": mos_equal_fraction,
    }


def main() -> None:
    files = sorted(DATASET_DIRECTORY.glob("*.tsv"))

    if not files:
        raise FileNotFoundError("No TSV files were found.")

    first_file = files[0]
    first_frame = pd.read_csv(first_file, sep="\t")

    column_groups = classify_columns(
        first_frame.columns.tolist()
    )

    print("=" * 80)
    print("COLUMN GROUPS")
    print("=" * 80)

    for group_name, columns in column_groups.items():
        print(f"\n{group_name.upper()} ({len(columns)} columns)")
        for column in columns:
            print(f"  - {column}")

    file_summaries = []

    print("\n" + "=" * 80)
    print("PER-FILE TEMPORAL SUMMARY")
    print("=" * 80)

    for path in files:
        base_station, prb = parse_metadata(path)

        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=[
                "User_ID",
                "Timestamp",
                "Tput",
                "Latitude",
                "Longitude",
            ],
        )

        temporal = timestamp_statistics(frame)

        summary = {
            "filename": path.name,
            "base_station": base_station,
            "prb": prb,
            "rows": len(frame),
            "users": frame["User_ID"].nunique(),
            "timestamps_unique": frame["Timestamp"].nunique(),
            **temporal,
        }

        file_summaries.append(summary)

        print(
            f"{path.name}: "
            f"users={summary['users']}, "
            f"median_gap={summary['median_gap_seconds']:.3f}s, "
            f"p95_gap={summary['p95_gap_seconds']:.3f}s"
        )

    output_directory = Path("results/tables")
    output_directory.mkdir(parents=True, exist_ok=True)

    temporal_summary = pd.DataFrame(file_summaries)

    temporal_output = (
        output_directory
        / "dataset_temporal_summary.csv"
    )

    temporal_summary.to_csv(
        temporal_output,
        index=False,
    )

    comparisons = []

    print("\n" + "=" * 80)
    print("PRB FILE COMPARISONS")
    print("=" * 80)

    for base_station in sorted(
        temporal_summary["base_station"].unique()
    ):
        base_station_files = [
            path
            for path in files
            if parse_metadata(path)[0] == base_station
        ]

        lowest_prb_file = min(
            base_station_files,
            key=lambda path: parse_metadata(path)[1],
        )

        highest_prb_file = max(
            base_station_files,
            key=lambda path: parse_metadata(path)[1],
        )

        comparison = compare_prb_files(
            lowest_prb_file,
            highest_prb_file,
        )

        comparison["base_station"] = base_station
        comparisons.append(comparison)

        print(f"\nBase station {base_station}")
        for key, value in comparison.items():
            if key != "base_station":
                print(f"  {key}: {value}")

    comparison_frame = pd.DataFrame(comparisons)

    comparison_output = (
        output_directory
        / "prb_file_comparisons.csv"
    )

    comparison_frame.to_csv(
        comparison_output,
        index=False,
    )

    mos_columns = column_groups["mos"]

    mos_values = first_frame[mos_columns]

    mos_summary = pd.DataFrame(
        {
            "column": mos_columns,
            "minimum": mos_values.min().values,
            "mean": mos_values.mean().values,
            "median": mos_values.median().values,
            "maximum": mos_values.max().values,
            "standard_deviation": mos_values.std().values,
        }
    )

    mos_output = (
        output_directory
        / "mos_column_summary.csv"
    )

    mos_summary.to_csv(
        mos_output,
        index=False,
    )

    print("\n" + "=" * 80)
    print("MOS SUMMARY FOR FIRST FILE")
    print("=" * 80)
    print(mos_summary.to_string(index=False))

    print("\nGenerated files:")
    print(f"- {temporal_output}")
    print(f"- {comparison_output}")
    print(f"- {mos_output}")


if __name__ == "__main__":
    main()
