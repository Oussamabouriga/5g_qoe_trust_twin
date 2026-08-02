"""Analyze relationships among throughput, application values, and MOS."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_DIRECTORY = Path("data/external/5G-QoERA/5G-QoERA")
OUTPUT_DIRECTORY = Path("results/tables")

APPLICATION_PATTERN = re.compile(
    r"(?P<resolution>\d+p)_(?P<bitrate>\d+)kbps"
)


def get_pairs(
    columns: list[str],
) -> list[dict[str, str | int]]:
    """Match every application column to its MOS column."""
    pairs = []

    for column in columns:
        match = APPLICATION_PATTERN.fullmatch(column)

        if match is None:
            continue

        bitrate = int(match.group("bitrate"))
        mos_column = f"MOS_{bitrate}kbps"

        if mos_column not in columns:
            raise ValueError(
                f"No matching MOS column for {column}"
            )

        pairs.append(
            {
                "application_column": column,
                "mos_column": mos_column,
                "resolution": match.group("resolution"),
                "bitrate_kbps": bitrate,
            }
        )

    return pairs


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use one representative file first.
    path = (
        DATASET_DIRECTORY
        / "MOS_BS_1_Driver_PRB_10.tsv"
    )

    frame = pd.read_csv(path, sep="\t")
    pairs = get_pairs(frame.columns.tolist())

    summaries = []

    for pair in pairs:
        application_column = str(
            pair["application_column"]
        )
        mos_column = str(pair["mos_column"])
        bitrate = int(pair["bitrate_kbps"])

        application = frame[application_column]
        mos = frame[mos_column]

        bitrate_mbps = bitrate / 1000

        summaries.append(
            {
                **pair,
                "application_minimum": application.min(),
                "application_mean": application.mean(),
                "application_median": application.median(),
                "application_maximum": application.max(),
                "application_zero_fraction": (
                    application.eq(0).mean()
                ),
                "mos_minimum": mos.min(),
                "mos_mean": mos.mean(),
                "mos_median": mos.median(),
                "mos_maximum": mos.max(),
                "correlation_application_mos": (
                    application.corr(mos)
                ),
                "correlation_throughput_mos": (
                    frame["Tput"].corr(mos)
                ),
                "fraction_throughput_below_bitrate": (
                    frame["Tput"] < bitrate_mbps
                ).mean(),
                "mean_mos_when_capacity_sufficient": (
                    mos[
                        frame["Tput"] >= bitrate_mbps
                    ].mean()
                ),
                "mean_mos_when_capacity_insufficient": (
                    mos[
                        frame["Tput"] < bitrate_mbps
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(summaries)

    output_path = (
        OUTPUT_DIRECTORY
        / "application_mos_relationships.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print("APPLICATION AND MOS RELATIONSHIPS")
    print("=" * 140)

    display_columns = [
        "resolution",
        "bitrate_kbps",
        "application_mean",
        "application_zero_fraction",
        "mos_mean",
        "correlation_application_mos",
        "correlation_throughput_mos",
        "fraction_throughput_below_bitrate",
        "mean_mos_when_capacity_sufficient",
        "mean_mos_when_capacity_insufficient",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(
            summary[display_columns].to_string(
                index=False
            )
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
