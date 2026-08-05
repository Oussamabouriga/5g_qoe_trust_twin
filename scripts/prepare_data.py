"""Build the unified modeling dataset for the 5G QoE twin."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qoe_twin.artifact_lineage import load_resolved_configuration
from qoe_twin.data_loader import load_scenario_file
from qoe_twin.preprocessing import convert_selected_bitrates_to_long
from qoe_twin.target import add_future_target, add_sessions

CONFIG_DIRECTORY = Path("configs")


def main() -> None:
    """Create and save the unified long-format dataset."""
    configuration = load_resolved_configuration(
        CONFIG_DIRECTORY
    )
    config = configuration.values["data"]

    dataset_config = config["dataset"]
    selected_bitrates = config["video"][
        "selected_bitrates_kbps"
    ]

    dataset_directory = Path(dataset_config["directory"])
    file_pattern = dataset_config["file_pattern"]

    files = sorted(dataset_directory.glob(file_pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found in {dataset_directory}"
        )

    processed_parts: list[pd.DataFrame] = []

    for index, path in enumerate(files, start=1):
        print(
            f"[{index:02d}/{len(files):02d}] "
            f"Processing {path.name}"
        )

        wide_frame = load_scenario_file(
            path=path,
            selected_bitrates=selected_bitrates,
        )

        long_frame = convert_selected_bitrates_to_long(
            frame=wide_frame,
            selected_bitrates=selected_bitrates,
        )

        processed_parts.append(long_frame)

    dataset = pd.concat(
        processed_parts,
        ignore_index=True,
    )

    print(f"\nRows after long conversion: {len(dataset):,}")

    dataset = add_sessions(
        dataset,
        gap_threshold_seconds=config["sessions"][
            "gap_threshold_seconds"
        ],
    )

    dataset = add_future_target(
        dataset,
        poor_mos_threshold=config["target"][
            "poor_mos_threshold"
        ],
        horizon_steps=config["target"]["horizon_steps"],
    )

    output_path = Path(
        config["processing"]["output_file"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(
        output_path,
        index=False,
    )

    valid_target = dataset[
        "future_poor_qoe"
    ].notna()

    target_distribution = (
        dataset.loc[
            valid_target,
            "future_poor_qoe",
        ]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nMODEL DATASET SUMMARY")
    print("=" * 70)
    print(f"Rows: {len(dataset):,}")
    print(
        f"Valid future targets: "
        f"{valid_target.sum():,}"
    )
    print(
        f"Sessions: "
        f"{dataset['session_id'].nunique():,}"
    )
    print(
        f"Users: "
        f"{dataset['user_id'].nunique():,}"
    )
    print(
        f"Base stations: "
        f"{sorted(dataset['base_station'].unique())}"
    )
    print(
        f"PRB values: "
        f"{sorted(dataset['prb'].unique())}"
    )
    print(
        f"Bitrates: "
        f"{sorted(dataset['bitrate_kbps'].unique())}"
    )

    print("\nFuture poor-QoE target:")
    print(target_distribution.to_string())

    poor_rate = dataset.loc[
        valid_target,
        "future_poor_qoe",
    ].astype(float).mean()

    print(f"\nPoor-QoE rate: {poor_rate:.4%}")

    print(
        "Median lead time:",
        dataset.loc[
            valid_target,
            "prediction_lead_seconds",
        ].median(),
        "seconds",
    )

    print(f"\nSaved dataset to: {output_path}")


if __name__ == "__main__":
    main()
