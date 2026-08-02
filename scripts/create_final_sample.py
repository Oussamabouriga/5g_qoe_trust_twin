"""Create a large deterministic final dataset using complete sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/qoe_modeling_dataset.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/qoe_modeling_final_sample.parquet"
)

GROUP_COLUMNS = [
    "base_station",
    "prb",
    "bitrate_kbps",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-rows",
        type=int,
        default=2_000_000,
        help="Approximate target number of rows.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print("Reading session metadata...")

    columns = [
        "session_id",
        "base_station",
        "prb",
        "bitrate_kbps",
        "timestamp",
        "future_poor_qoe",
    ]

    metadata = pd.read_parquet(
        INPUT_PATH,
        columns=columns,
    )

    sessions = (
        metadata.groupby(
            ["session_id", *GROUP_COLUMNS],
            as_index=False,
        )
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            row_count=("timestamp", "size"),
            poor_events=("future_poor_qoe", "sum"),
        )
        .sort_values(
            [*GROUP_COLUMNS, "session_start", "session_id"]
        )
    )

    number_of_groups = sessions[
        GROUP_COLUMNS
    ].drop_duplicates().shape[0]

    rows_per_group = max(
        args.target_rows // number_of_groups,
        1,
    )

    selected_parts = []

    for _, group in sessions.groupby(
        GROUP_COLUMNS,
        sort=True,
    ):
        cumulative_rows = group[
            "row_count"
        ].cumsum()

        selected = group[
            cumulative_rows <= rows_per_group
        ]

        if selected.empty:
            selected = group.iloc[[0]]

        selected_parts.append(selected)

    selected_sessions = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected_ids = set(
        selected_sessions["session_id"]
    )

    expected_rows = int(
        selected_sessions["row_count"].sum()
    )

    print(
        f"Selected sessions: "
        f"{len(selected_ids):,}"
    )
    print(
        f"Expected rows: "
        f"{expected_rows:,}"
    )

    print("Reading full processed dataset...")

    dataset = pd.read_parquet(INPUT_PATH)

    final_sample = dataset[
        dataset["session_id"].isin(
            selected_ids
        )
    ].copy()

    final_sample = final_sample.sort_values(
        [
            "base_station",
            "prb",
            "bitrate_kbps",
            "user_id",
            "session_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_sample.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    valid = final_sample[
        "future_poor_qoe"
    ].notna()

    print("\nFINAL SAMPLE SUMMARY")
    print("=" * 70)
    print(
        f"Rows: "
        f"{len(final_sample):,}"
    )
    print(
        f"Sessions: "
        f"{final_sample['session_id'].nunique():,}"
    )
    print(
        f"Users: "
        f"{final_sample['user_id'].nunique():,}"
    )
    print(
        f"Valid targets: "
        f"{valid.sum():,}"
    )
    print(
        "Poor-QoE rate:",
        f"{final_sample.loc[valid, 'future_poor_qoe'].astype(float).mean():.2%}",
    )
    print(
        "Start:",
        final_sample["timestamp"].min(),
    )
    print(
        "End:",
        final_sample["timestamp"].max(),
    )
    print(
        "Base stations:",
        sorted(
            final_sample[
                "base_station"
            ].unique()
        ),
    )
    print(
        "PRB values:",
        sorted(
            final_sample["prb"].unique()
        ),
    )
    print(
        "Bitrates:",
        sorted(
            final_sample[
                "bitrate_kbps"
            ].unique()
        ),
    )
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
