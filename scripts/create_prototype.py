"""Create a deterministic development dataset using complete sessions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/qoe_modeling_dataset.parquet")
OUTPUT_PATH = Path("data/processed/qoe_modeling_prototype.parquet")

MAX_SESSIONS_PER_GROUP = 250

GROUP_COLUMNS = [
    "base_station",
    "prb",
    "bitrate_kbps",
]


def main() -> None:
    """Select complete sessions across all network and video scenarios."""
    print(f"Reading session information from {INPUT_PATH}")

    session_columns = [
        "session_id",
        "base_station",
        "prb",
        "bitrate_kbps",
        "timestamp",
        "future_poor_qoe",
    ]

    session_rows = pd.read_parquet(
        INPUT_PATH,
        columns=session_columns,
    )

    session_summary = (
        session_rows.groupby(
            ["session_id", *GROUP_COLUMNS],
            as_index=False,
        )
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            row_count=("timestamp", "size"),
            poor_event_count=("future_poor_qoe", "sum"),
        )
        .sort_values(
            [*GROUP_COLUMNS, "session_start", "session_id"]
        )
    )

    selected_sessions = (
        session_summary.groupby(
            GROUP_COLUMNS,
            group_keys=False,
        )
        .head(MAX_SESSIONS_PER_GROUP)
        ["session_id"]
    )

    selected_session_ids = set(selected_sessions)

    print(f"Selected sessions: {len(selected_session_ids):,}")

    print("Reading full processed dataset...")
    full_dataset = pd.read_parquet(INPUT_PATH)

    prototype = full_dataset[
        full_dataset["session_id"].isin(selected_session_ids)
    ].copy()

    prototype = prototype.sort_values(
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

    prototype.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    valid_target = prototype["future_poor_qoe"].notna()

    print("\nPROTOTYPE SUMMARY")
    print("=" * 60)
    print(f"Rows: {len(prototype):,}")
    print(f"Sessions: {prototype['session_id'].nunique():,}")
    print(f"Users: {prototype['user_id'].nunique():,}")
    print(
        "Valid targets:",
        f"{valid_target.sum():,}",
    )
    print(
        "Poor-QoE rate:",
        f"{prototype.loc[valid_target, 'future_poor_qoe'].astype(float).mean():.2%}",
    )
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
