"""Generate digital-twin temporal features and chronological splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qoe_twin.artifact_lineage import load_resolved_configuration
from qoe_twin.features import (
    build_temporal_features,
    get_cross_layer_feature_names,
    get_network_feature_names,
)
from qoe_twin.splitting import (
    assign_chronological_split,
    calculate_time_boundaries,
    remove_cross_boundary_targets,
    validate_split_order,
)

PROTOTYPE_INPUT = Path(
    "data/processed/qoe_modeling_final_sample.parquet"
)

FULL_INPUT = Path(
    "data/processed/qoe_modeling_dataset.parquet"
)
CONFIG_DIRECTORY = Path("configs")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="Process the complete 16.6-million-row dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    configuration = load_resolved_configuration(
        CONFIG_DIRECTORY
    )
    split_configuration = configuration.values[
        "data"
    ]["splitting"]

    input_path = FULL_INPUT if args.full else PROTOTYPE_INPUT

    suffix = "full" if args.full else "final"

    output_path = Path(
        f"data/processed/qoe_features_{suffix}.parquet"
    )

    split_summary_path = Path(
        f"results/tables/split_summary_{suffix}.csv"
    )

    metadata_path = Path(
        f"results/metrics/feature_metadata_{suffix}.json"
    )

    print(f"Reading: {input_path}")
    frame = pd.read_parquet(input_path)

    print(f"Input rows: {len(frame):,}")
    print("Building temporal features...")

    featured = build_temporal_features(frame)

    print("Calculating chronological boundaries...")

    boundaries = calculate_time_boundaries(
        featured,
        train_fraction=float(
            split_configuration["train_fraction"]
        ),
        validation_fraction=float(
            split_configuration[
                "validation_fraction"
            ]
        ),
    )

    featured = assign_chronological_split(
        featured,
        boundaries,
    )

    rows_before_filtering = len(featured)

    featured = remove_cross_boundary_targets(
        featured,
        boundaries,
    )

    validate_split_order(featured)

    rows_removed = rows_before_filtering - len(featured)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    featured.to_parquet(
        output_path,
        index=False,
    )

    summary = (
        featured.groupby("split")
        .agg(
            rows=("future_poor_qoe", "size"),
            sessions=("session_id", "nunique"),
            users=("user_id", "nunique"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        )
        .reset_index()
    )

    split_order = pd.CategoricalDtype(
        ["train", "validation", "test"],
        ordered=True,
    )

    summary["split"] = summary["split"].astype(
        split_order
    )

    summary = summary.sort_values("split")

    summary.to_csv(
        split_summary_path,
        index=False,
    )

    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_before_filtering": rows_before_filtering,
        "rows_after_filtering": len(featured),
        "rows_removed_at_boundaries": rows_removed,
        "train_end": boundaries.train_end.isoformat(),
        "validation_end": boundaries.validation_end.isoformat(),
        "configuration_sha256": configuration.sha256,
        "split_method": split_configuration["method"],
        "split_fractions": {
            "train": split_configuration["train_fraction"],
            "validation": split_configuration[
                "validation_fraction"
            ],
            "test": split_configuration["test_fraction"],
        },
        "network_features": get_network_feature_names(),
        "cross_layer_features": get_cross_layer_feature_names(),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nFEATURE DATASET SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nChronological boundaries")
    print(f"Train ends before: {boundaries.train_end}")
    print(
        "Validation ends before:",
        boundaries.validation_end,
    )

    print(f"\nRows removed: {rows_removed:,}")
    print(f"Feature dataset: {output_path}")
    print(f"Split summary: {split_summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
