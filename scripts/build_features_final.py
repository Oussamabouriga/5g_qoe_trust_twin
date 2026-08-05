"""Validate the audited sample and build the corrected CP6 feature dataset."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from qoe_twin.artifact_lineage import (
    load_resolved_configuration,
    sha256_file,
)
from qoe_twin.config import canonical_sha256
from qoe_twin.features import (
    build_temporal_features,
    get_cross_layer_feature_names,
    get_network_feature_names,
)
from qoe_twin.sample_validation import (
    BASE_SAMPLE_COLUMNS,
    INHERITED_TARGET_COLUMNS,
    ROW_IDENTITY_COLUMNS,
    parquet_metadata,
    validate_and_rebuild_sample,
    validate_audited_sample_artifact,
    validate_raw_content_compatibility,
)
from qoe_twin.splitting import (
    assign_chronological_split,
    calculate_time_boundaries,
    remove_cross_boundary_targets,
    validate_split_order,
)

CONFIG_DIRECTORY = Path("configs")
REPOSITORY_ROOT = Path(".")
SPLIT_ORDER = ("train", "validation", "test")


def _split_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_name in SPLIT_ORDER:
        split = frame[frame["split"].eq(split_name)]
        records.append(
            {
                "end_timestamp": split["timestamp"].max().isoformat(),
                "rows": len(split),
                "sessions": int(split["session_id"].nunique()),
                "split": split_name,
                "start_timestamp": split["timestamp"].min().isoformat(),
            }
        )
    return records


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """Run the bounded CP6 validation and causal feature rebuild."""
    configuration = load_resolved_configuration(CONFIG_DIRECTORY)
    data_configuration = configuration.values["data"]
    sample_configuration = data_configuration["sample"]
    split_configuration = data_configuration["splitting"]
    target_configuration = data_configuration["target"]
    video_configuration = data_configuration["video"]

    source_artifact = validate_audited_sample_artifact(
        REPOSITORY_ROOT,
        source_relative_path=str(sample_configuration["input_file"]),
        expected_source_sha256=str(sample_configuration["input_sha256"]),
        provenance_manifest_relative_path=str(
            sample_configuration["provenance_manifest"]
        ),
        expected_provenance_manifest_sha256=str(
            sample_configuration["provenance_manifest_sha256"]
        ),
    )
    print(
        "Validated source identity/footer:",
        source_artifact.source_relative_path,
    )

    frame = pd.read_parquet(source_artifact.source_path)
    selected_bitrates = tuple(video_configuration["selected_bitrates_kbps"])
    bitrate_to_resolution = dict(video_configuration["bitrate_to_resolution"])
    rebuilt, sample_validation = validate_and_rebuild_sample(
        frame,
        artifact=source_artifact,
        gap_threshold_seconds=float(
            data_configuration["sessions"]["gap_threshold_seconds"]
        ),
        poor_mos_threshold=float(target_configuration["poor_mos_threshold"]),
        horizon_steps=int(target_configuration["horizon_steps"]),
        selected_bitrates=selected_bitrates,
        bitrate_to_resolution=bitrate_to_resolution,
    )
    del frame
    gc.collect()

    print("Matching every sample row to the trusted raw TSVs...")
    raw_validation = validate_raw_content_compatibility(
        rebuilt.loc[:, BASE_SAMPLE_COLUMNS],
        repository_root=REPOSITORY_ROOT,
        artifact=source_artifact,
        separator=str(data_configuration["dataset"]["separator"]),
        selected_bitrates=selected_bitrates,
        bitrate_to_resolution=bitrate_to_resolution,
    )

    print("Building causal past-only features...")
    featured = build_temporal_features(rebuilt)
    del rebuilt
    gc.collect()

    boundaries = calculate_time_boundaries(
        featured,
        train_fraction=float(split_configuration["train_fraction"]),
        validation_fraction=float(split_configuration["validation_fraction"]),
    )
    featured = assign_chronological_split(featured, boundaries)

    terminal_rows = int(featured["future_poor_qoe"].isna().sum())
    train_crossing_rows = int(
        (
            featured["split"].eq("train")
            & featured["future_timestamp"].ge(boundaries.train_end)
        ).sum()
    )
    validation_crossing_rows = int(
        (
            featured["split"].eq("validation")
            & featured["future_timestamp"].ge(boundaries.validation_end)
        ).sum()
    )
    rows_before_filtering = len(featured)
    featured = remove_cross_boundary_targets(featured, boundaries)
    validate_split_order(featured)
    featured = featured.sort_values(
        ["timestamp", "session_id"],
        kind="stable",
    ).reset_index(drop=True)

    output_relative_path = str(sample_configuration["corrected_feature_file"])
    output_path = REPOSITORY_ROOT / output_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(output_path, index=False)

    output_metadata = parquet_metadata(output_path)
    manifest = {
        "artifact_kind": "corrected_causal_feature_dataset",
        "configuration_sha256": configuration.sha256,
        "input": {
            "byte_size": source_artifact.byte_size,
            "columns": source_artifact.columns,
            "friend_root_commit": source_artifact.friend_root_commit,
            "path": source_artifact.source_relative_path,
            "provenance_manifest_path": (
                source_artifact.provenance_manifest_relative_path
            ),
            "provenance_manifest_sha256": (
                source_artifact.provenance_manifest_sha256
            ),
            "provenance_status": source_artifact.provenance_status,
            "rows": source_artifact.rows,
            "schema_sha256": source_artifact.schema_sha256,
            "sha256": source_artifact.source_sha256,
            "upstream_repository_commit": (
                source_artifact.upstream_repository_commit
            ),
        },
        "limitations": [
            (
                "The historical sampling command and producing configuration "
                "are unavailable."
            ),
            (
                "The group-balanced sample is an earliest-time prefix, not a "
                "random or full-dataset sample."
            ),
            (
                "Four bitrate rows share each physical radio observation and "
                "are not independent."
            ),
        ],
        "output": {
            "byte_size": output_path.stat().st_size,
            "columns": output_metadata["columns"],
            "path": output_relative_path,
            "rows": output_metadata["rows"],
            "schema_sha256": canonical_sha256(output_metadata["schema"]),
            "sha256": sha256_file(output_path),
        },
        "policies": {
            "discarded_inherited_session_columns": [
                "gap_seconds",
                "is_session_start",
                "session_number",
                "session_id",
            ],
            "discarded_inherited_target_columns": list(
                INHERITED_TARGET_COLUMNS
            ),
            "global_split_fractions": {
                "test": split_configuration["test_fraction"],
                "train": split_configuration["train_fraction"],
                "validation": split_configuration["validation_fraction"],
            },
            "horizon_steps": target_configuration["horizon_steps"],
            "ordered_cross_layer_features": get_cross_layer_feature_names(),
            "ordered_network_features": get_network_feature_names(),
            "poor_qoe_rule": (
                "future_mos < "
                f"{float(target_configuration['poor_mos_threshold'])}"
            ),
            "row_identity_columns": list(ROW_IDENTITY_COLUMNS),
            "selected_bitrates_kbps": list(selected_bitrates),
            "bitrate_to_resolution": bitrate_to_resolution,
            "session_gap_seconds": data_configuration["sessions"][
                "gap_threshold_seconds"
            ],
            "split_method": split_configuration["method"],
            "stored_row_order": ["timestamp", "session_id"],
        },
        "raw_compatibility": raw_validation,
        "sample_validation": sample_validation,
        "schema_version": "1.0.0",
        "split": {
            "records": _split_records(featured),
            "rows_before_filtering": rows_before_filtering,
            "rows_removed": {
                "session_terminal_target_na": terminal_rows,
                "train_to_validation_crossing": train_crossing_rows,
                "validation_to_test_crossing": validation_crossing_rows,
            },
            "train_end_exclusive": boundaries.train_end.isoformat(),
            "validation_end_exclusive": boundaries.validation_end.isoformat(),
        },
    }
    manifest_relative_path = str(sample_configuration["split_manifest_file"])
    manifest_path = REPOSITORY_ROOT / manifest_relative_path
    _write_manifest(manifest_path, manifest)

    print("\nCP6 DATASET SUMMARY")
    print("=" * 80)
    print(pd.DataFrame(manifest["split"]["records"]).to_string(index=False))
    print(f"Rows removed: {rows_before_filtering - len(featured):,}")
    print(f"Corrected dataset: {output_path}")
    print(f"Sample/split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
