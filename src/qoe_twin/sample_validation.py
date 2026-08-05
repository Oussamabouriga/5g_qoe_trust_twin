"""Validation for the frozen, manageable CP6 modeling sample."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qoe_twin.artifact_lineage import sha256_file
from qoe_twin.config import canonical_sha256
from qoe_twin.data_loader import parse_filename_metadata
from qoe_twin.target import TRAJECTORY_COLUMNS, add_future_target, add_sessions


class SampleValidationError(ValueError):
    """Raised when the audited CP6 input does not satisfy its contract."""


BASE_SAMPLE_COLUMNS = (
    "user_id",
    "timestamp",
    "throughput_mbps",
    "latitude",
    "longitude",
    "base_station",
    "mobility",
    "prb",
    "bitrate_kbps",
    "resolution",
    "plr_percent",
    "current_mos",
)

SESSION_COLUMNS = (
    "gap_seconds",
    "is_session_start",
    "session_number",
    "session_id",
)

INHERITED_TARGET_COLUMNS = (
    "future_mos",
    "future_timestamp",
    "prediction_lead_seconds",
    "future_poor_qoe",
)

EXPECTED_SAMPLE_COLUMNS = (
    *BASE_SAMPLE_COLUMNS,
    *SESSION_COLUMNS,
    *INHERITED_TARGET_COLUMNS,
)

ROW_IDENTITY_COLUMNS = (
    "base_station",
    "mobility",
    "prb",
    "user_id",
    "timestamp",
    "bitrate_kbps",
)

PHYSICAL_OBSERVATION_COLUMNS = (
    "base_station",
    "mobility",
    "prb",
    "user_id",
    "timestamp",
)

EXPECTED_ARROW_SCHEMA = (
    ("user_id", "int64", True),
    ("timestamp", "timestamp[ns, tz=UTC]", True),
    ("throughput_mbps", "double", True),
    ("latitude", "double", True),
    ("longitude", "double", True),
    ("base_station", "int64", True),
    ("mobility", "string", True),
    ("prb", "int64", True),
    ("bitrate_kbps", "int64", True),
    ("resolution", "string", True),
    ("plr_percent", "double", True),
    ("current_mos", "double", True),
    ("gap_seconds", "double", True),
    ("is_session_start", "bool", True),
    ("session_number", "int32", True),
    ("session_id", "string", True),
    ("future_mos", "double", True),
    ("future_timestamp", "timestamp[ns, tz=UTC]", True),
    ("prediction_lead_seconds", "double", True),
    ("future_poor_qoe", "int8", True),
)

_RAW_FILENAME_RE = re.compile(
    r"MOS_BS_(?P<base_station>\d+)_"
    r"(?P<mobility>[A-Za-z]+)_PRB_(?P<prb>\d+)\.tsv"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RawTsvArtifact:
    """Pinned byte identity for one trusted upstream TSV."""

    relative_path: str
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class AuditedSampleArtifact:
    """Pinned provenance and footer metadata for the manageable input."""

    source_relative_path: str
    source_path: Path
    source_sha256: str
    byte_size: int
    rows: int
    columns: int
    schema: tuple[tuple[str, str, bool], ...]
    schema_sha256: str
    provenance_manifest_relative_path: str
    provenance_manifest_sha256: str
    provenance_status: str
    friend_root_commit: str
    upstream_repository_commit: str
    raw_tsv_header: tuple[str, ...]
    raw_tsv_artifacts: tuple[RawTsvArtifact, ...]

    @property
    def raw_tsv_relative_paths(self) -> tuple[str, ...]:
        """Return the ordered raw paths for scenario validation."""
        return tuple(artifact.relative_path for artifact in self.raw_tsv_artifacts)


def parquet_metadata(path: Path) -> dict[str, Any]:
    """Return deterministic footer metadata without loading the table."""
    try:
        parquet = pq.ParquetFile(path)
    except (OSError, ValueError) as exc:
        raise SampleValidationError(
            f"Cannot inspect Parquet file {path}: {exc}"
        ) from exc

    schema = [
        {
            "name": field.name,
            "nullable": field.nullable,
            "type": str(field.type),
        }
        for field in parquet.schema_arrow
    ]
    compression = sorted(
        {
            parquet.metadata.row_group(row_group)
            .column(column)
            .compression
            for row_group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.num_columns)
        }
    )
    return {
        "columns": parquet.metadata.num_columns,
        "compression": compression,
        "row_groups": parquet.metadata.num_row_groups,
        "rows": parquet.metadata.num_rows,
        "schema": schema,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SampleValidationError(
            f"Cannot load provenance manifest {path}: {exc}"
        ) from exc
    if type(value) is not dict:
        raise SampleValidationError("The provenance manifest must contain one object.")
    return value


def _schema_tuples(schema: list[dict[str, Any]]) -> tuple[tuple[str, str, bool], ...]:
    try:
        result = tuple(
            (field["name"], field["type"], field["nullable"])
            for field in schema
        )
    except (KeyError, TypeError) as exc:
        raise SampleValidationError(
            "The recorded Parquet schema is malformed."
        ) from exc
    if any(
        type(name) is not str
        or type(data_type) is not str
        or type(nullable) is not bool
        for name, data_type, nullable in result
    ):
        raise SampleValidationError("The recorded Parquet schema has invalid fields.")
    return result


def validate_audited_sample_artifact(
    repository_root: Path,
    *,
    source_relative_path: str,
    expected_source_sha256: str,
    provenance_manifest_relative_path: str,
    expected_provenance_manifest_sha256: str,
) -> AuditedSampleArtifact:
    """Validate hashes, provenance record and exact Parquet footer first."""
    root = repository_root.resolve()
    source_path = (root / source_relative_path).resolve()
    manifest_path = (root / provenance_manifest_relative_path).resolve()
    if not source_path.is_relative_to(root) or not manifest_path.is_relative_to(
        root
    ):
        raise SampleValidationError(
            "CP6 artifact paths must remain inside the repository."
        )

    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != expected_provenance_manifest_sha256:
        raise SampleValidationError(
            "The provenance manifest SHA-256 does not match the frozen CP6 value."
        )
    source_hash = sha256_file(source_path)
    if source_hash != expected_source_sha256:
        raise SampleValidationError(
            "The manageable sample SHA-256 does not match the frozen CP6 value."
        )

    manifest = _load_manifest(manifest_path)
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not list:
        raise SampleValidationError(
            "The provenance manifest has no artifact inventory."
        )
    records = [
        record
        for record in artifacts
        if type(record) is dict
        and record.get("destination_path") == source_relative_path
    ]
    if len(records) != 1:
        raise SampleValidationError(
            "The manageable sample must have exactly one provenance record."
        )
    record = records[0]
    if record.get("role") != "friend_processed_snapshot":
        raise SampleValidationError("The manageable sample has an unexpected role.")
    if record.get("status") != "quarantine":
        raise SampleValidationError(
            "The supplied sample must retain its recorded quarantine status."
        )
    if record.get("sha256") != source_hash:
        raise SampleValidationError(
            "The sample hash disagrees with its provenance record."
        )
    if record.get("byte_size") != source_path.stat().st_size:
        raise SampleValidationError(
            "The sample byte size disagrees with its provenance record."
        )

    actual_metadata = parquet_metadata(source_path)
    if record.get("parquet_metadata") != actual_metadata:
        raise SampleValidationError(
            "The sample Parquet footer or schema disagrees with its provenance record."
        )
    schema = _schema_tuples(actual_metadata["schema"])
    if schema != EXPECTED_ARROW_SCHEMA:
        raise SampleValidationError(
            "The manageable sample schema is not the CP6 schema."
        )

    raw_contract = manifest.get("raw_tsv_contract")
    if type(raw_contract) is not dict:
        raise SampleValidationError("The provenance manifest has no raw TSV contract.")
    raw_header = raw_contract.get("header")
    if (
        type(raw_header) is not list
        or not raw_header
        or any(type(column) is not str for column in raw_header)
        or len(raw_header) != len(set(raw_header))
    ):
        raise SampleValidationError("The raw TSV header contract is invalid.")
    raw_records = sorted(
        (
            record
            for record in artifacts
            if type(record) is dict
            and record.get("role") == "upstream_raw_tsv"
            and record.get("status") == "trusted_raw_input"
        ),
        key=lambda value: str(value.get("destination_path")),
    )
    if len(raw_records) != raw_contract.get("file_count"):
        raise SampleValidationError("The trusted raw TSV inventory is incomplete.")
    raw_artifacts: list[RawTsvArtifact] = []
    for raw_record in raw_records:
        relative_path = raw_record.get("destination_path")
        byte_size = raw_record.get("byte_size")
        raw_sha256 = raw_record.get("sha256")
        if (
            type(relative_path) is not str
            or type(byte_size) is not int
            or byte_size < 0
            or type(raw_sha256) is not str
            or _SHA256_RE.fullmatch(raw_sha256) is None
        ):
            raise SampleValidationError(
                "A trusted raw TSV provenance record is malformed."
            )
        raw_artifacts.append(
            RawTsvArtifact(
                relative_path=relative_path,
                byte_size=byte_size,
                sha256=raw_sha256,
            )
        )

    return AuditedSampleArtifact(
        source_relative_path=source_relative_path,
        source_path=source_path,
        source_sha256=source_hash,
        byte_size=source_path.stat().st_size,
        rows=actual_metadata["rows"],
        columns=actual_metadata["columns"],
        schema=schema,
        schema_sha256=canonical_sha256(actual_metadata["schema"]),
        provenance_manifest_relative_path=provenance_manifest_relative_path,
        provenance_manifest_sha256=manifest_hash,
        provenance_status="quarantine",
        friend_root_commit=str(manifest.get("friend_root_commit")),
        upstream_repository_commit=str(manifest.get("upstream_repository_commit")),
        raw_tsv_header=tuple(raw_header),
        raw_tsv_artifacts=tuple(raw_artifacts),
    )


def _scenario_key(path: str | Path) -> tuple[int, str, int]:
    match = _RAW_FILENAME_RE.fullmatch(Path(path).name)
    if match is None:
        raise SampleValidationError(f"Unexpected raw TSV filename: {Path(path).name}")
    values = match.groupdict()
    return int(values["base_station"]), values["mobility"], int(values["prb"])


def _equal_with_na(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def validate_and_rebuild_sample(
    frame: pd.DataFrame,
    *,
    artifact: AuditedSampleArtifact,
    gap_threshold_seconds: float,
    poor_mos_threshold: float,
    horizon_steps: int,
    selected_bitrates: tuple[int, ...],
    bitrate_to_resolution: dict[int, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate identities/session order, then discard and rebuild derivations."""
    if tuple(frame.columns) != EXPECTED_SAMPLE_COLUMNS:
        raise SampleValidationError(
            "Loaded sample columns are missing, extra or reordered."
        )
    if len(frame) != artifact.rows:
        raise SampleValidationError(
            "Loaded sample row count disagrees with its footer."
        )
    if frame.loc[:, BASE_SAMPLE_COLUMNS].isna().any().any():
        raise SampleValidationError("The manageable sample has null base measurements.")
    if not isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype):
        raise SampleValidationError("Sample timestamps must be timezone-aware.")

    duplicate_count = int(frame.duplicated(list(ROW_IDENTITY_COLUMNS)).sum())
    if duplicate_count:
        raise SampleValidationError(
            "The manageable sample has duplicate row identities."
        )
    repeated_session_time = int(frame.duplicated(["session_id", "timestamp"]).sum())
    if repeated_session_time:
        raise SampleValidationError("A session contains duplicate timestamps.")
    non_increasing = frame.groupby("session_id", sort=False)["timestamp"].diff().le(
        pd.Timedelta(0)
    )
    if bool(non_increasing.any()):
        raise SampleValidationError("Session timestamps are not strictly increasing.")

    expected_bitrates = set(selected_bitrates)
    if set(frame["bitrate_kbps"].unique()) != expected_bitrates:
        raise SampleValidationError("Sample bitrates do not match the resolved YAML.")
    for bitrate, resolution in bitrate_to_resolution.items():
        observed = set(
            frame.loc[frame["bitrate_kbps"].eq(bitrate), "resolution"].unique()
        )
        if observed != {resolution}:
            raise SampleValidationError(
                f"Resolution mapping is invalid for bitrate {bitrate}."
            )
    sibling_counts = frame.groupby(
        list(PHYSICAL_OBSERVATION_COLUMNS),
        sort=False,
    )["bitrate_kbps"].nunique()
    sibling_violations = int(sibling_counts.ne(len(selected_bitrates)).sum())
    if sibling_violations:
        raise SampleValidationError(
            "A physical observation does not contain every selected bitrate."
        )

    raw_scenarios = {_scenario_key(path) for path in artifact.raw_tsv_relative_paths}
    sample_scenarios = set(
        frame[["base_station", "mobility", "prb"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if sample_scenarios != raw_scenarios:
        raise SampleValidationError(
            "Sample scenario coverage does not match the trusted raw TSV inventory."
        )

    base = frame.loc[:, BASE_SAMPLE_COLUMNS].copy()
    rebuilt_sessions = add_sessions(
        base,
        gap_threshold_seconds=gap_threshold_seconds,
    )
    source_ordered = frame.sort_values(
        [*TRAJECTORY_COLUMNS, "timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    session_mismatches = {
        "gap_seconds": int(
            (~np.isclose(
                source_ordered["gap_seconds"].to_numpy(),
                rebuilt_sessions["gap_seconds"].to_numpy(),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )).sum()
        ),
        "is_session_start": int(
            source_ordered["is_session_start"]
            .ne(rebuilt_sessions["is_session_start"])
            .sum()
        ),
        "session_number": int(
            source_ordered["session_number"]
            .ne(rebuilt_sessions["session_number"])
            .sum()
        ),
        "session_id": int(
            source_ordered["session_id"].ne(rebuilt_sessions["session_id"]).sum()
        ),
    }
    if any(session_mismatches.values()):
        raise SampleValidationError(
            "Rebuilt session fields disagree with the audited sample."
        )

    rebuilt = add_future_target(
        rebuilt_sessions,
        poor_mos_threshold=poor_mos_threshold,
        horizon_steps=horizon_steps,
    )
    target_mismatches = {
        "future_mos": int(
            (~np.isclose(
                source_ordered["future_mos"].to_numpy(),
                rebuilt["future_mos"].to_numpy(),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )).sum()
        ),
        "future_timestamp": int(
            (~_equal_with_na(
                source_ordered["future_timestamp"],
                rebuilt["future_timestamp"],
            )).sum()
        ),
        "prediction_lead_seconds": int(
            (~np.isclose(
                source_ordered["prediction_lead_seconds"].to_numpy(),
                rebuilt["prediction_lead_seconds"].to_numpy(),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )).sum()
        ),
        "future_poor_qoe": int(
            (~_equal_with_na(
                source_ordered["future_poor_qoe"],
                rebuilt["future_poor_qoe"],
            )).sum()
        ),
    }
    if any(target_mismatches.values()):
        raise SampleValidationError(
            "Rebuilt next-observation targets disagree with the audited sample."
        )
    valid_target = rebuilt["future_mos"].notna()
    expected_target = rebuilt["future_mos"].lt(poor_mos_threshold).astype("Int8")
    expected_target = expected_target.mask(~valid_target)
    if not _equal_with_na(expected_target, rebuilt["future_poor_qoe"]).all():
        raise SampleValidationError(
            "The rebuilt target does not use strict MOS < threshold."
        )
    if not rebuilt.loc[valid_target, "prediction_lead_seconds"].gt(0).all():
        raise SampleValidationError(
            "A rebuilt target does not point to a later observation."
        )

    session_count = int(rebuilt["session_id"].nunique())
    terminal_count = int(rebuilt["future_poor_qoe"].isna().sum())
    if terminal_count != session_count:
        raise SampleValidationError(
            "Every rebuilt session must have one terminal target NA."
        )

    summary = {
        "duplicate_row_identities": duplicate_count,
        "duplicate_session_timestamps": repeated_session_time,
        "input_globally_timestamp_sorted": bool(
            frame["timestamp"].is_monotonic_increasing
        ),
        "physical_observations": len(sibling_counts),
        "physical_observations_with_missing_bitrates": sibling_violations,
        "rows": len(rebuilt),
        "sessions": session_count,
        "session_field_mismatches": session_mismatches,
        "target_field_mismatches": target_mismatches,
        "terminal_target_rows": terminal_count,
        "valid_target_rows": int(valid_target.sum()),
        "boundary_value_counts": {
            "2.999": int(rebuilt["future_mos"].eq(2.999).sum()),
            "3.000": int(rebuilt["future_mos"].eq(3.0).sum()),
            "3.001": int(rebuilt["future_mos"].eq(3.001).sum()),
        },
    }
    return rebuilt, summary


def _required_raw_columns(
    selected_bitrates: tuple[int, ...],
    bitrate_to_resolution: dict[int, str],
) -> list[str]:
    columns = ["User_ID", "Timestamp", "Tput", "Latitude", "Longitude"]
    for bitrate in selected_bitrates:
        columns.extend(
            [
                f"{bitrate_to_resolution[bitrate]}_{bitrate}kbps",
                f"MOS_{bitrate}kbps",
            ]
        )
    return columns


def _numeric_mismatch_count(left: pd.Series, right: pd.Series) -> int:
    return int(
        (~np.isclose(
            left.to_numpy(),
            right.to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )).sum()
    )


def validate_raw_content_compatibility(
    frame: pd.DataFrame,
    *,
    repository_root: Path,
    artifact: AuditedSampleArtifact,
    separator: str,
    selected_bitrates: tuple[int, ...],
    bitrate_to_resolution: dict[int, str],
) -> dict[str, Any]:
    """Match every long-form sample row to one trusted raw TSV row."""
    required_raw_columns = _required_raw_columns(
        selected_bitrates,
        bitrate_to_resolution,
    )
    missing_header = sorted(set(required_raw_columns) - set(artifact.raw_tsv_header))
    if missing_header:
        raise SampleValidationError(
            "Trusted raw TSV structure lacks required columns: "
            + ", ".join(missing_header)
        )

    mismatch_counts = {
        "throughput_mbps": 0,
        "latitude": 0,
        "longitude": 0,
        "plr_percent": 0,
        "current_mos": 0,
    }
    checked_rows = 0
    raw_duplicate_key_rows = 0
    missing_source_rows = 0

    for raw_artifact in artifact.raw_tsv_artifacts:
        relative_path = raw_artifact.relative_path
        metadata = parse_filename_metadata(Path(relative_path))
        scenario = frame[
            frame["base_station"].eq(metadata["base_station"])
            & frame["mobility"].eq(metadata["mobility"])
            & frame["prb"].eq(metadata["prb"])
        ].reset_index(drop=True)
        if scenario.empty:
            raise SampleValidationError(
                "The sample has no rows for trusted scenario "
                f"{Path(relative_path).name}."
            )

        raw_path = repository_root / relative_path
        try:
            byte_size = raw_path.stat().st_size
        except OSError as exc:
            raise SampleValidationError(
                f"Cannot inspect trusted TSV {raw_path}: {exc}"
            ) from exc
        if byte_size != raw_artifact.byte_size:
            raise SampleValidationError(
                f"Trusted TSV byte size mismatch: {raw_path}."
            )
        if sha256_file(raw_path) != raw_artifact.sha256:
            raise SampleValidationError(
                f"Trusted TSV SHA-256 mismatch: {raw_path}."
            )
        try:
            raw = pd.read_csv(
                raw_path,
                sep=separator,
                usecols=required_raw_columns,
            )
        except (OSError, ValueError) as exc:
            raise SampleValidationError(
                f"Cannot read trusted TSV {raw_path}: {exc}"
            ) from exc
        raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="raise", utc=True)
        duplicate_keys = int(raw.duplicated(["User_ID", "Timestamp"], keep=False).sum())
        raw_duplicate_key_rows += duplicate_keys
        if duplicate_keys:
            raise SampleValidationError(
                f"Trusted TSV has ambiguous row identities: {raw_path}."
            )

        raw = raw.set_index(["User_ID", "Timestamp"], verify_integrity=True)
        scenario_keys = pd.MultiIndex.from_frame(
            scenario[["user_id", "timestamp"]],
            names=["User_ID", "Timestamp"],
        )
        positions = raw.index.get_indexer(scenario_keys)
        missing = int((positions < 0).sum())
        missing_source_rows += missing
        if missing:
            raise SampleValidationError(
                f"Sample rows are absent from trusted TSV {raw_path}."
            )
        aligned = raw.iloc[positions].reset_index(drop=True)

        mismatch_counts["throughput_mbps"] += _numeric_mismatch_count(
            scenario["throughput_mbps"], aligned["Tput"]
        )
        mismatch_counts["latitude"] += _numeric_mismatch_count(
            scenario["latitude"], aligned["Latitude"]
        )
        mismatch_counts["longitude"] += _numeric_mismatch_count(
            scenario["longitude"], aligned["Longitude"]
        )
        for bitrate in selected_bitrates:
            mask = scenario["bitrate_kbps"].eq(bitrate)
            resolution = bitrate_to_resolution[bitrate]
            mismatch_counts["plr_percent"] += _numeric_mismatch_count(
                scenario.loc[mask, "plr_percent"].reset_index(drop=True),
                aligned.loc[mask, f"{resolution}_{bitrate}kbps"].reset_index(
                    drop=True
                ),
            )
            mismatch_counts["current_mos"] += _numeric_mismatch_count(
                scenario.loc[mask, "current_mos"].reset_index(drop=True),
                aligned.loc[mask, f"MOS_{bitrate}kbps"].reset_index(drop=True),
            )
        checked_rows += len(scenario)

    if checked_rows != len(frame):
        raise SampleValidationError("Raw compatibility did not cover every sample row.")
    if missing_source_rows or any(mismatch_counts.values()):
        raise SampleValidationError("Sample values disagree with the trusted raw TSVs.")

    return {
        "checked_rows": checked_rows,
        "missing_source_rows": missing_source_rows,
        "raw_duplicate_key_rows": raw_duplicate_key_rows,
        "raw_artifact_hashes_verified": len(artifact.raw_tsv_artifacts),
        "scenario_files_checked": len(artifact.raw_tsv_artifacts),
        "value_mismatches": mismatch_counts,
    }
