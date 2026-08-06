"""Synthetic contracts for the bounded sample-validation path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import qoe_twin.sample_validation as sample_validation
import scripts.build_features as final_feature_build
from qoe_twin.artifact_lineage import sha256_file
from qoe_twin.config import canonical_sha256, load_config_directory
from qoe_twin.sample_validation import (
    EXPECTED_SAMPLE_COLUMNS,
    AuditedSampleArtifact,
    RawTsvArtifact,
    SampleValidationError,
    parquet_metadata,
    validate_and_rebuild_sample,
    validate_audited_sample_artifact,
    validate_raw_content_compatibility,
)
from qoe_twin.target import add_future_target, add_sessions

BITRATES = (2000, 4000)
RESOLUTIONS = {2000: "720p", 4000: "1080p"}
RAW_RELATIVE_PATH = (
    "data/external/5G-QoERA/5G-QoERA/MOS_BS_1_Driver_PRB_10.tsv"
)


class _StopAfterArtifactValidation(Exception):
    """Stop the build entry point at its mandatory first artifact check."""


def _base_rows() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=4,
        freq="10s",
    )
    rows: list[dict[str, object]] = []
    mos_values = [4.0, 2.999, 3.0, 3.001]
    for bitrate in BITRATES:
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "user_id": 7,
                    "timestamp": timestamp,
                    "throughput_mbps": float(10 + index),
                    "latitude": 48.0 + index / 1000,
                    "longitude": 2.0 + index / 1000,
                    "base_station": 1,
                    "mobility": "Driver",
                    "prb": 10,
                    "bitrate_kbps": bitrate,
                    "resolution": RESOLUTIONS[bitrate],
                    "plr_percent": float(index + bitrate / 1000),
                    "current_mos": mos_values[index],
                }
            )
    return pd.DataFrame(rows)


def _sample_frame() -> pd.DataFrame:
    sessions = add_sessions(_base_rows(), gap_threshold_seconds=30.0)
    sample = add_future_target(
        sessions,
        poor_mos_threshold=3.0,
        horizon_steps=1,
    )
    return sample.loc[:, EXPECTED_SAMPLE_COLUMNS]


def _artifact(
    rows: int,
    *,
    source_path: Path = Path("synthetic.parquet"),
) -> AuditedSampleArtifact:
    return AuditedSampleArtifact(
        source_relative_path="data/processed/quarantine/sample.parquet",
        source_path=source_path,
        source_sha256="a" * 64,
        byte_size=1,
        rows=rows,
        columns=len(EXPECTED_SAMPLE_COLUMNS),
        schema=(),
        schema_sha256="b" * 64,
        provenance_manifest_relative_path="provenance/manifest.json",
        provenance_manifest_sha256="c" * 64,
        provenance_status="quarantine",
        friend_root_commit="d" * 40,
        upstream_repository_commit="e" * 40,
        raw_tsv_header=tuple(_raw_frame().columns),
        raw_tsv_artifacts=(
            RawTsvArtifact(
                relative_path=RAW_RELATIVE_PATH,
                byte_size=0,
                sha256="f" * 64,
            ),
        ),
    )


def _validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    return validate_and_rebuild_sample(
        frame,
        artifact=_artifact(len(frame)),
        gap_threshold_seconds=30.0,
        poor_mos_threshold=3.0,
        horizon_steps=1,
        selected_bitrates=BITRATES,
        bitrate_to_resolution=RESOLUTIONS,
    )


def _raw_frame() -> pd.DataFrame:
    base = _base_rows()
    physical = base[base["bitrate_kbps"].eq(2000)].reset_index(drop=True)
    other = base[base["bitrate_kbps"].eq(4000)].reset_index(drop=True)
    return pd.DataFrame(
        {
            "User_ID": physical["user_id"],
            "Timestamp": physical["timestamp"],
            "Tput": physical["throughput_mbps"],
            "Latitude": physical["latitude"],
            "Longitude": physical["longitude"],
            "720p_2000kbps": physical["plr_percent"],
            "MOS_2000kbps": physical["current_mos"],
            "1080p_4000kbps": other["plr_percent"],
            "MOS_4000kbps": other["current_mos"],
        }
    )


def _artifact_for_raw(raw_path: Path, rows: int) -> AuditedSampleArtifact:
    return replace(
        _artifact(rows),
        raw_tsv_header=tuple(_raw_frame().columns),
        raw_tsv_artifacts=(
            RawTsvArtifact(
                relative_path=RAW_RELATIVE_PATH,
                byte_size=raw_path.stat().st_size,
                sha256=sha256_file(raw_path),
            ),
        ),
    )


def test_rebuild_uses_strict_mos_boundary_and_retains_terminal_na() -> None:
    rebuilt, summary = _validate(_sample_frame())
    one_session = rebuilt[rebuilt["bitrate_kbps"].eq(2000)].reset_index(
        drop=True
    )

    assert one_session["future_mos"].tolist()[:3] == [2.999, 3.0, 3.001]
    assert one_session["future_poor_qoe"].tolist()[:3] == [1, 0, 0]
    assert pd.isna(one_session.loc[3, "future_poor_qoe"])
    assert summary["boundary_value_counts"] == {
        "2.999": 2,
        "3.000": 2,
        "3.001": 2,
    }


def test_duplicate_row_identity_is_rejected() -> None:
    sample = _sample_frame()
    duplicate = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)

    with pytest.raises(SampleValidationError, match="duplicate row identities"):
        _validate(duplicate)


def test_nonchronological_session_is_rejected() -> None:
    sample = _sample_frame()
    first_session = sample["session_id"].eq(sample.loc[0, "session_id"])
    indices = sample.index[first_session].tolist()
    reordered = pd.concat(
        [
            sample.iloc[[indices[1], indices[0]]],
            sample.drop(index=indices[:2]),
        ],
        ignore_index=True,
    )

    with pytest.raises(SampleValidationError, match="strictly increasing"):
        _validate(reordered)


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(SampleValidationError, match="missing, extra or reordered"):
        _validate(_sample_frame().drop(columns="current_mos"))


def test_missing_bitrate_sibling_is_rejected() -> None:
    sample = _sample_frame()
    missing_sibling = sample.drop(index=sample.index[-1]).reset_index(drop=True)

    with pytest.raises(SampleValidationError, match="every selected bitrate"):
        _validate(missing_sibling)


def test_inherited_target_disagreement_is_rejected_before_rebuild_use() -> None:
    sample = _sample_frame()
    sample.loc[0, "future_poor_qoe"] = 0

    with pytest.raises(SampleValidationError, match="targets disagree"):
        _validate(sample)


def test_exact_raw_content_compatibility(tmp_path: Path) -> None:
    raw_path = tmp_path / RAW_RELATIVE_PATH
    raw_path.parent.mkdir(parents=True)
    _raw_frame().to_csv(raw_path, sep="\t", index=False)
    sample = _sample_frame()
    artifact = _artifact_for_raw(raw_path, len(sample))

    summary = validate_raw_content_compatibility(
        sample,
        repository_root=tmp_path,
        artifact=artifact,
        separator="\t",
        selected_bitrates=BITRATES,
        bitrate_to_resolution=RESOLUTIONS,
    )

    assert summary["checked_rows"] == len(sample)
    assert summary["missing_source_rows"] == 0
    assert set(summary["value_mismatches"].values()) == {0}


def test_raw_value_mismatch_is_rejected(tmp_path: Path) -> None:
    raw_path = tmp_path / RAW_RELATIVE_PATH
    raw_path.parent.mkdir(parents=True)
    _raw_frame().to_csv(raw_path, sep="\t", index=False)
    sample = _sample_frame()
    sample.loc[0, "throughput_mbps"] += 1.0

    with pytest.raises(SampleValidationError, match="disagree"):
        validate_raw_content_compatibility(
            sample,
            repository_root=tmp_path,
            artifact=_artifact_for_raw(raw_path, len(sample)),
            separator="\t",
            selected_bitrates=BITRATES,
            bitrate_to_resolution=RESOLUTIONS,
        )


def test_artifact_hash_footer_and_schema_are_validated(tmp_path: Path) -> None:
    source_relative = "data/processed/quarantine/sample.parquet"
    manifest_relative = "provenance/manifest.json"
    source_path = tmp_path / source_relative
    source_path.parent.mkdir(parents=True)
    _sample_frame().to_parquet(source_path, index=False)
    metadata = parquet_metadata(source_path)
    manifest = {
        "artifacts": [
            {
                "byte_size": source_path.stat().st_size,
                "destination_path": source_relative,
                "parquet_metadata": metadata,
                "role": "friend_processed_snapshot",
                "sha256": sha256_file(source_path),
                "status": "quarantine",
            },
            {
                "byte_size": 0,
                "destination_path": RAW_RELATIVE_PATH,
                "role": "upstream_raw_tsv",
                "sha256": "f" * 64,
                "status": "trusted_raw_input",
            },
        ],
        "friend_root_commit": "d" * 40,
        "raw_tsv_contract": {
            "file_count": 1,
            "header": list(_raw_frame().columns),
        },
        "upstream_repository_commit": "e" * 40,
    }
    manifest_path = tmp_path / manifest_relative
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )

    artifact = validate_audited_sample_artifact(
        tmp_path,
        source_relative_path=source_relative,
        expected_source_sha256=sha256_file(source_path),
        provenance_manifest_relative_path=manifest_relative,
        expected_provenance_manifest_sha256=sha256_file(manifest_path),
    )

    assert artifact.rows == 8
    assert artifact.schema_sha256 == canonical_sha256(metadata["schema"])


def test_artifact_byte_tampering_fails_before_parquet_use(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.parquet"
    source_path.write_bytes(b"changed")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SampleValidationError, match="sample SHA-256"):
        validate_audited_sample_artifact(
            tmp_path,
            source_relative_path="sample.parquet",
            expected_source_sha256="0" * 64,
            provenance_manifest_relative_path="manifest.json",
            expected_provenance_manifest_sha256=sha256_file(manifest_path),
        )


def test_raw_size_tampering_fails_before_tsv_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = tmp_path / RAW_RELATIVE_PATH
    raw_path.parent.mkdir(parents=True)
    _raw_frame().to_csv(raw_path, sep="\t", index=False)
    sample = _sample_frame()
    artifact = _artifact_for_raw(raw_path, len(sample))
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    read_csv = Mock(side_effect=AssertionError("tampered TSV must not be parsed"))
    monkeypatch.setattr(sample_validation.pd, "read_csv", read_csv)

    with pytest.raises(SampleValidationError, match="byte size mismatch"):
        validate_raw_content_compatibility(
            sample,
            repository_root=tmp_path,
            artifact=artifact,
            separator="\t",
            selected_bitrates=BITRATES,
            bitrate_to_resolution=RESOLUTIONS,
        )

    read_csv.assert_not_called()


def test_same_size_raw_tampering_fails_hash_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = tmp_path / RAW_RELATIVE_PATH
    raw_path.parent.mkdir(parents=True)
    _raw_frame().to_csv(raw_path, sep="\t", index=False)
    sample = _sample_frame()
    artifact = _artifact_for_raw(raw_path, len(sample))
    changed = bytearray(raw_path.read_bytes())
    changed[-2] ^= 1
    raw_path.write_bytes(changed)
    read_csv = Mock(side_effect=AssertionError("tampered TSV must not be parsed"))
    monkeypatch.setattr(sample_validation.pd, "read_csv", read_csv)

    with pytest.raises(SampleValidationError, match="SHA-256 mismatch"):
        validate_raw_content_compatibility(
            sample,
            repository_root=tmp_path,
            artifact=artifact,
            separator="\t",
            selected_bitrates=BITRATES,
            bitrate_to_resolution=RESOLUTIONS,
        )

    read_csv.assert_not_called()


def test_final_build_validates_artifact_before_reading_parquet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    configuration = load_config_directory(repository_root / "configs")
    artifact_validation = Mock(side_effect=_StopAfterArtifactValidation)
    parquet_read = Mock(
        side_effect=AssertionError("Parquet must not be read before validation")
    )
    monkeypatch.setattr(
        final_feature_build,
        "load_resolved_configuration",
        Mock(return_value=configuration),
    )
    monkeypatch.setattr(
        final_feature_build,
        "validate_audited_sample_artifact",
        artifact_validation,
    )
    monkeypatch.setattr(final_feature_build.pd, "read_parquet", parquet_read)

    with pytest.raises(_StopAfterArtifactValidation):
        final_feature_build.main()

    artifact_validation.assert_called_once()
    parquet_read.assert_not_called()


def test_split_manifest_serialization_is_deterministic_lf(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    final_feature_build._write_manifest(path, {"z": 2, "a": 1})

    assert path.read_bytes() == b'{\n  "a": 1,\n  "z": 2\n}\n'
