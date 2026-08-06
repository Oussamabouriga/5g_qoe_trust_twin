"""Synthetic tests for the training-dataset preflight."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import scripts.train_models as final_training
from qoe_twin.artifact_lineage import sha256_file
from qoe_twin.config import LoadedConfiguration, canonical_sha256
from qoe_twin.sample_validation import parquet_metadata

CONFIGURATION_SHA256 = "a" * 64
DATA_RELATIVE_PATH = Path("data/processed/synthetic_features.parquet")
MANIFEST_RELATIVE_PATH = Path("manifests/synthetic_split_manifest.json")


class _FakeEstimator:
    def fit(self, features: pd.DataFrame, target: object) -> _FakeEstimator:
        del features, target
        return self


def _configuration() -> LoadedConfiguration:
    values = {
        "data": {
            "sample": {
                "corrected_feature_file": DATA_RELATIVE_PATH.as_posix(),
                "split_manifest_file": MANIFEST_RELATIVE_PATH.as_posix(),
            },
            "target": {
                "horizon_steps": 1,
                "poor_mos_threshold": 3.0,
            },
        },
        "model": {
            "experiment": {"random_seed": 42},
            "models": {
                "logistic_regression": {
                    "class_weight": "balanced",
                    "enabled": True,
                    "max_iter": 100,
                },
                "random_forest": {
                    "class_weight": "balanced",
                    "enabled": True,
                    "max_depth": 14,
                    "max_samples": 0.5,
                    "min_samples_leaf": 20,
                    "n_estimators": 100,
                    "n_jobs": 4,
                },
            },
            "target": {"name": "future_poor_qoe"},
        },
    }
    return LoadedConfiguration(
        values=values,
        canonical_json="{}",
        sha256=CONFIGURATION_SHA256,
        unresolved_paths=(),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "train"],
            "future_poor_qoe": [0, 1],
            "network_feature": [1.0, 2.0],
            "cross_feature": [3.0, 4.0],
            "resolution": ["720p", "1080p"],
        }
    )


def _write_contract(
    repository_root: Path,
) -> tuple[LoadedConfiguration, Path, Path, dict[str, object]]:
    configuration = _configuration()
    dataset_path = repository_root / DATA_RELATIVE_PATH
    dataset_path.parent.mkdir(parents=True)
    _frame().to_parquet(dataset_path, index=False)
    footer = parquet_metadata(dataset_path)
    manifest: dict[str, object] = {
        "artifact_kind": "corrected_causal_feature_dataset",
        "configuration_sha256": configuration.sha256,
        "output": {
            "columns": footer["columns"],
            "path": DATA_RELATIVE_PATH.as_posix(),
            "rows": footer["rows"],
            "schema_sha256": canonical_sha256(footer["schema"]),
            "sha256": sha256_file(dataset_path),
        },
    }
    manifest_path = repository_root / MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return configuration, dataset_path, manifest_path, manifest


def _rewrite_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_valid_manifest_authenticates_exact_dataset_and_footer(
    tmp_path: Path,
) -> None:
    configuration, dataset_path, manifest_path, _ = _write_contract(tmp_path)

    identity = final_training.validate_training_dataset_manifest(
        configuration,
        data_path=DATA_RELATIVE_PATH,
        repository_root=tmp_path,
    )

    footer = parquet_metadata(dataset_path)
    assert identity == {
        "columns": footer["columns"],
        "dataset_path": str(DATA_RELATIVE_PATH),
        "dataset_sha256": sha256_file(dataset_path),
        "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": footer["rows"],
        "schema_sha256": canonical_sha256(footer["schema"]),
    }


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("artifact_kind", "artifact_kind"),
        ("configuration_sha256", "configuration SHA-256"),
        ("path", "output.path"),
        ("sha256", "dataset SHA-256"),
        ("rows", "output.rows"),
        ("columns", "output.columns"),
        ("schema_sha256", "schema SHA-256"),
    ],
)
def test_manifest_identity_or_footer_mismatch_is_rejected(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    configuration, _, manifest_path, manifest = _write_contract(tmp_path)
    output = manifest["output"]
    assert isinstance(output, dict)
    if case == "artifact_kind":
        manifest[case] = "uncorrected_dataset"
    elif case == "configuration_sha256":
        manifest[case] = "b" * 64
    elif case == "path":
        output[case] = "data/processed/other.parquet"
    elif case in {"sha256", "schema_sha256"}:
        output[case] = "b" * 64
    else:
        output[case] = int(output[case]) + 1
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(
        final_training.TrainingDatasetManifestError,
        match=message,
    ):
        final_training.validate_training_dataset_manifest(
            configuration,
            data_path=DATA_RELATIVE_PATH,
            repository_root=tmp_path,
        )


def test_missing_or_non_json_manifest_is_rejected(tmp_path: Path) -> None:
    configuration = _configuration()

    with pytest.raises(
        final_training.TrainingDatasetManifestError,
        match="manifest is missing",
    ):
        final_training.validate_training_dataset_manifest(
            configuration,
            data_path=DATA_RELATIVE_PATH,
            repository_root=tmp_path,
        )

    manifest_path = tmp_path / MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(
        final_training.TrainingDatasetManifestError,
        match="not readable JSON",
    ):
        final_training.validate_training_dataset_manifest(
            configuration,
            data_path=DATA_RELATIVE_PATH,
            repository_root=tmp_path,
        )


def test_main_rejects_bad_manifest_before_reading_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration, _, manifest_path, manifest = _write_contract(tmp_path)
    manifest["artifact_kind"] = "wrong"
    _rewrite_manifest(manifest_path, manifest)
    parquet_read = Mock(
        side_effect=AssertionError("pd.read_parquet must not run")
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(final_training, "DATA_PATH", DATA_RELATIVE_PATH)
    monkeypatch.setattr(
        final_training,
        "load_resolved_configuration",
        Mock(return_value=configuration),
    )
    monkeypatch.setattr(final_training.pd, "read_parquet", parquet_read)

    with pytest.raises(
        final_training.TrainingDatasetManifestError,
        match="artifact_kind",
    ):
        final_training.main()

    parquet_read.assert_not_called()


def test_training_metadata_records_validated_dataset_and_manifest_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration, _, _, _ = _write_contract(tmp_path)
    model_directory = Path("models/uncalibrated")
    result_directory = Path("results/metrics")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(final_training, "DATA_PATH", DATA_RELATIVE_PATH)
    monkeypatch.setattr(final_training, "MODEL_DIRECTORY", model_directory)
    monkeypatch.setattr(final_training, "RESULT_DIRECTORY", result_directory)
    monkeypatch.setattr(
        final_training,
        "load_resolved_configuration",
        Mock(return_value=configuration),
    )
    monkeypatch.setattr(
        final_training,
        "clean_feature_lists",
        Mock(
            return_value=(
                ["network_feature"],
                ["cross_feature"],
                ["resolution"],
            )
        ),
    )
    monkeypatch.setattr(
        final_training,
        "build_configured_models",
        Mock(return_value=(object(), _FakeEstimator(), _FakeEstimator())),
    )
    monkeypatch.setattr(final_training.joblib, "dump", Mock())
    monkeypatch.setattr(
        final_training,
        "save_final_random_forest",
        Mock(
            return_value=(
                model_directory / "cross_layer_random_forest_final.joblib"
            )
        ),
    )
    monkeypatch.setattr(final_training, "write_model_lineage", Mock())

    final_training.main()

    metadata = json.loads(
        (tmp_path / result_directory / "final_training_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    identity = metadata["dataset_identity"]
    assert identity["dataset_sha256"] == sha256_file(
        tmp_path / DATA_RELATIVE_PATH
    )
    assert identity["manifest_sha256"] == sha256_file(
        tmp_path / MANIFEST_RELATIVE_PATH
    )
