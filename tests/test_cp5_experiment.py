"""Focused tests for the frozen CP5 scientific protocol."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import scripts.calibrate_models_final as final_calibration
import scripts.train_models_final as final_training
from qoe_twin.artifact_lineage import (
    ArtifactLineageError,
    load_selected_final_calibrator,
)
from qoe_twin.config import load_config_directory


class _StopAtPartitionRead(Exception):
    """Stop an entry point at its first permitted partition read."""


def _repository_configuration():
    repository_root = Path(__file__).resolve().parents[1]
    return load_config_directory(repository_root / "configs")


def test_final_model_builders_receive_frozen_yaml_parameters() -> None:
    configuration = _repository_configuration()

    persistence, logistic, forest = final_training.build_configured_models(
        configuration,
        network_features=["network_feature"],
        cross_numeric_features=["numeric_feature"],
        categorical_features=["resolution"],
    )

    logistic_classifier = logistic.named_steps["classifier"]
    forest_classifier = forest.named_steps["classifier"]
    assert persistence.poor_mos_threshold == 3.0
    assert logistic_classifier.random_state == 42
    assert logistic_classifier.max_iter == 2_000
    assert logistic_classifier.class_weight == "balanced"
    assert forest_classifier.random_state == 42
    assert forest_classifier.n_estimators == 100
    assert forest_classifier.max_depth == 14
    assert forest_classifier.min_samples_leaf == 20
    assert forest_classifier.max_samples == 0.5
    assert forest_classifier.n_jobs == 4
    assert forest_classifier.class_weight == "balanced_subsample"


def test_final_training_requests_only_train_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_read = Mock(side_effect=_StopAtPartitionRead)
    monkeypatch.setattr(
        final_training,
        "load_resolved_configuration",
        Mock(return_value=_repository_configuration()),
    )
    monkeypatch.setattr(
        final_training,
        "validate_training_dataset_manifest",
        Mock(
            return_value={
                "columns": 1,
                "dataset_path": "synthetic.parquet",
                "dataset_sha256": "a" * 64,
                "manifest_path": "synthetic-manifest.json",
                "manifest_sha256": "b" * 64,
                "rows": 1,
                "schema_sha256": "c" * 64,
            }
        ),
    )
    monkeypatch.setattr(final_training.pd, "read_parquet", parquet_read)

    with pytest.raises(_StopAtPartitionRead):
        final_training.main()

    assert parquet_read.call_args.kwargs["filters"] == [
        ("split", "==", "train")
    ]


def test_final_calibration_requests_only_validation_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_read = Mock(side_effect=_StopAtPartitionRead)
    joblib_load = Mock(
        side_effect=AssertionError("joblib.load must not be called")
    )
    monkeypatch.setattr(
        final_calibration,
        "required_final_random_forest_artifact_path",
        Mock(return_value=tmp_path / "synthetic-final-rf.joblib"),
    )
    monkeypatch.setattr(
        final_calibration,
        "load_resolved_configuration",
        Mock(return_value=_repository_configuration()),
    )
    monkeypatch.setattr(
        final_calibration,
        "validate_model_lineage",
        Mock(return_value=Mock(artifact_sha256="a" * 64)),
    )
    monkeypatch.setattr(final_calibration.pd, "read_parquet", parquet_read)
    monkeypatch.setattr(final_calibration.joblib, "load", joblib_load)

    with pytest.raises(_StopAtPartitionRead):
        final_calibration.main()

    assert parquet_read.call_args.kwargs["filters"] == [
        ("split", "==", "validation")
    ]
    joblib_load.assert_not_called()


def test_selected_calibration_is_bound_to_resolved_yaml_hash(
    tmp_path: Path,
) -> None:
    configuration = _repository_configuration()
    selection_path = tmp_path / "selected.json"
    selection_path.write_text(
        json.dumps(
            {
                "configuration_sha256": "0" * 64,
                "cross_layer_random_forest": {
                    "calibration_method": "sigmoid"
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactLineageError, match="does not match"):
        load_selected_final_calibrator(
            selection_path,
            tmp_path,
            configuration=configuration,
        )
