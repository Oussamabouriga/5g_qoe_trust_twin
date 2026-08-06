"""Synthetic behavioral tests for final-artifact lineage contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import scripts.apply_trust as final_inference
import scripts.calibrate_models as final_calibration
import scripts.final_evaluation as final_evaluation
import scripts.train_models as final_training
from qoe_twin.artifact_lineage import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactLineageError,
    feature_list_sha256,
    require_feature_columns,
    resolved_configuration_sha256,
    sha256_file,
    sidecar_path,
    validate_calibrator_lineage,
    validate_model_lineage,
    validate_prediction_chain,
    validate_prediction_lineage,
    write_calibrator_lineage,
    write_model_lineage,
    write_prediction_lineage,
)
from qoe_twin.config import (
    LoadedConfiguration,
    canonical_json,
)
from qoe_twin.features import get_cross_layer_feature_names

NUMERIC_FEATURES = ("synthetic_numeric_a", "synthetic_numeric_b")
CATEGORICAL_FEATURES = ("synthetic_category",)


class _StopAfterSelection(Exception):
    """Stop a final entry point immediately after observing its chosen path."""


def _configuration(
    identity: str = "configuration-a",
    *,
    unresolved_paths: tuple[str, ...] = (),
) -> LoadedConfiguration:
    values = {
        "synthetic": {"identity": identity},
        "trust": {
            "trust": {
                "confidence_weight": 0.4,
                "validation_performance_weight": 0.3,
                "calibration_weight": 0.2,
                "data_quality_weight": 0.1,
            },
            "levels": {"high": 0.8, "medium": 0.55},
            "abstention": {"enabled": True, "minimum_coverage": 0.9},
        },
    }
    canonical = canonical_json(values)
    return LoadedConfiguration(
        values=values,
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        unresolved_paths=unresolved_paths,
    )


def _write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_selected_configuration(
    path: Path,
    method: str,
    configuration: LoadedConfiguration | None = None,
    *,
    include_abstention_threshold: bool = True,
) -> Path:
    effective_configuration = configuration or _configuration()
    selected = {
        "calibration_method": method,
        "decision_threshold": 0.4,
        "f1": 0.75,
        "expected_calibration_error": 0.05,
    }
    if include_abstention_threshold:
        selected["abstention_threshold"] = 0.6
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "configuration_sha256": resolved_configuration_sha256(
                    effective_configuration
                ),
                "cross_layer_random_forest": selected,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_chain(
    directory: Path,
    configuration: LoadedConfiguration,
    *,
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> tuple[
    Path,
    Path,
    Path,
    ArtifactLineage,
    ArtifactLineage,
    ArtifactLineage,
]:
    model_path = _write_bytes(directory / "final-model.joblib", b"model-v1")
    model_lineage = write_model_lineage(
        model_path,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    calibrator_path = _write_bytes(
        directory / "cross_layer_random_forest_isotonic_final.joblib",
        b"calibrator-v1",
    )
    calibrator_lineage = write_calibrator_lineage(
        calibrator_path,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        parent_model_sha256=model_lineage.artifact_sha256,
    )

    prediction_path = _write_bytes(
        directory / "final-predictions.parquet",
        b"predictions-v1",
    )
    prediction_lineage = write_prediction_lineage(
        prediction_path,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        parent_model_sha256=model_lineage.artifact_sha256,
        parent_calibrator_sha256=calibrator_lineage.artifact_sha256,
    )

    return (
        model_path,
        calibrator_path,
        prediction_path,
        model_lineage,
        calibrator_lineage,
        prediction_lineage,
    )


def test_exact_model_calibrator_prediction_chain_passes(tmp_path: Path) -> None:
    configuration = _configuration()
    (
        model_path,
        calibrator_path,
        prediction_path,
        written_model,
        written_calibrator,
        written_prediction,
    ) = _write_chain(tmp_path, configuration)

    validated_model = validate_model_lineage(
        model_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    validated_calibrator = validate_calibrator_lineage(
        calibrator_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        expected_parent_model_sha256=validated_model.artifact_sha256,
    )
    validated_prediction = validate_prediction_lineage(
        prediction_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        expected_parent_model_sha256=validated_model.artifact_sha256,
        expected_parent_calibrator_sha256=(
            validated_calibrator.artifact_sha256
        ),
    )

    validate_prediction_chain(
        model_path=model_path,
        calibrator_path=calibrator_path,
        prediction_path=prediction_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    assert validated_model == written_model
    assert validated_calibrator == written_calibrator
    assert validated_prediction == written_prediction
    assert validated_model.artifact_kind is ArtifactKind.FINAL_RANDOM_FOREST
    assert validated_calibrator.artifact_kind is ArtifactKind.FINAL_CALIBRATOR
    assert validated_prediction.artifact_kind is ArtifactKind.FINAL_PREDICTIONS
    assert isinstance(validated_prediction, ArtifactLineage)
    assert validated_model.artifact_sha256 == sha256_file(model_path)
    assert validated_model.feature_list_sha256 == feature_list_sha256(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )

    common_keys = {
        "schema_version",
        "artifact_kind",
        "artifact_filename",
        "artifact_sha256",
        "configuration_sha256",
        "numeric_features",
        "categorical_features",
        "feature_list_sha256",
    }
    model_document = json.loads(sidecar_path(model_path).read_text("utf-8"))
    calibrator_document = json.loads(
        sidecar_path(calibrator_path).read_text("utf-8")
    )
    prediction_document = json.loads(
        sidecar_path(prediction_path).read_text("utf-8")
    )
    assert set(model_document) == common_keys
    assert set(calibrator_document) == common_keys | {"parent_model_sha256"}
    assert set(prediction_document) == common_keys | {
        "parent_model_sha256",
        "parent_calibrator_sha256",
    }


def test_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    configuration = _configuration()
    model_path = _write_bytes(tmp_path / "final-model.joblib", b"model")

    with pytest.raises(ArtifactLineageError, match="sidecar"):
        validate_model_lineage(
            model_path,
            configuration=configuration,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )


def test_changed_artifact_bytes_fail_validation(tmp_path: Path) -> None:
    configuration = _configuration()
    model_path = _write_bytes(tmp_path / "final-model.joblib", b"model")
    write_model_lineage(
        model_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    model_path.write_bytes(b"changed model bytes")

    with pytest.raises(ArtifactLineageError, match="SHA-256"):
        validate_model_lineage(
            model_path,
            configuration=configuration,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )


def test_calibrator_from_another_model_fails(tmp_path: Path) -> None:
    configuration = _configuration()
    model_a_path = _write_bytes(tmp_path / "model-a.joblib", b"model-a")
    model_a = write_model_lineage(
        model_a_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    model_b_path = _write_bytes(tmp_path / "model-b.joblib", b"model-b")
    model_b = write_model_lineage(
        model_b_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    calibrator_path = _write_bytes(tmp_path / "calibrator.joblib", b"calibrator")
    write_calibrator_lineage(
        calibrator_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        parent_model_sha256=model_b.artifact_sha256,
    )

    assert model_a.artifact_sha256 != model_b.artifact_sha256
    with pytest.raises(ArtifactLineageError, match=r"parent[- ]model"):
        validate_calibrator_lineage(
            calibrator_path,
            configuration=configuration,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            expected_parent_model_sha256=model_a.artifact_sha256,
        )


def test_reordered_features_fail_validation(tmp_path: Path) -> None:
    configuration = _configuration()
    model_path = _write_bytes(tmp_path / "final-model.joblib", b"model")
    write_model_lineage(
        model_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    with pytest.raises(ArtifactLineageError, match="feature"):
        validate_model_lineage(
            model_path,
            configuration=configuration,
            numeric_features=tuple(reversed(NUMERIC_FEATURES)),
            categorical_features=CATEGORICAL_FEATURES,
        )


def test_configuration_hash_mismatch_fails(tmp_path: Path) -> None:
    original_configuration = _configuration("configuration-a")
    different_configuration = _configuration("configuration-b")
    model_path = _write_bytes(tmp_path / "final-model.joblib", b"model")
    write_model_lineage(
        model_path,
        configuration=original_configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    with pytest.raises(ArtifactLineageError, match="configuration"):
        validate_model_lineage(
            model_path,
            configuration=different_configuration,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )


def test_tampered_predictions_fail_chain_validation(tmp_path: Path) -> None:
    configuration = _configuration()
    model_path, calibrator_path, prediction_path, *_ = _write_chain(
        tmp_path,
        configuration,
    )
    prediction_path.write_bytes(b"tampered predictions")

    with pytest.raises(ArtifactLineageError, match="SHA-256"):
        validate_prediction_chain(
            model_path=model_path,
            calibrator_path=calibrator_path,
            prediction_path=prediction_path,
            configuration=configuration,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )


def test_required_feature_checker_rejects_missing_column() -> None:
    with pytest.raises(ArtifactLineageError, match="missing_numeric"):
        require_feature_columns(
            ["available_numeric", "available_category"],
            ["available_numeric", "missing_numeric", "available_category"],
            context="synthetic final model",
        )


@pytest.mark.parametrize(
    "feature_selector",
    [
        final_training.clean_feature_lists,
        final_calibration.get_feature_lists,
    ],
    ids=["final-training", "final-calibration"],
)
def test_final_feature_helpers_reject_instead_of_dropping(
    feature_selector: object,
) -> None:
    missing_feature = "capacity_insufficient"
    available_features = [
        feature
        for feature in get_cross_layer_feature_names()
        if feature != missing_feature
    ]
    frame = pd.DataFrame(columns=available_features)

    with pytest.raises(ArtifactLineageError, match=missing_feature):
        feature_selector(frame)  # type: ignore[operator]


def test_unresolved_configuration_hash_is_rejected() -> None:
    configuration = _configuration(
        unresolved_paths=("model.target.horizon_steps",),
    )

    with pytest.raises(ArtifactLineageError, match="unresolved"):
        resolved_configuration_sha256(configuration)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_final_calibration_writes_lineage_for_each_rf_calibrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    configuration = _configuration()
    parent_model_sha256 = "a" * 64
    monkeypatch.setattr(
        final_calibration,
        "CALIBRATED_DIRECTORY",
        tmp_path,
    )

    calibrator_path = (
        final_calibration.save_final_random_forest_calibrator(
            {"synthetic_method": method},  # type: ignore[arg-type]
            calibration_method=method,
            configuration=configuration,
            numeric_features=list(NUMERIC_FEATURES),
            categorical_features=list(CATEGORICAL_FEATURES),
            parent_model_sha256=parent_model_sha256,
        )
    )

    assert calibrator_path == (
        tmp_path
        / f"cross_layer_random_forest_{method}_final.joblib"
    )
    validated = validate_calibrator_lineage(
        calibrator_path,
        configuration=configuration,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        expected_parent_model_sha256=parent_model_sha256,
    )
    assert validated.parent_model_sha256 == parent_model_sha256


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_final_inference_routes_to_selected_calibrator_before_data_or_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    selection_path = _write_selected_configuration(
        tmp_path / "selected.json",
        method,
    )
    configuration = _configuration()
    model_lineage = Mock(artifact_sha256="a" * 64)
    calibrator_validation = Mock(side_effect=_StopAfterSelection)
    parquet_read = Mock(
        side_effect=AssertionError("pd.read_parquet must not be called")
    )
    joblib_load = Mock(
        side_effect=AssertionError("joblib.load must not be called")
    )

    monkeypatch.setattr(final_inference, "CONFIGURATION_PATH", selection_path)
    monkeypatch.setattr(final_inference, "CALIBRATED_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        final_inference,
        "load_resolved_configuration",
        Mock(return_value=configuration),
    )
    monkeypatch.setattr(
        final_inference,
        "validate_model_lineage",
        Mock(return_value=model_lineage),
    )
    monkeypatch.setattr(
        final_inference,
        "validate_calibrator_lineage",
        calibrator_validation,
    )
    monkeypatch.setattr(final_inference.pd, "read_parquet", parquet_read)
    monkeypatch.setattr(final_inference.joblib, "load", joblib_load)

    with pytest.raises(_StopAfterSelection):
        final_inference.main()

    assert calibrator_validation.call_args.args[0] == (
        tmp_path
        / f"cross_layer_random_forest_{method}_final.joblib"
    )
    parquet_read.assert_not_called()
    joblib_load.assert_not_called()


def test_inference_requires_selected_abstention_threshold_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = _write_selected_configuration(
        tmp_path / "selected.json",
        "sigmoid",
        include_abstention_threshold=False,
    )
    model_validation = Mock(
        side_effect=AssertionError("model lineage must not be read")
    )
    parquet_read = Mock(
        side_effect=AssertionError("pd.read_parquet must not be called")
    )
    joblib_load = Mock(
        side_effect=AssertionError("joblib.load must not be called")
    )

    monkeypatch.setattr(final_inference, "CONFIGURATION_PATH", selection_path)
    monkeypatch.setattr(
        final_inference,
        "load_resolved_configuration",
        Mock(return_value=_configuration()),
    )
    monkeypatch.setattr(
        final_inference,
        "validate_model_lineage",
        model_validation,
    )
    monkeypatch.setattr(final_inference.pd, "read_parquet", parquet_read)
    monkeypatch.setattr(final_inference.joblib, "load", joblib_load)

    with pytest.raises(ValueError, match="validation-selected"):
        final_inference.main()

    model_validation.assert_not_called()
    parquet_read.assert_not_called()
    joblib_load.assert_not_called()


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_final_evaluation_routes_to_selected_calibrator_before_data_or_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    selection_path = _write_selected_configuration(
        tmp_path / "selected.json",
        method,
    )
    chain_validation = Mock(side_effect=_StopAfterSelection)
    parquet_read = Mock(
        side_effect=AssertionError("pd.read_parquet must not be called")
    )
    joblib_load = Mock(
        side_effect=AssertionError("joblib.load must not be called")
    )

    monkeypatch.setattr(final_evaluation, "CONFIGURATION_PATH", selection_path)
    monkeypatch.setattr(final_evaluation, "CALIBRATED_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        final_evaluation,
        "load_resolved_configuration",
        Mock(return_value=_configuration()),
    )
    monkeypatch.setattr(
        final_evaluation,
        "validate_prediction_chain",
        chain_validation,
    )
    monkeypatch.setattr(final_evaluation.pd, "read_parquet", parquet_read)
    monkeypatch.setattr(final_evaluation.joblib, "load", joblib_load)

    with pytest.raises(_StopAfterSelection):
        final_evaluation.main()

    assert chain_validation.call_args.kwargs["calibrator_path"] == (
        tmp_path
        / f"cross_layer_random_forest_{method}_final.joblib"
    )
    parquet_read.assert_not_called()
    joblib_load.assert_not_called()


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("uncalibrated", "requires a calibrator"),
        ("unknown_method", "Unsupported final calibration_method"),
    ],
)
@pytest.mark.parametrize(
    "consumer",
    [final_inference, final_evaluation],
    ids=["final-inference", "final-evaluation"],
)
def test_final_consumers_reject_unsupported_selection_before_data_or_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: object,
    method: str,
    message: str,
) -> None:
    selection_path = _write_selected_configuration(
        tmp_path / "selected.json",
        method,
    )
    parquet_read = Mock(
        side_effect=AssertionError("pd.read_parquet must not be called")
    )
    joblib_load = Mock(
        side_effect=AssertionError("joblib.load must not be called")
    )

    monkeypatch.setattr(consumer, "CONFIGURATION_PATH", selection_path)
    monkeypatch.setattr(consumer, "CALIBRATED_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        consumer,
        "load_resolved_configuration",
        Mock(return_value=_configuration()),
    )
    monkeypatch.setattr(consumer.pd, "read_parquet", parquet_read)  # type: ignore[attr-defined]
    monkeypatch.setattr(consumer.joblib, "load", joblib_load)  # type: ignore[attr-defined]

    with pytest.raises(ArtifactLineageError, match=message):
        consumer.main()  # type: ignore[attr-defined]

    parquet_read.assert_not_called()
    joblib_load.assert_not_called()


def test_final_calibration_rejects_lineage_before_joblib_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration()
    model_directory = tmp_path / "uncalibrated"
    model_path = _write_bytes(
        model_directory / "cross_layer_random_forest_final.joblib",
        b"model-without-sidecar",
    )
    joblib_load = Mock(
        side_effect=AssertionError("joblib.load must not be called")
    )
    parquet_read = Mock(
        side_effect=AssertionError("pd.read_parquet must not be called")
    )

    monkeypatch.setattr(final_calibration, "MODEL_DIRECTORY", model_directory)
    monkeypatch.setattr(
        final_calibration,
        "load_resolved_configuration",
        Mock(return_value=configuration),
    )
    monkeypatch.setattr(final_calibration.joblib, "load", joblib_load)
    monkeypatch.setattr(final_calibration.pd, "read_parquet", parquet_read)

    with pytest.raises(ArtifactLineageError, match="sidecar"):
        final_calibration.main()

    assert model_path.is_file()
    joblib_load.assert_not_called()
    parquet_read.assert_not_called()


def test_final_inference_rejects_lineage_before_joblib_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration()
    numeric_features, categorical_features = _final_feature_lists()
    model_path = _write_bytes(tmp_path / "final-model.joblib", b"model")
    write_model_lineage(
        model_path,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    _write_bytes(
        tmp_path / "cross_layer_random_forest_isotonic_final.joblib",
        b"calibrator-without-sidecar",
    )
    selection_path = _write_selected_configuration(
        tmp_path / "selected.json",
        "isotonic",
    )
    joblib_load = Mock(
        side_effect=AssertionError("joblib.load must not be called")
    )
    parquet_read = Mock(
        side_effect=AssertionError("pd.read_parquet must not be called")
    )

    monkeypatch.setattr(final_inference, "MODEL_PATH", model_path)
    monkeypatch.setattr(final_inference, "CONFIGURATION_PATH", selection_path)
    monkeypatch.setattr(final_inference, "CALIBRATED_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        final_inference,
        "load_resolved_configuration",
        Mock(return_value=configuration),
    )
    monkeypatch.setattr(final_inference.joblib, "load", joblib_load)
    monkeypatch.setattr(final_inference.pd, "read_parquet", parquet_read)

    with pytest.raises(ArtifactLineageError, match="sidecar"):
        final_inference.main()

    joblib_load.assert_not_called()
    parquet_read.assert_not_called()


def test_final_evaluation_rejects_predictions_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration()
    numeric_features, categorical_features = _final_feature_lists()
    (
        model_path,
        calibrator_path,
        prediction_path,
        *_,
    ) = _write_chain(
        tmp_path,
        configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    prediction_path.write_bytes(b"tampered after sidecar creation")
    joblib_load = Mock(
        side_effect=AssertionError("joblib.load must not be called")
    )
    parquet_read = Mock(
        side_effect=AssertionError("pd.read_parquet must not be called")
    )

    monkeypatch.setattr(final_evaluation, "MODEL_PATH", model_path)
    selection_path = _write_selected_configuration(
        tmp_path / "selected.json",
        "isotonic",
    )
    monkeypatch.setattr(final_evaluation, "CONFIGURATION_PATH", selection_path)
    monkeypatch.setattr(final_evaluation, "CALIBRATED_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        final_evaluation,
        "TRUSTED_PREDICTION_PATH",
        prediction_path,
    )
    monkeypatch.setattr(
        final_evaluation,
        "load_resolved_configuration",
        Mock(return_value=configuration),
    )
    monkeypatch.setattr(final_evaluation.joblib, "load", joblib_load)
    monkeypatch.setattr(final_evaluation.pd, "read_parquet", parquet_read)
    monkeypatch.setattr(final_evaluation, "TABLE_DIRECTORY", tmp_path / "tables")
    monkeypatch.setattr(final_evaluation, "FIGURE_DIRECTORY", tmp_path / "figures")
    monkeypatch.setattr(final_evaluation, "METRIC_DIRECTORY", tmp_path / "metrics")

    with pytest.raises(ArtifactLineageError, match="SHA-256"):
        final_evaluation.main()

    joblib_load.assert_not_called()
    parquet_read.assert_not_called()


def _final_feature_lists() -> tuple[tuple[str, ...], tuple[str, ...]]:
    categorical = ("resolution",)
    numeric = tuple(
        feature
        for feature in get_cross_layer_feature_names()
        if feature not in categorical
    )
    return numeric, categorical
