from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from qoe_twin.config import (
    BLOCKED_PENDING_ARTIFACTS,
    ConfigurationError,
    ConflictStatus,
    canonical_json,
    canonical_sha256,
    detect_conflicts,
    load_config_directory,
)


def _synthetic_documents(*, resolved_model_target: bool = False) -> dict[str, Any]:
    model_mos: float | None = 2.75 if resolved_model_target else None
    model_horizon: int | None = 3 if resolved_model_target else None
    return {
        "data.yaml": {
            "dataset": {
                "name": "synthetic-qoe-fixture",
                "directory": "fixtures/raw",
                "file_pattern": "SYNTHETIC_*.tsv",
                "separator": "\t",
            },
            "video": {
                "selected_bitrates_kbps": [1111, 7777],
                "bitrate_to_resolution": {
                    1111: "320x180",
                    7777: "1920x1080",
                },
            },
            "sessions": {"gap_threshold_seconds": 17.5},
            "target": {"poor_mos_threshold": 2.75, "horizon_steps": 3},
            "processing": {
                "output_file": "fixtures/processed/synthetic.parquet",
                "sort_chronologically": True,
                "preserve_original_files": False,
            },
            "splitting": {
                "method": "synthetic_temporal_split",
                "train_fraction": 0.6,
                "validation_fraction": 0.2,
                "test_fraction": 0.2,
            },
        },
        "model.yaml": {
            "experiment": {"random_seed": 424242},
            "target": {
                "name": "synthetic_future_event",
                "mos_threshold": model_mos,
                "horizon_steps": model_horizon,
            },
            "models": {
                "persistence": {"enabled": True},
                "logistic_regression": {
                    "enabled": True,
                    "class_weight": "synthetic_balanced",
                    "max_iter": 123,
                },
                "random_forest": {
                    "enabled": True,
                    "n_estimators": 17,
                    "max_depth": None,
                    "class_weight": "synthetic_balanced",
                    "n_jobs": 1,
                },
            },
            "calibration": {"methods": ["synthetic_a", "synthetic_b"]},
        },
        "trust.yaml": {
            "trust": {
                "confidence_weight": 0.25,
                "validation_performance_weight": 0.25,
                "calibration_weight": 0.25,
                "data_quality_weight": 0.25,
            },
            "levels": {"high": 0.8, "medium": 0.4},
            "abstention": {"enabled": True, "minimum_trust": 0.33},
        },
        "llm.yaml": {
            "openai": {
                "api_key_environment_variable": "SYNTHETIC_API_KEY",
                "model_environment_variable": "SYNTHETIC_MODEL_NAME",
                "temperature": 0.25,
                "maximum_retries": 2,
                "timeout_seconds": 9.5,
            },
            "explanation": {
                "require_predicted_event": True,
                "require_likely_causes": True,
                "require_confidence": True,
                "require_limitations": True,
                "require_operator_checks": True,
                "require_supporting_fields": True,
            },
            "grounding": {
                "reject_unsupported_numbers": True,
                "reject_unknown_fields": True,
                "reject_unsupported_causal_claims": True,
                "reject_missing_required_sections": True,
            },
        },
    }


def _reverse_mappings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _reverse_mappings(value[key])
            for key in reversed(tuple(value.keys()))
        }
    if isinstance(value, list):
        return [_reverse_mappings(item) for item in value]
    return value


def _write_documents(
    root: Path,
    documents: dict[str, Any] | None = None,
    *,
    reverse_keys: bool = False,
) -> Path:
    config_directory = root / "synthetic-config"
    config_directory.mkdir(parents=True)
    content = documents if documents is not None else _synthetic_documents()
    if reverse_keys:
        content = _reverse_mappings(content)
    for filename, document in content.items():
        (config_directory / filename).write_text(
            yaml.safe_dump(document, sort_keys=False),
            encoding="utf-8",
        )
    return config_directory


def _observation_values(record: Any) -> dict[str, Any]:
    return {
        observation.source: observation.value
        for observation in record.observations
    }


def test_loads_all_four_synthetic_files_and_distinguishes_unresolved_nulls(
    tmp_path: Path,
) -> None:
    config_directory = _write_documents(tmp_path)

    loaded = load_config_directory(config_directory)

    assert tuple(loaded.values) == ("data", "llm", "model", "trust")
    assert loaded.unresolved_paths == (
        "model.target.horizon_steps",
        "model.target.mos_threshold",
    )
    assert (
        loaded.values["model"]["target"]["horizon_steps"]
        == BLOCKED_PENDING_ARTIFACTS
    )
    assert (
        loaded.values["model"]["target"]["mos_threshold"]
        == BLOCKED_PENDING_ARTIFACTS
    )
    assert loaded.values["model"]["models"]["random_forest"]["max_depth"] is None
    assert loaded.canonical_json == canonical_json(loaded.values)
    assert loaded.sha256 == canonical_sha256(loaded.values)
    assert loaded.sha256 == hashlib.sha256(
        loaded.canonical_json.encode("utf-8")
    ).hexdigest()


def test_resolved_synthetic_model_target_removes_all_blocks(tmp_path: Path) -> None:
    documents = _synthetic_documents(resolved_model_target=True)
    loaded = load_config_directory(_write_documents(tmp_path, documents))

    assert loaded.unresolved_paths == ()
    assert loaded.values["model"]["target"] == {
        "horizon_steps": 3,
        "mos_threshold": 2.75,
        "name": "synthetic_future_event",
    }


def test_rejects_a_missing_required_yaml_file(tmp_path: Path) -> None:
    config_directory = _write_documents(tmp_path)
    (config_directory / "llm.yaml").unlink()

    with pytest.raises(ConfigurationError, match="llm.yaml"):
        load_config_directory(config_directory)


def test_rejects_an_unexpected_yaml_file(tmp_path: Path) -> None:
    config_directory = _write_documents(tmp_path)
    (config_directory / "surprise.yaml").write_text(
        "surprise: true\n", encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="surprise.yaml"):
        load_config_directory(config_directory)


def test_rejects_a_missing_nested_key(tmp_path: Path) -> None:
    documents = _synthetic_documents()
    del documents["data.yaml"]["dataset"]["name"]

    with pytest.raises(ConfigurationError, match="dataset.name"):
        load_config_directory(_write_documents(tmp_path, documents))


def test_rejects_an_unknown_nested_key(tmp_path: Path) -> None:
    documents = _synthetic_documents()
    documents["trust.yaml"]["trust"]["implicit_precedence"] = 1.0

    with pytest.raises(ConfigurationError, match="implicit_precedence"):
        load_config_directory(_write_documents(tmp_path, documents))


def test_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    config_directory = _write_documents(tmp_path)
    trust_path = config_directory / "trust.yaml"
    trust_path.write_text(
        """
trust:
  confidence_weight: 0.25
  confidence_weight: 0.25
  validation_performance_weight: 0.25
  calibration_weight: 0.25
  data_quality_weight: 0.25
levels:
  high: 0.8
  medium: 0.4
abstention:
  enabled: true
  minimum_trust: 0.33
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="(?i)duplicate"):
        load_config_directory(config_directory)


@pytest.mark.parametrize(
    ("filename", "path", "invalid_value"),
    [
        ("model.yaml", ("experiment", "random_seed"), True),
        ("data.yaml", ("processing", "sort_chronologically"), 1),
        ("model.yaml", ("models", "random_forest", "n_estimators"), 4.5),
    ],
)
def test_rejects_values_with_the_wrong_strict_type(
    tmp_path: Path,
    filename: str,
    path: tuple[str, ...],
    invalid_value: Any,
) -> None:
    documents = _synthetic_documents()
    target = documents[filename]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value

    with pytest.raises(ConfigurationError):
        load_config_directory(_write_documents(tmp_path, documents))


@pytest.mark.parametrize(
    ("filename", "path", "invalid_value"),
    [
        ("data.yaml", ("target", "poor_mos_threshold"), 5.01),
        ("model.yaml", ("models", "random_forest", "n_jobs"), 0),
        ("trust.yaml", ("levels", "high"), 1.01),
        ("llm.yaml", ("openai", "temperature"), 2.01),
        ("llm.yaml", ("openai", "maximum_retries"), -1),
    ],
)
def test_rejects_out_of_range_values(
    tmp_path: Path,
    filename: str,
    path: tuple[str, ...],
    invalid_value: Any,
) -> None:
    documents = _synthetic_documents()
    target = documents[filename]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value

    with pytest.raises(ConfigurationError):
        load_config_directory(_write_documents(tmp_path, documents))


@pytest.mark.parametrize("case", ["split_sum", "bitrates", "weights", "levels"])
def test_rejects_cross_field_inconsistencies(tmp_path: Path, case: str) -> None:
    documents = _synthetic_documents()
    if case == "split_sum":
        documents["data.yaml"]["splitting"]["train_fraction"] = 0.5
    elif case == "bitrates":
        documents["data.yaml"]["video"]["selected_bitrates_kbps"] = [1111]
    elif case == "weights":
        documents["trust.yaml"]["trust"]["confidence_weight"] = 0.1
    else:
        documents["trust.yaml"]["levels"] = {"high": 0.4, "medium": 0.4}

    with pytest.raises(ConfigurationError):
        load_config_directory(_write_documents(tmp_path, documents))


@pytest.mark.parametrize(
    ("filename", "path", "invalid_value"),
    [
        ("data.yaml", ("dataset", "name"), ""),
        ("data.yaml", ("dataset", "directory"), "../outside"),
        ("data.yaml", ("dataset", "file_pattern"), "../outside/*.tsv"),
        ("data.yaml", ("dataset", "file_pattern"), "C:\\outside\\*.tsv"),
        ("data.yaml", ("processing", "output_file"), "/absolute/output.parquet"),
        ("llm.yaml", ("openai", "model_environment_variable"), "bad-name"),
    ],
)
def test_rejects_invalid_strings_and_paths(
    tmp_path: Path,
    filename: str,
    path: tuple[str, ...],
    invalid_value: str,
) -> None:
    documents = _synthetic_documents()
    target = documents[filename]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value

    with pytest.raises(ConfigurationError):
        load_config_directory(_write_documents(tmp_path, documents))


def test_canonical_json_and_hash_ignore_yaml_key_order(tmp_path: Path) -> None:
    first = load_config_directory(_write_documents(tmp_path / "first"))
    second = load_config_directory(
        _write_documents(tmp_path / "second", reverse_keys=True)
    )

    assert first.values == second.values
    assert first.canonical_json == second.canonical_json
    assert first.sha256 == second.sha256
    assert first.canonical_json == canonical_json(first.values)
    assert first.sha256 == canonical_sha256(first.values)


def test_loaded_values_are_deeply_immutable_and_hash_stable(tmp_path: Path) -> None:
    loaded = load_config_directory(_write_documents(tmp_path))
    original_json = loaded.canonical_json
    original_hash = loaded.sha256

    with pytest.raises(TypeError):
        loaded.values["data"]["dataset"]["name"] = "mutated"
    with pytest.raises(TypeError):
        loaded.values["data"]["video"]["selected_bitrates_kbps"][0] = 9999
    with pytest.raises(TypeError):
        loaded.values["data"]["video"]["bitrate_to_resolution"][1111] = "1x1"
    with pytest.raises(AttributeError):
        loaded.sha256 = "mutated"

    assert loaded.canonical_json == original_json
    assert loaded.sha256 == original_hash
    assert canonical_json(loaded.values) == original_json
    assert canonical_sha256(loaded.values) == original_hash


def test_detect_conflicts_reports_each_status_without_selecting_a_value() -> None:
    observations = {
        "synthetic.z_conflict": {"yaml": 11, "code": 22, "report": 11},
        "synthetic.a_consistent": {"yaml": 7, "code": 7, "report": 7},
        "synthetic.m_pending": {
            "yaml": BLOCKED_PENDING_ARTIFACTS,
            "code": 5,
            "report": 5,
        },
    }

    records = detect_conflicts(
        observations,
        required_sources=("yaml", "code", "report"),
    )

    assert tuple(record.path for record in records) == (
        "synthetic.a_consistent",
        "synthetic.m_pending",
        "synthetic.z_conflict",
    )
    by_path = {record.path: record for record in records}
    assert by_path["synthetic.a_consistent"].status is ConflictStatus.CONSISTENT
    assert (
        by_path["synthetic.m_pending"].status
        is ConflictStatus.BLOCKED_PENDING_ARTIFACTS
    )
    assert by_path["synthetic.z_conflict"].status is ConflictStatus.CONFLICT
    assert _observation_values(by_path["synthetic.z_conflict"]) == {
        "code": 22,
        "report": 11,
        "yaml": 11,
    }
    for record in records:
        assert not hasattr(record, "selected_value")
        assert not hasattr(record, "resolved_value")
        assert not hasattr(record, "preferred_value")


def test_missing_required_observation_is_blocked_without_precedence() -> None:
    (record,) = detect_conflicts(
        {"synthetic.pending": {"yaml": 9, "code": 9}},
        required_sources=("yaml", "code", "report"),
    )

    assert record.status is ConflictStatus.BLOCKED_PENDING_ARTIFACTS
    assert record.missing_sources == ("report",)
    assert _observation_values(record) == {"code": 9, "yaml": 9}
    assert not hasattr(record, "selected_value")


def test_known_disagreement_remains_conflict_when_report_is_missing() -> None:
    (record,) = detect_conflicts(
        {"synthetic.threshold": {"yaml": 300, "code": 100}},
        required_sources=("yaml", "code", "report"),
    )

    assert record.status is ConflictStatus.CONFLICT
    assert record.missing_sources == ("report",)
    assert _observation_values(record) == {"code": 100, "yaml": 300}
    assert not hasattr(record, "selected_value")
    assert not hasattr(record, "resolved_value")
    assert not hasattr(record, "preferred_value")


def test_known_disagreement_remains_conflict_when_report_is_blocked() -> None:
    observations = {
        "yaml": 300,
        "code": 100,
        "report": BLOCKED_PENDING_ARTIFACTS,
    }

    (record,) = detect_conflicts(
        {"synthetic.threshold": observations},
        required_sources=("yaml", "code", "report"),
    )

    assert record.status is ConflictStatus.CONFLICT
    assert record.missing_sources == ()
    assert _observation_values(record) == {
        "code": 100,
        "report": BLOCKED_PENDING_ARTIFACTS,
        "yaml": 300,
    }
    assert not hasattr(record, "selected_value")
    assert not hasattr(record, "resolved_value")
    assert not hasattr(record, "preferred_value")


def test_known_agreement_is_blocked_when_report_is_missing() -> None:
    (record,) = detect_conflicts(
        {"synthetic.threshold": {"yaml": 300, "code": 300}},
        required_sources=("yaml", "code", "report"),
    )

    assert record.status is ConflictStatus.BLOCKED_PENDING_ARTIFACTS
    assert record.missing_sources == ("report",)
    assert _observation_values(record) == {"code": 300, "yaml": 300}
    assert not hasattr(record, "selected_value")
    assert not hasattr(record, "resolved_value")
    assert not hasattr(record, "preferred_value")


def test_known_agreement_is_blocked_when_report_is_blocked() -> None:
    (record,) = detect_conflicts(
        {
            "synthetic.threshold": {
                "yaml": 300,
                "code": 300,
                "report": BLOCKED_PENDING_ARTIFACTS,
            }
        },
        required_sources=("yaml", "code", "report"),
    )

    assert record.status is ConflictStatus.BLOCKED_PENDING_ARTIFACTS
    assert record.missing_sources == ()
    assert _observation_values(record) == {
        "code": 300,
        "report": BLOCKED_PENDING_ARTIFACTS,
        "yaml": 300,
    }
    assert not hasattr(record, "selected_value")
    assert not hasattr(record, "resolved_value")
    assert not hasattr(record, "preferred_value")


def test_differing_loaded_targets_are_reported_without_precedence(
    tmp_path: Path,
) -> None:
    documents = _synthetic_documents(resolved_model_target=True)
    documents["data.yaml"]["target"]["poor_mos_threshold"] = 2.25
    documents["model.yaml"]["target"]["mos_threshold"] = 3.75
    loaded = load_config_directory(_write_documents(tmp_path, documents))

    data_value = loaded.values["data"]["target"]["poor_mos_threshold"]
    model_value = loaded.values["model"]["target"]["mos_threshold"]
    assert data_value == 2.25
    assert model_value == 3.75

    (record,) = detect_conflicts(
        {
            "synthetic.target.mos_threshold": {
                "data_yaml": data_value,
                "model_yaml": model_value,
                "report": 4.5,
            }
        },
        required_sources=("data_yaml", "model_yaml", "report"),
    )

    assert record.status is ConflictStatus.CONFLICT
    assert _observation_values(record) == {
        "data_yaml": 2.25,
        "model_yaml": 3.75,
        "report": 4.5,
    }
    assert not hasattr(record, "selected_value")
    assert not hasattr(record, "resolved_value")
    assert not hasattr(record, "preferred_value")


def test_conflict_detection_is_independent_of_input_order() -> None:
    forward = {
        "synthetic.beta": {
            "yaml": {"b": 2, "a": 1},
            "code": {"a": 1, "b": 2},
            "report": {"a": 1, "b": 2},
        },
        "synthetic.alpha": {"report": False, "yaml": True, "code": True},
    }
    reverse = {
        path: dict(reversed(tuple(source_values.items())))
        for path, source_values in reversed(tuple(forward.items()))
    }

    required_sources = ("yaml", "code", "report")
    first = detect_conflicts(forward, required_sources=required_sources)
    second = detect_conflicts(reverse, required_sources=required_sources)

    assert tuple((record.path, record.status) for record in first) == tuple(
        (record.path, record.status) for record in second
    )
    assert [_observation_values(record) for record in first] == [
        _observation_values(record) for record in second
    ]
    assert first[0].status is ConflictStatus.CONFLICT
    assert first[1].status is ConflictStatus.CONSISTENT


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf, object()])
def test_canonicalization_rejects_non_json_or_nonfinite_values(
    invalid_value: Any,
) -> None:
    with pytest.raises(ConfigurationError):
        canonical_json({"synthetic": invalid_value})
    with pytest.raises(ConfigurationError):
        canonical_sha256({"synthetic": invalid_value})


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, object()])
def test_conflict_detector_rejects_non_json_or_nonfinite_observations(
    invalid_value: Any,
) -> None:
    with pytest.raises(ConfigurationError):
        detect_conflicts(
            {"synthetic.invalid": {"yaml": invalid_value}},
            required_sources=("yaml",),
        )


@pytest.mark.parametrize(
    "observations",
    [
        [],
        "synthetic.invalid",
        {"synthetic.valid": {"yaml": 1}, 7: {"yaml": 1}},
        {"": {"yaml": 1}},
        {"synthetic.invalid": [("yaml", 1)]},
        {"synthetic.invalid": {7: 1}},
        {"synthetic.invalid": {"": 1}},
    ],
)
def test_conflict_detector_rejects_malformed_observation_containers_and_names(
    observations: Any,
) -> None:
    with pytest.raises(ConfigurationError):
        detect_conflicts(observations, required_sources=("yaml",))


@pytest.mark.parametrize("required_sources", ["yaml", 7, (), []])
def test_conflict_detector_rejects_invalid_required_sources(
    required_sources: Any,
) -> None:
    with pytest.raises(ConfigurationError):
        detect_conflicts(
            {"synthetic.valid": {"yaml": 1}},
            required_sources=required_sources,
        )


def test_conflict_observations_are_immutable_snapshots() -> None:
    original = {"nested": [1, 2]}
    (record,) = detect_conflicts(
        {
            "synthetic.snapshot": {
                "yaml": original,
                "code": {"nested": [1, 2]},
                "report": {"nested": [1, 2]},
            }
        },
        required_sources=("yaml", "code", "report"),
    )

    original["nested"].append(3)
    yaml_observation = next(
        observation
        for observation in record.observations
        if observation.source == "yaml"
    )
    assert yaml_observation.value["nested"] == (1, 2)
    assert yaml_observation.canonical_value == '{"nested":[1,2]}'
    with pytest.raises(TypeError):
        yaml_observation.value["nested"] = (9,)


def test_each_test_load_uses_an_independent_document_copy(tmp_path: Path) -> None:
    documents = _synthetic_documents()
    mutated = copy.deepcopy(documents)
    mutated["data.yaml"]["dataset"]["name"] = "another-synthetic-fixture"

    original = load_config_directory(_write_documents(tmp_path / "original", documents))
    changed = load_config_directory(_write_documents(tmp_path / "changed", mutated))

    assert original.sha256 != changed.sha256
    assert original.values["data"]["dataset"]["name"] == "synthetic-qoe-fixture"
