"""Fail-closed lineage sidecars for final model artifacts.

The version-1 format deliberately records only artifact identity,
configuration identity, ordered feature identity, and direct parent hashes.
It does not provide a legacy bypass or choose unresolved configuration values.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from qoe_twin.config import (
    BLOCKED_PENDING_ARTIFACTS,
    ConfigurationError,
    LoadedConfiguration,
    canonical_json,
    load_config_directory,
)

_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_SIZE = 1024 * 1024


class ArtifactLineageError(ValueError):
    """Raised when artifact lineage is missing, malformed, or inconsistent."""


class ArtifactKind(StrEnum):
    """Artifact kinds supported by the version-1 lineage format."""

    FINAL_RANDOM_FOREST = "final_random_forest"
    FINAL_CALIBRATOR = "final_calibrator"
    FINAL_PREDICTIONS = "final_predictions"


@dataclass(frozen=True)
class ArtifactLineage:
    """Validated contents of one artifact lineage sidecar."""

    schema_version: int
    artifact_kind: ArtifactKind
    artifact_filename: str
    artifact_sha256: str
    configuration_sha256: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    feature_list_sha256: str
    parent_model_sha256: str | None = None
    parent_calibrator_sha256: str | None = None


_COMMON_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_filename",
        "artifact_sha256",
        "configuration_sha256",
        "numeric_features",
        "categorical_features",
        "feature_list_sha256",
    }
)
_EXPECTED_KEYS = {
    ArtifactKind.FINAL_RANDOM_FOREST: _COMMON_KEYS,
    ArtifactKind.FINAL_CALIBRATOR: _COMMON_KEYS | {"parent_model_sha256"},
    ArtifactKind.FINAL_PREDICTIONS: _COMMON_KEYS
    | {"parent_model_sha256", "parent_calibrator_sha256"},
}


def sidecar_path(artifact_path: str | Path) -> Path:
    """Return the adjacent lineage-sidecar path for an artifact."""
    path = Path(artifact_path)
    return path.with_name(f"{path.name}.lineage.json")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file's exact bytes."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ArtifactLineageError(f"Artifact file is missing: {artifact_path}")

    digest = hashlib.sha256()
    try:
        with artifact_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactLineageError(
            f"Cannot hash artifact {artifact_path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _ordered_names(
    values: Iterable[str],
    *,
    label: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ArtifactLineageError(f"{label} must be an ordered iterable of names.")
    try:
        names = tuple(values)
    except TypeError as exc:
        raise ArtifactLineageError(
            f"{label} must be an ordered iterable of names."
        ) from exc
    if any(type(name) is not str or not name.strip() for name in names):
        raise ArtifactLineageError(f"{label} must contain nonempty strings.")
    if len(names) != len(set(names)):
        raise ArtifactLineageError(f"{label} must not contain duplicates.")
    return names


def _feature_lists(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = _ordered_names(numeric_features, label="numeric_features")
    categorical = _ordered_names(
        categorical_features,
        label="categorical_features",
    )
    if not numeric and not categorical:
        raise ArtifactLineageError("At least one required feature must be provided.")
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ArtifactLineageError(
            "Numeric and categorical feature lists overlap: " + ", ".join(overlap)
        )
    return numeric, categorical


def feature_list_sha256(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> str:
    """Hash ordered numeric and categorical feature lists canonically."""
    numeric, categorical = _feature_lists(
        numeric_features,
        categorical_features,
    )
    payload = canonical_json(
        {
            "numeric_features": list(numeric),
            "categorical_features": list(categorical),
        }
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_blocked_value(value: Any) -> bool:
    if value == BLOCKED_PENDING_ARTIFACTS:
        return True
    if isinstance(value, Mapping):
        return any(_contains_blocked_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_blocked_value(item) for item in value)
    return False


def resolved_configuration_sha256(
    configuration: LoadedConfiguration,
) -> str:
    """Return a hash only when the configuration is resolved and intact."""
    if not isinstance(configuration, LoadedConfiguration):
        raise ArtifactLineageError(
            "configuration must be a LoadedConfiguration instance."
        )
    if configuration.unresolved_paths:
        unresolved = ", ".join(configuration.unresolved_paths)
        raise ArtifactLineageError(
            f"Configuration is {BLOCKED_PENDING_ARTIFACTS}; "
            "unresolved paths: "
            + unresolved
        )
    if _contains_blocked_value(configuration.values):
        raise ArtifactLineageError(
            "Configuration contains BLOCKED_PENDING_ARTIFACTS."
        )

    try:
        calculated_json = canonical_json(configuration.values)
    except ConfigurationError as exc:
        raise ArtifactLineageError(
            f"Configuration cannot be canonicalized: {exc}"
        ) from exc
    if calculated_json != configuration.canonical_json:
        raise ArtifactLineageError(
            "Configuration canonical JSON does not match its validated values."
        )

    calculated_hash = hashlib.sha256(calculated_json.encode("utf-8")).hexdigest()
    if calculated_hash != configuration.sha256:
        raise ArtifactLineageError(
            "Configuration SHA-256 does not match its canonical JSON."
        )
    return calculated_hash


def load_resolved_configuration(
    directory: str | Path,
) -> LoadedConfiguration:
    """Load the four YAML files and fail when any value is unresolved."""
    try:
        configuration = load_config_directory(directory)
    except ConfigurationError as exc:
        raise ArtifactLineageError(f"Invalid configuration: {exc}") from exc
    resolved_configuration_sha256(configuration)
    return configuration


def require_feature_columns(
    available_columns: Iterable[str],
    required_features: Iterable[str],
    *,
    context: str = "input data",
) -> tuple[str, ...]:
    """Fail if any required feature is absent; never reduce the feature list."""
    required = _ordered_names(required_features, label="required_features")
    if not required:
        raise ArtifactLineageError("required_features must not be empty.")
    try:
        available = set(available_columns)
    except TypeError as exc:
        raise ArtifactLineageError(
            "available_columns must be an iterable of column names."
        ) from exc
    missing = [feature for feature in required if feature not in available]
    if missing:
        raise ArtifactLineageError(
            f"{context} is missing required feature columns: " + ", ".join(missing)
        )
    return required


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactLineageError(
            f"{label} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _artifact_filename(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ArtifactLineageError(
            "artifact_filename must be a nonempty filename without directories."
        )
    return value


def _payload(lineage: ArtifactLineage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": lineage.schema_version,
        "artifact_kind": lineage.artifact_kind.value,
        "artifact_filename": lineage.artifact_filename,
        "artifact_sha256": lineage.artifact_sha256,
        "configuration_sha256": lineage.configuration_sha256,
        "numeric_features": list(lineage.numeric_features),
        "categorical_features": list(lineage.categorical_features),
        "feature_list_sha256": lineage.feature_list_sha256,
    }
    if lineage.artifact_kind is ArtifactKind.FINAL_CALIBRATOR:
        payload["parent_model_sha256"] = lineage.parent_model_sha256
    elif lineage.artifact_kind is ArtifactKind.FINAL_PREDICTIONS:
        payload["parent_model_sha256"] = lineage.parent_model_sha256
        payload["parent_calibrator_sha256"] = lineage.parent_calibrator_sha256
    return payload


def _write_sidecar(lineage: ArtifactLineage, artifact_path: Path) -> None:
    path = sidecar_path(artifact_path)
    text = json.dumps(
        _payload(lineage),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ArtifactLineageError(
            f"Cannot write lineage sidecar {path}: {exc}"
        ) from exc


def _new_lineage(
    artifact_path: str | Path,
    *,
    artifact_kind: ArtifactKind,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    parent_model_sha256: str | None = None,
    parent_calibrator_sha256: str | None = None,
) -> ArtifactLineage:
    path = Path(artifact_path)
    numeric, categorical = _feature_lists(
        numeric_features,
        categorical_features,
    )
    configuration_hash = resolved_configuration_sha256(configuration)

    if artifact_kind is ArtifactKind.FINAL_RANDOM_FOREST:
        if parent_model_sha256 is not None or parent_calibrator_sha256 is not None:
            raise ArtifactLineageError("A model sidecar cannot contain parent hashes.")
    elif artifact_kind is ArtifactKind.FINAL_CALIBRATOR:
        parent_model_sha256 = _require_sha256(
            parent_model_sha256,
            label="parent_model_sha256",
        )
        if parent_calibrator_sha256 is not None:
            raise ArtifactLineageError(
                "A calibrator sidecar cannot contain a calibrator parent."
            )
    else:
        parent_model_sha256 = _require_sha256(
            parent_model_sha256,
            label="parent_model_sha256",
        )
        parent_calibrator_sha256 = _require_sha256(
            parent_calibrator_sha256,
            label="parent_calibrator_sha256",
        )

    return ArtifactLineage(
        schema_version=_SCHEMA_VERSION,
        artifact_kind=artifact_kind,
        artifact_filename=path.name,
        artifact_sha256=sha256_file(path),
        configuration_sha256=configuration_hash,
        numeric_features=numeric,
        categorical_features=categorical,
        feature_list_sha256=feature_list_sha256(numeric, categorical),
        parent_model_sha256=parent_model_sha256,
        parent_calibrator_sha256=parent_calibrator_sha256,
    )


def write_model_lineage(
    artifact_path: str | Path,
    *,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> ArtifactLineage:
    """Write lineage for a newly saved final Random Forest."""
    path = Path(artifact_path)
    lineage = _new_lineage(
        path,
        artifact_kind=ArtifactKind.FINAL_RANDOM_FOREST,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    _write_sidecar(lineage, path)
    return lineage


def write_calibrator_lineage(
    artifact_path: str | Path,
    *,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    parent_model_sha256: str,
) -> ArtifactLineage:
    """Write lineage for a newly saved final calibrator."""
    path = Path(artifact_path)
    lineage = _new_lineage(
        path,
        artifact_kind=ArtifactKind.FINAL_CALIBRATOR,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        parent_model_sha256=parent_model_sha256,
    )
    _write_sidecar(lineage, path)
    return lineage


def write_prediction_lineage(
    artifact_path: str | Path,
    *,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    parent_model_sha256: str,
    parent_calibrator_sha256: str,
) -> ArtifactLineage:
    """Write lineage for a newly saved final prediction file."""
    path = Path(artifact_path)
    lineage = _new_lineage(
        path,
        artifact_kind=ArtifactKind.FINAL_PREDICTIONS,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        parent_model_sha256=parent_model_sha256,
        parent_calibrator_sha256=parent_calibrator_sha256,
    )
    _write_sidecar(lineage, path)
    return lineage


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactLineageError(f"Duplicate JSON key in sidecar: {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ArtifactLineageError(f"Non-finite JSON value is not allowed: {value}.")


def load_selected_final_calibrator(
    selection_path: str | Path,
    calibrated_directory: str | Path,
    *,
    configuration: LoadedConfiguration,
) -> tuple[str, Path, dict[str, Any]]:
    """Load the recorded final RF selection and resolve its calibrator."""
    path = Path(selection_path)
    if not path.is_file():
        raise ArtifactLineageError(
            f"Selected calibration configuration is missing: {path}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactLineageError(
            f"Cannot read selected calibration configuration {path}: {exc}"
        ) from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except ArtifactLineageError:
        raise
    except json.JSONDecodeError as exc:
        raise ArtifactLineageError(
            f"Selected calibration configuration is not valid JSON: {path}: {exc}"
        ) from exc

    if type(payload) is not dict:
        raise ArtifactLineageError(
            "Selected calibration configuration must contain one object."
        )
    configuration_hash = resolved_configuration_sha256(configuration)
    recorded_configuration_hash = _require_sha256(
        payload.get("configuration_sha256"),
        label="selected configuration_sha256",
    )
    if recorded_configuration_hash != configuration_hash:
        raise ArtifactLineageError(
            "Selected calibration configuration SHA-256 does not match "
            "the resolved configuration."
        )
    selected = payload.get("cross_layer_random_forest")
    if type(selected) is not dict:
        raise ArtifactLineageError(
            "Selected calibration configuration must contain a "
            "cross_layer_random_forest object."
        )

    method = selected.get("calibration_method")
    if method == "uncalibrated":
        raise ArtifactLineageError(
            "The selected calibration_method is 'uncalibrated', but the final "
            "pipeline requires a calibrator."
        )
    if method not in {"sigmoid", "isotonic"}:
        raise ArtifactLineageError(
            f"Unsupported final calibration_method {method!r}; expected "
            "'sigmoid' or 'isotonic'."
        )

    calibrator_path = Path(calibrated_directory) / (
        f"cross_layer_random_forest_{method}_final.joblib"
    )
    return method, calibrator_path, selected


def _read_sidecar(path: Path) -> ArtifactLineage:
    if not path.is_file():
        raise ArtifactLineageError(f"Required lineage sidecar is missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactLineageError(
            f"Cannot read lineage sidecar {path}: {exc}"
        ) from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except ArtifactLineageError:
        raise
    except json.JSONDecodeError as exc:
        raise ArtifactLineageError(
            f"Lineage sidecar is not valid JSON: {path}: {exc}"
        ) from exc
    if type(payload) is not dict:
        raise ArtifactLineageError(f"Lineage sidecar must contain one object: {path}")

    raw_kind = payload.get("artifact_kind")
    if type(raw_kind) is not str:
        raise ArtifactLineageError("artifact_kind must be a string.")
    try:
        kind = ArtifactKind(raw_kind)
    except ValueError as exc:
        raise ArtifactLineageError(f"Unknown artifact_kind: {raw_kind!r}.") from exc

    expected_keys = _EXPECTED_KEYS[kind]
    actual_keys = set(payload)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing:
        raise ArtifactLineageError(
            "Lineage sidecar is missing required values: " + ", ".join(missing)
        )
    if unknown:
        raise ArtifactLineageError(
            "Lineage sidecar contains unknown values: " + ", ".join(unknown)
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ArtifactLineageError("schema_version must be integer 1.")

    numeric_raw = payload["numeric_features"]
    categorical_raw = payload["categorical_features"]
    if type(numeric_raw) is not list or type(categorical_raw) is not list:
        raise ArtifactLineageError(
            "numeric_features and categorical_features must be JSON arrays."
        )
    numeric, categorical = _feature_lists(numeric_raw, categorical_raw)
    recorded_feature_hash = _require_sha256(
        payload["feature_list_sha256"],
        label="feature_list_sha256",
    )
    calculated_feature_hash = feature_list_sha256(numeric, categorical)
    if recorded_feature_hash != calculated_feature_hash:
        raise ArtifactLineageError(
            "feature_list_sha256 does not match the ordered feature lists."
        )

    parent_model_hash = None
    parent_calibrator_hash = None
    if kind is ArtifactKind.FINAL_CALIBRATOR:
        parent_model_hash = _require_sha256(
            payload["parent_model_sha256"],
            label="parent_model_sha256",
        )
    elif kind is ArtifactKind.FINAL_PREDICTIONS:
        parent_model_hash = _require_sha256(
            payload["parent_model_sha256"],
            label="parent_model_sha256",
        )
        parent_calibrator_hash = _require_sha256(
            payload["parent_calibrator_sha256"],
            label="parent_calibrator_sha256",
        )

    return ArtifactLineage(
        schema_version=payload["schema_version"],
        artifact_kind=kind,
        artifact_filename=_artifact_filename(payload["artifact_filename"]),
        artifact_sha256=_require_sha256(
            payload["artifact_sha256"],
            label="artifact_sha256",
        ),
        configuration_sha256=_require_sha256(
            payload["configuration_sha256"],
            label="configuration_sha256",
        ),
        numeric_features=numeric,
        categorical_features=categorical,
        feature_list_sha256=recorded_feature_hash,
        parent_model_sha256=parent_model_hash,
        parent_calibrator_sha256=parent_calibrator_hash,
    )


def _validate_lineage(
    artifact_path: str | Path,
    *,
    expected_kind: ArtifactKind,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    expected_parent_model_sha256: str | None = None,
    expected_parent_calibrator_sha256: str | None = None,
) -> ArtifactLineage:
    path = Path(artifact_path)
    numeric, categorical = _feature_lists(
        numeric_features,
        categorical_features,
    )
    configuration_hash = resolved_configuration_sha256(configuration)
    lineage = _read_sidecar(sidecar_path(path))

    if lineage.artifact_kind is not expected_kind:
        raise ArtifactLineageError(
            f"Wrong artifact kind for {path}: {lineage.artifact_kind.value}; "
            f"expected {expected_kind.value}."
        )
    if lineage.artifact_filename != path.name:
        raise ArtifactLineageError(
            f"Artifact filename mismatch for {path}: "
            f"sidecar records {lineage.artifact_filename!r}."
        )
    current_hash = sha256_file(path)
    if lineage.artifact_sha256 != current_hash:
        raise ArtifactLineageError(
            f"Artifact SHA-256 mismatch for {path}; its bytes have changed."
        )
    if lineage.configuration_sha256 != configuration_hash:
        raise ArtifactLineageError(
            f"Configuration SHA-256 mismatch for {path}."
        )
    if lineage.numeric_features != numeric:
        raise ArtifactLineageError(
            f"Ordered numeric feature list mismatch for {path}."
        )
    if lineage.categorical_features != categorical:
        raise ArtifactLineageError(
            f"Ordered categorical feature list mismatch for {path}."
        )

    expected_feature_hash = feature_list_sha256(numeric, categorical)
    if lineage.feature_list_sha256 != expected_feature_hash:
        raise ArtifactLineageError(f"Feature-list SHA-256 mismatch for {path}.")

    if expected_kind is ArtifactKind.FINAL_RANDOM_FOREST:
        if (
            expected_parent_model_sha256 is not None
            or expected_parent_calibrator_sha256 is not None
        ):
            raise ArtifactLineageError("A model validator cannot expect parent hashes.")
    elif expected_kind is ArtifactKind.FINAL_CALIBRATOR:
        expected_model_hash = _require_sha256(
            expected_parent_model_sha256,
            label="expected_parent_model_sha256",
        )
        if lineage.parent_model_sha256 != expected_model_hash:
            raise ArtifactLineageError(
                f"Calibrator parent model SHA-256 mismatch for {path}."
            )
        if expected_parent_calibrator_sha256 is not None:
            raise ArtifactLineageError(
                "A calibrator validator cannot expect a calibrator parent."
            )
    else:
        expected_model_hash = _require_sha256(
            expected_parent_model_sha256,
            label="expected_parent_model_sha256",
        )
        expected_calibrator_hash = _require_sha256(
            expected_parent_calibrator_sha256,
            label="expected_parent_calibrator_sha256",
        )
        if lineage.parent_model_sha256 != expected_model_hash:
            raise ArtifactLineageError(
                f"Prediction parent model SHA-256 mismatch for {path}."
            )
        if lineage.parent_calibrator_sha256 != expected_calibrator_hash:
            raise ArtifactLineageError(
                f"Prediction parent calibrator SHA-256 mismatch for {path}."
            )
    return lineage


def validate_model_lineage(
    artifact_path: str | Path,
    *,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> ArtifactLineage:
    """Validate a final Random Forest artifact and sidecar."""
    return _validate_lineage(
        artifact_path,
        expected_kind=ArtifactKind.FINAL_RANDOM_FOREST,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def validate_calibrator_lineage(
    artifact_path: str | Path,
    *,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    expected_parent_model_sha256: str,
) -> ArtifactLineage:
    """Validate a final calibrator and its exact model parent."""
    return _validate_lineage(
        artifact_path,
        expected_kind=ArtifactKind.FINAL_CALIBRATOR,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        expected_parent_model_sha256=expected_parent_model_sha256,
    )


def validate_prediction_lineage(
    artifact_path: str | Path,
    *,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    expected_parent_model_sha256: str,
    expected_parent_calibrator_sha256: str,
) -> ArtifactLineage:
    """Validate final predictions and both exact artifact parents."""
    return _validate_lineage(
        artifact_path,
        expected_kind=ArtifactKind.FINAL_PREDICTIONS,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        expected_parent_model_sha256=expected_parent_model_sha256,
        expected_parent_calibrator_sha256=expected_parent_calibrator_sha256,
    )


def validate_prediction_chain(
    model_path: str | Path,
    calibrator_path: str | Path,
    prediction_path: str | Path,
    *,
    configuration: LoadedConfiguration,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> tuple[ArtifactLineage, ArtifactLineage, ArtifactLineage]:
    """Validate an exact model -> calibrator -> prediction chain."""
    model = validate_model_lineage(
        model_path,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    calibrator = validate_calibrator_lineage(
        calibrator_path,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        expected_parent_model_sha256=model.artifact_sha256,
    )
    predictions = validate_prediction_lineage(
        prediction_path,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        expected_parent_model_sha256=model.artifact_sha256,
        expected_parent_calibrator_sha256=calibrator.artifact_sha256,
    )
    return model, calibrator, predictions
