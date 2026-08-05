"""Strict, deterministic loading for the project's four YAML documents.

This module validates configuration as evidence.  It does not reconcile
scientific disagreements, copy values between files, or apply source
precedence.  Explicitly unresolved target values remain blocked until the
required artifacts are available.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import yaml

BLOCKED_PENDING_ARTIFACTS = "BLOCKED_PENDING_ARTIFACTS"

CONFIG_FILENAMES = (
    "data.yaml",
    "llm.yaml",
    "model.yaml",
    "trust.yaml",
)

_ENVIRONMENT_VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UINT32_MAX = 2**32 - 1
_SUM_TOLERANCE = 1e-9


class ConfigurationError(ValueError):
    """Raised when configuration evidence is incomplete or invalid."""


class ConflictStatus(StrEnum):
    """Outcome of comparing caller-supplied configuration observations."""

    CONSISTENT = "CONSISTENT"
    CONFLICT = "CONFLICT"
    BLOCKED_PENDING_ARTIFACTS = BLOCKED_PENDING_ARTIFACTS


@dataclass(frozen=True)
class LoadedConfiguration:
    """Validated configuration values and their deterministic identity."""

    values: Mapping[str, Any]
    canonical_json: str
    sha256: str
    unresolved_paths: tuple[str, ...]


@dataclass(frozen=True)
class ConfigObservation:
    """One source's observation, retained without selecting it."""

    source: str
    value: Any
    canonical_value: str


@dataclass(frozen=True)
class ConflictRecord:
    """Order-independent comparison result with no resolved-value field."""

    path: str
    status: ConflictStatus
    observations: tuple[ConfigObservation, ...]
    missing_sources: tuple[str, ...]
    distinct_values: tuple[str, ...]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that fails instead of overwriting duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ConfigurationError("YAML merge keys are not allowed.")
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConfigurationError("YAML mapping keys must be scalar.") from exc
        if duplicate:
            raise ConfigurationError(f"Duplicate YAML key: {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _normalize_json_value(value: Any, path: str = "$") -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ConfigurationError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is str:
                normalized_key = key
            elif type(key) is int:
                normalized_key = str(key)
            else:
                raise ConfigurationError(
                    f"{path} has a non-JSON mapping key: {key!r}."
                )
            if normalized_key in normalized:
                raise ConfigurationError(
                    f"{path} has colliding canonical key {normalized_key!r}."
                )
            normalized[normalized_key] = _normalize_json_value(
                item,
                f"{path}.{normalized_key}",
            )
        return normalized
    raise ConfigurationError(
        f"{path} contains a non-JSON value of type {type(value).__name__}."
    )


def _deep_freeze(value: Any) -> Any:
    """Copy JSON-like values into immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    """Return compact, key-sorted JSON with no filesystem-dependent content."""
    normalized = _normalize_json_value(value)
    try:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Value cannot be canonicalized: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    """Hash the UTF-8 bytes of :func:`canonical_json`."""
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


Validator = Callable[[Any, str, list[str]], Any]


def _mapping(fields: Mapping[str, Validator]) -> Validator:
    expected = set(fields)

    def validate(value: Any, path: str, unresolved: list[str]) -> dict[str, Any]:
        if type(value) is not dict:
            raise ConfigurationError(f"{path} must be a mapping.")
        actual = set(value)
        non_string_keys = [key for key in actual if type(key) is not str]
        if non_string_keys:
            raise ConfigurationError(
                f"{path} contains non-string keys: {non_string_keys!r}."
            )
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            qualified = ", ".join(f"{path}.{key}" for key in missing)
            raise ConfigurationError(f"Missing required values: {qualified}.")
        if unknown:
            qualified = ", ".join(f"{path}.{key}" for key in unknown)
            raise ConfigurationError(f"Unknown values: {qualified}.")
        return {
            key: fields[key](value[key], f"{path}.{key}", unresolved)
            for key in sorted(fields)
        }

    return validate


def _nonempty_string(value: Any, path: str, unresolved: list[str]) -> str:
    del unresolved
    if type(value) is not str or not value.strip():
        raise ConfigurationError(f"{path} must be a nonempty string.")
    return value


def _environment_variable(value: Any, path: str, unresolved: list[str]) -> str:
    value = _nonempty_string(value, path, unresolved)
    if _ENVIRONMENT_VARIABLE_RE.fullmatch(value) is None:
        raise ConfigurationError(
            f"{path} must be a valid environment-variable name."
        )
    return value


def _relative_path(value: Any, path: str, unresolved: list[str]) -> str:
    value = _nonempty_string(value, path, unresolved)
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ConfigurationError(f"{path} must be a safe relative path.")
    return value


def _sha256(value: Any, path: str, unresolved: list[str]) -> str:
    value = _nonempty_string(value, path, unresolved)
    if _SHA256_RE.fullmatch(value) is None:
        raise ConfigurationError(
            f"{path} must be a lowercase 64-digit SHA-256 digest."
        )
    return value


def _separator(value: Any, path: str, unresolved: list[str]) -> str:
    del unresolved
    if type(value) is not str or len(value) != 1:
        raise ConfigurationError(f"{path} must contain exactly one character.")
    return value


def _boolean(value: Any, path: str, unresolved: list[str]) -> bool:
    del unresolved
    if type(value) is not bool:
        raise ConfigurationError(f"{path} must be a Boolean.")
    return value


def _integer(
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    forbidden: frozenset[int] = frozenset(),
) -> Validator:
    def validate(value: Any, path: str, unresolved: list[str]) -> int:
        del unresolved
        if type(value) is not int:
            raise ConfigurationError(f"{path} must be an integer.")
        if minimum is not None and value < minimum:
            raise ConfigurationError(f"{path} must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ConfigurationError(f"{path} must be at most {maximum}.")
        if value in forbidden:
            raise ConfigurationError(f"{path} cannot be {value}.")
        return value

    return validate


def _number(
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> Validator:
    def validate(value: Any, path: str, unresolved: list[str]) -> int | float:
        del unresolved
        if type(value) not in {int, float}:
            raise ConfigurationError(f"{path} must be a number.")
        if not math.isfinite(value):
            raise ConfigurationError(f"{path} must be finite.")
        if minimum is not None:
            below = value < minimum if minimum_inclusive else value <= minimum
            if below:
                qualifier = "at least" if minimum_inclusive else "greater than"
                raise ConfigurationError(f"{path} must be {qualifier} {minimum}.")
        if maximum is not None:
            above = value > maximum if maximum_inclusive else value >= maximum
            if above:
                qualifier = "at most" if maximum_inclusive else "less than"
                raise ConfigurationError(f"{path} must be {qualifier} {maximum}.")
        return value

    return validate


def _nullable(validator: Validator) -> Validator:
    def validate(value: Any, path: str, unresolved: list[str]) -> Any:
        if value is None:
            return None
        return validator(value, path, unresolved)

    return validate


def _blocked_if_null(validator: Validator) -> Validator:
    def validate(value: Any, path: str, unresolved: list[str]) -> Any:
        if value is None:
            unresolved.append(path)
            return BLOCKED_PENDING_ARTIFACTS
        return validator(value, path, unresolved)

    return validate


def _list_of(
    validator: Validator,
    *,
    unique: bool = False,
) -> Validator:
    def validate(value: Any, path: str, unresolved: list[str]) -> list[Any]:
        if type(value) is not list or not value:
            raise ConfigurationError(f"{path} must be a nonempty list.")
        result = [
            validator(item, f"{path}[{index}]", unresolved)
            for index, item in enumerate(value)
        ]
        if unique:
            identities = [canonical_json(item) for item in result]
            if len(identities) != len(set(identities)):
                raise ConfigurationError(f"{path} must contain unique values.")
        return result

    return validate


def _bitrate_resolution_mapping(
    value: Any,
    path: str,
    unresolved: list[str],
) -> dict[int, str]:
    if type(value) is not dict or not value:
        raise ConfigurationError(f"{path} must be a nonempty mapping.")
    result: dict[int, str] = {}
    for key, resolution in value.items():
        if type(key) is not int or key <= 0:
            raise ConfigurationError(
                f"{path} keys must be positive integer bitrates."
            )
        result[key] = _nonempty_string(
            resolution,
            f"{path}.{key}",
            unresolved,
        )
    return {key: result[key] for key in sorted(result)}


_POSITIVE_INTEGER = _integer(minimum=1)
_POSITIVE_NUMBER = _number(minimum=0.0, minimum_inclusive=False)
_UNIT_INTERVAL = _number(minimum=0.0, maximum=1.0)
_OPEN_UNIT_INTERVAL = _number(
    minimum=0.0,
    maximum=1.0,
    minimum_inclusive=False,
    maximum_inclusive=False,
)
_POSITIVE_UNIT_INTERVAL = _number(
    minimum=0.0,
    maximum=1.0,
    minimum_inclusive=False,
)
_MOS_VALUE = _number(minimum=1.0, maximum=5.0)

_DATA_SCHEMA = _mapping(
    {
        "dataset": _mapping(
            {
                "directory": _relative_path,
                "file_pattern": _relative_path,
                "name": _nonempty_string,
                "separator": _separator,
            }
        ),
        "processing": _mapping(
            {
                "output_file": _relative_path,
                "preserve_original_files": _boolean,
                "sort_chronologically": _boolean,
            }
        ),
        "sample": _mapping(
            {
                "corrected_feature_file": _relative_path,
                "input_file": _relative_path,
                "input_sha256": _sha256,
                "provenance_manifest": _relative_path,
                "provenance_manifest_sha256": _sha256,
                "split_manifest_file": _relative_path,
            }
        ),
        "sessions": _mapping({"gap_threshold_seconds": _POSITIVE_NUMBER}),
        "splitting": _mapping(
            {
                "method": _nonempty_string,
                "test_fraction": _OPEN_UNIT_INTERVAL,
                "train_fraction": _OPEN_UNIT_INTERVAL,
                "validation_fraction": _OPEN_UNIT_INTERVAL,
            }
        ),
        "target": _mapping(
            {
                "horizon_steps": _POSITIVE_INTEGER,
                "poor_mos_threshold": _MOS_VALUE,
            }
        ),
        "video": _mapping(
            {
                "bitrate_to_resolution": _bitrate_resolution_mapping,
                "selected_bitrates_kbps": _list_of(
                    _POSITIVE_INTEGER,
                    unique=True,
                ),
            }
        ),
    }
)

_MODEL_SCHEMA = _mapping(
    {
        "calibration": _mapping(
            {
                "fit_fraction": _OPEN_UNIT_INTERVAL,
                "methods": _list_of(_nonempty_string, unique=True),
            }
        ),
        "experiment": _mapping(
            {"random_seed": _integer(minimum=0, maximum=_UINT32_MAX)}
        ),
        "models": _mapping(
            {
                "logistic_regression": _mapping(
                    {
                        "class_weight": _nonempty_string,
                        "enabled": _boolean,
                        "max_iter": _POSITIVE_INTEGER,
                    }
                ),
                "persistence": _mapping({"enabled": _boolean}),
                "random_forest": _mapping(
                    {
                        "class_weight": _nonempty_string,
                        "enabled": _boolean,
                        "max_depth": _nullable(_POSITIVE_INTEGER),
                        "max_samples": _nullable(_POSITIVE_UNIT_INTERVAL),
                        "min_samples_leaf": _POSITIVE_INTEGER,
                        "n_estimators": _POSITIVE_INTEGER,
                        "n_jobs": _integer(forbidden=frozenset({0})),
                    }
                ),
            }
        ),
        "target": _mapping(
            {
                "horizon_steps": _blocked_if_null(_POSITIVE_INTEGER),
                "mos_threshold": _blocked_if_null(_MOS_VALUE),
                "name": _nonempty_string,
            }
        ),
    }
)

_TRUST_SCHEMA = _mapping(
    {
        "abstention": _mapping(
            {
                "enabled": _boolean,
                "minimum_trust": _UNIT_INTERVAL,
            }
        ),
        "levels": _mapping(
            {
                "high": _UNIT_INTERVAL,
                "medium": _UNIT_INTERVAL,
            }
        ),
        "trust": _mapping(
            {
                "calibration_weight": _UNIT_INTERVAL,
                "confidence_weight": _UNIT_INTERVAL,
                "data_quality_weight": _UNIT_INTERVAL,
                "validation_performance_weight": _UNIT_INTERVAL,
            }
        ),
    }
)

_LLM_SCHEMA = _mapping(
    {
        "explanation": _mapping(
            {
                "require_confidence": _boolean,
                "require_likely_causes": _boolean,
                "require_limitations": _boolean,
                "require_operator_checks": _boolean,
                "require_predicted_event": _boolean,
                "require_supporting_fields": _boolean,
            }
        ),
        "grounding": _mapping(
            {
                "reject_missing_required_sections": _boolean,
                "reject_unknown_fields": _boolean,
                "reject_unsupported_causal_claims": _boolean,
                "reject_unsupported_numbers": _boolean,
            }
        ),
        "openai": _mapping(
            {
                "api_key_environment_variable": _environment_variable,
                "maximum_retries": _integer(minimum=0),
                "model_environment_variable": _environment_variable,
                "temperature": _number(minimum=0.0, maximum=2.0),
                "timeout_seconds": _POSITIVE_NUMBER,
            }
        ),
    }
)

_SCHEMAS: dict[str, Validator] = {
    "data.yaml": _DATA_SCHEMA,
    "llm.yaml": _LLM_SCHEMA,
    "model.yaml": _MODEL_SCHEMA,
    "trust.yaml": _TRUST_SCHEMA,
}


def _load_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"Cannot read {path.name}: {exc}") from exc
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except ConfigurationError as exc:
        raise ConfigurationError(f"{path.name}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{path.name} is not valid YAML: {exc}") from exc


def _validate_cross_field_constraints(values: Mapping[str, Any]) -> None:
    data = values["data"]
    selected_bitrates = data["video"]["selected_bitrates_kbps"]
    resolution_bitrates = set(data["video"]["bitrate_to_resolution"])
    if set(selected_bitrates) != resolution_bitrates:
        raise ConfigurationError(
            "data.video.bitrate_to_resolution keys must exactly match "
            "data.video.selected_bitrates_kbps."
        )

    splitting = data["splitting"]
    if splitting["method"] != "global_chronological":
        raise ConfigurationError(
            "data.splitting.method must be 'global_chronological'."
        )
    split_sum = sum(
        splitting[name]
        for name in (
            "train_fraction",
            "validation_fraction",
            "test_fraction",
        )
    )
    if not math.isclose(
        split_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=_SUM_TOLERANCE,
    ):
        raise ConfigurationError("data.splitting fractions must sum to 1.0.")

    model = values["model"]
    model_target = model["target"]
    target_pairs = (
        (
            "horizon_steps",
            data["target"]["horizon_steps"],
            model_target["horizon_steps"],
        ),
        (
            "MOS threshold",
            data["target"]["poor_mos_threshold"],
            model_target["mos_threshold"],
        ),
    )
    for label, data_value, model_value in target_pairs:
        if (
            model_value != BLOCKED_PENDING_ARTIFACTS
            and data_value != model_value
        ):
            raise ConfigurationError(
                f"Data and model target {label} values must match."
            )

    calibration_methods = tuple(model["calibration"]["methods"])
    if calibration_methods != ("sigmoid", "isotonic"):
        raise ConfigurationError(
            "model.calibration.methods must be exactly "
            "['sigmoid', 'isotonic'] in that order."
        )

    trust = values["trust"]
    weight_sum = sum(trust["trust"].values())
    if not math.isclose(
        weight_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=_SUM_TOLERANCE,
    ):
        raise ConfigurationError("trust.trust weights must sum to 1.0.")
    if trust["levels"]["high"] <= trust["levels"]["medium"]:
        raise ConfigurationError("trust.levels.high must exceed trust.levels.medium.")


def load_config_directory(directory: str | Path) -> LoadedConfiguration:
    """Load and strictly validate the four project YAML files.

    No default or precedence rule is applied.  Explicit unresolved model-target
    nulls become :data:`BLOCKED_PENDING_ARTIFACTS` and are listed separately.
    """
    config_directory = Path(directory)
    if not config_directory.is_dir():
        raise ConfigurationError(
            f"Configuration directory does not exist: {config_directory}."
        )

    actual_yaml_files = {
        path.name
        for path in config_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    }
    expected_yaml_files = set(CONFIG_FILENAMES)
    missing = sorted(expected_yaml_files - actual_yaml_files)
    unknown = sorted(actual_yaml_files - expected_yaml_files)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unexpected: {', '.join(unknown)}")
        raise ConfigurationError(
            "Configuration file set is invalid (" + "; ".join(details) + ")."
        )

    unresolved: list[str] = []
    values: dict[str, Any] = {}
    for filename in CONFIG_FILENAMES:
        document_name = Path(filename).stem
        document = _load_yaml(config_directory / filename)
        values[document_name] = _SCHEMAS[filename](
            document,
            document_name,
            unresolved,
        )

    _validate_cross_field_constraints(values)
    frozen_values = _deep_freeze(values)
    canonical = canonical_json(frozen_values)
    return LoadedConfiguration(
        values=frozen_values,
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        unresolved_paths=tuple(sorted(unresolved)),
    )


def _contains_blocked_value(value: Any) -> bool:
    if value == BLOCKED_PENDING_ARTIFACTS:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_blocked_value(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_blocked_value(item) for item in value.values())
    return False


def _validated_names(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ConfigurationError(f"{label} must be an iterable of names, not text.")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ConfigurationError(f"{label} must be an iterable of names.") from exc
    if not result:
        raise ConfigurationError(f"{label} must not be empty.")
    if any(type(value) is not str or not value.strip() for value in result):
        raise ConfigurationError(f"{label} must contain nonempty strings.")
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{label} must not contain duplicates.")
    return tuple(sorted(result))


def detect_conflicts(
    observations: Mapping[str, Mapping[str, Any]],
    *,
    required_sources: Iterable[str],
) -> tuple[ConflictRecord, ...]:
    """Compare observations without resolving them or applying precedence.

    Disagreement between available, non-blocked observations produces a
    conflict even when another required source is missing or blocked.  With no
    known disagreement, missing or blocked evidence produces a blocked record.
    Every source value is retained; no winner is exposed.
    """
    if not isinstance(observations, Mapping):
        raise ConfigurationError("observations must be a mapping.")
    required = _validated_names(required_sources, "required_sources")
    paths = _validated_names(observations.keys(), "observation paths")
    records: list[ConflictRecord] = []

    for path in paths:
        source_values = observations[path]
        if not isinstance(source_values, Mapping) or not source_values:
            raise ConfigurationError(
                f"Observations for {path!r} must be a nonempty mapping."
            )
        sources = _validated_names(source_values.keys(), f"sources for {path}")
        entries_list: list[ConfigObservation] = []
        for source in sources:
            observed_value = _deep_freeze(source_values[source])
            entries_list.append(
                ConfigObservation(
                    source=source,
                    value=observed_value,
                    canonical_value=canonical_json(observed_value),
                )
            )
        entries = tuple(entries_list)
        missing_sources = tuple(sorted(set(required) - set(sources)))
        distinct_values = tuple(
            sorted({entry.canonical_value for entry in entries})
        )
        blocked_observations = tuple(
            _contains_blocked_value(entry.value) for entry in entries
        )
        available_distinct_values = {
            entry.canonical_value
            for entry, is_blocked in zip(entries, blocked_observations, strict=True)
            if not is_blocked
        }
        blocked = bool(missing_sources) or any(blocked_observations)
        if len(available_distinct_values) > 1:
            status = ConflictStatus.CONFLICT
        elif blocked:
            status = ConflictStatus.BLOCKED_PENDING_ARTIFACTS
        else:
            status = ConflictStatus.CONSISTENT
        records.append(
            ConflictRecord(
                path=path,
                status=status,
                observations=entries,
                missing_sources=missing_sources,
                distinct_values=distinct_values,
            )
        )

    return tuple(records)
