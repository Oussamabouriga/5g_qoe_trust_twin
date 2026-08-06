"""Fit the configured models using training rows only."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd

from qoe_twin.artifact_lineage import (
    load_resolved_configuration,
    require_feature_columns,
    sha256_file,
    write_model_lineage,
)
from qoe_twin.baselines import CurrentQoEPersistenceBaseline
from qoe_twin.config import LoadedConfiguration, canonical_sha256
from qoe_twin.features import (
    get_cross_layer_feature_names,
    get_network_feature_names,
)
from qoe_twin.models import (
    build_cross_layer_random_forest,
    build_network_logistic_regression,
)
from qoe_twin.sample_validation import parquet_metadata

DATA_PATH = Path(
    "data/processed/qoe_features_final.parquet"
)

MODEL_DIRECTORY = Path("models/uncalibrated")
RESULT_DIRECTORY = Path("results/metrics")
CONFIG_DIRECTORY = Path("configs")


class TrainingDatasetManifestError(ValueError):
    """Raised when the training dataset cannot be authenticated."""


def _repository_path(repository_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repository_root / candidate).resolve()


def validate_training_dataset_manifest(
    configuration: LoadedConfiguration,
    *,
    data_path: Path = DATA_PATH,
    repository_root: Path = Path("."),
) -> dict[str, str | int]:
    """Authenticate the configured dataset without loading its rows."""
    sample_configuration = configuration.values["data"]["sample"]
    configured_data_path = str(sample_configuration["corrected_feature_file"])
    expected_data_path = _repository_path(repository_root, data_path)
    if _repository_path(repository_root, configured_data_path) != expected_data_path:
        raise TrainingDatasetManifestError(
            "Configured feature path does not match the training "
            f"training DATA_PATH: {configured_data_path!r} != {str(data_path)!r}."
        )

    manifest_relative_path = str(sample_configuration["split_manifest_file"])
    manifest_path = _repository_path(repository_root, manifest_relative_path)
    if not manifest_path.is_file():
        raise TrainingDatasetManifestError(
            f"Configured split manifest is missing: {manifest_path}."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingDatasetManifestError(
            f"Configured split manifest is not readable JSON: {manifest_path}."
        ) from exc
    if not isinstance(manifest, Mapping):
        raise TrainingDatasetManifestError(
            "Configured split manifest must contain a JSON object."
        )
    if manifest.get("artifact_kind") != "corrected_causal_feature_dataset":
        raise TrainingDatasetManifestError(
            "Split manifest artifact_kind must be "
            "'corrected_causal_feature_dataset'."
        )
    if manifest.get("configuration_sha256") != configuration.sha256:
        raise TrainingDatasetManifestError(
            "Split manifest configuration SHA-256 does not match the "
            "current resolved configuration."
        )

    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise TrainingDatasetManifestError(
            "Split manifest output must be a JSON object."
        )
    manifest_output_path = output.get("path")
    if not isinstance(manifest_output_path, str) or not manifest_output_path:
        raise TrainingDatasetManifestError(
            "Split manifest output.path must be a nonempty string."
        )
    if (
        _repository_path(repository_root, manifest_output_path)
        != expected_data_path
    ):
        raise TrainingDatasetManifestError(
            "Split manifest output.path does not match the configured "
            f"training DATA_PATH: {manifest_output_path!r} != {str(data_path)!r}."
        )

    dataset_sha256 = sha256_file(expected_data_path)
    if output.get("sha256") != dataset_sha256:
        raise TrainingDatasetManifestError(
            "Split manifest dataset SHA-256 does not match the exact "
            "Parquet bytes."
        )
    footer = parquet_metadata(expected_data_path)
    for field in ("rows", "columns"):
        recorded = output.get(field)
        if type(recorded) is not int or recorded != footer[field]:
            raise TrainingDatasetManifestError(
                f"Split manifest output.{field} does not match the "
                "Parquet footer."
            )
    schema_sha256 = canonical_sha256(footer["schema"])
    if output.get("schema_sha256") != schema_sha256:
        raise TrainingDatasetManifestError(
            "Split manifest schema SHA-256 does not match the Parquet footer."
        )

    return {
        "columns": footer["columns"],
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_sha256,
        "manifest_path": manifest_relative_path,
        "manifest_sha256": sha256_file(manifest_path),
        "rows": footer["rows"],
        "schema_sha256": schema_sha256,
    }


def final_random_forest_artifact_path(
    model_directory: Path = MODEL_DIRECTORY,
) -> Path:
    """Return the artifact path produced by final RF training."""
    return (
        model_directory
        / "cross_layer_random_forest_final.joblib"
    )


def save_final_random_forest(
    model: object,
    model_directory: Path = MODEL_DIRECTORY,
) -> Path:
    """Persist the final RF to its contracted artifact path."""
    artifact_path = final_random_forest_artifact_path(
        model_directory
    )
    joblib.dump(model, artifact_path)
    return artifact_path


def clean_feature_lists(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Return the complete required feature lists or fail clearly."""
    network_features = list(get_network_feature_names())
    cross_layer_features = list(get_cross_layer_feature_names())

    categorical_features = [
        feature
        for feature in ["resolution"]
        if feature in cross_layer_features
    ]
    cross_numeric_features = [
        feature
        for feature in cross_layer_features
        if feature not in categorical_features
    ]

    require_feature_columns(
        frame.columns,
        network_features,
        context="final network model",
    )
    require_feature_columns(
        frame.columns,
        [*cross_numeric_features, *categorical_features],
        context="final Random Forest",
    )
    return (
        network_features,
        cross_numeric_features,
        categorical_features,
    )


def prepare_split(
    frame: pd.DataFrame,
    split_name: str,
    *,
    target_column: str = "future_poor_qoe",
) -> pd.DataFrame:
    """Return valid rows from one named split."""
    split = frame[frame["split"].eq(split_name)].copy()
    split = split[split[target_column].notna()].copy()
    split[target_column] = split[target_column].astype(int)
    return split


def build_configured_models(
    configuration: LoadedConfiguration,
    *,
    network_features: list[str],
    cross_numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[CurrentQoEPersistenceBaseline, object, object]:
    """Construct the frozen models exclusively from validated YAML values."""
    data_target = configuration.values["data"]["target"]
    model_configuration = configuration.values["model"]
    models = model_configuration["models"]
    required_models = (
        "persistence",
        "logistic_regression",
        "random_forest",
    )
    disabled = [
        name
        for name in required_models
        if not models[name]["enabled"]
    ]
    if disabled:
        raise ValueError(
            "The frozen final experiment requires enabled models: "
            + ", ".join(disabled)
        )

    random_seed = int(
        model_configuration["experiment"]["random_seed"]
    )
    logistic_configuration = models["logistic_regression"]
    forest_configuration = models["random_forest"]
    max_depth_value = forest_configuration["max_depth"]
    max_samples_value = forest_configuration["max_samples"]

    persistence = CurrentQoEPersistenceBaseline(
        poor_mos_threshold=float(
            data_target["poor_mos_threshold"]
        )
    )
    logistic = build_network_logistic_regression(
        numeric_features=network_features,
        random_seed=random_seed,
        max_iter=int(logistic_configuration["max_iter"]),
        class_weight=str(logistic_configuration["class_weight"]),
    )
    random_forest = build_cross_layer_random_forest(
        numeric_features=cross_numeric_features,
        categorical_features=categorical_features,
        random_seed=random_seed,
        n_estimators=int(forest_configuration["n_estimators"]),
        max_depth=(
            None
            if max_depth_value is None
            else int(max_depth_value)
        ),
        min_samples_leaf=int(
            forest_configuration["min_samples_leaf"]
        ),
        class_weight=str(forest_configuration["class_weight"]),
        n_jobs=int(forest_configuration["n_jobs"]),
        max_samples=(
            None
            if max_samples_value is None
            else float(max_samples_value)
        ),
    )
    return persistence, logistic, random_forest


def main() -> None:
    """Fit and save models without inspecting validation or test rows."""
    configuration = load_resolved_configuration(
        CONFIG_DIRECTORY
    )
    dataset_identity = validate_training_dataset_manifest(
        configuration,
        data_path=DATA_PATH,
        repository_root=Path("."),
    )
    target_column = str(
        configuration.values["model"]["target"]["name"]
    )

    print(f"Reading training partition: {DATA_PATH}")
    frame = pd.read_parquet(
        DATA_PATH,
        filters=[("split", "==", "train")],
    )
    train = prepare_split(
        frame,
        "train",
        target_column=target_column,
    )
    (
        network_features,
        cross_numeric_features,
        categorical_features,
    ) = clean_feature_lists(train)
    cross_features = [
        *cross_numeric_features,
        *categorical_features,
    ]

    print("\nTRAINING DATA")
    print("=" * 70)
    print(f"Rows: {len(train):,}")
    print(
        "Poor-QoE rate:",
        f"{train[target_column].mean():.2%}",
    )
    print(f"Network features: {len(network_features)}")
    print(f"Cross-layer features: {len(cross_features)}")

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    persistence, logistic, random_forest = build_configured_models(
        configuration,
        network_features=network_features,
        cross_numeric_features=cross_numeric_features,
        categorical_features=categorical_features,
    )
    target = train[target_column].to_numpy()

    print("\nFitting network-only Logistic Regression...")
    logistic.fit(train[network_features], target)

    print("Fitting cross-layer Random Forest...")
    random_forest.fit(train[cross_features], target)

    persistence_path = MODEL_DIRECTORY / "persistence_final.joblib"
    logistic_path = MODEL_DIRECTORY / "network_logistic_final.joblib"
    joblib.dump(persistence, persistence_path)
    joblib.dump(logistic, logistic_path)
    random_forest_path = save_final_random_forest(
        random_forest,
        MODEL_DIRECTORY,
    )
    write_model_lineage(
        random_forest_path,
        configuration=configuration,
        numeric_features=cross_numeric_features,
        categorical_features=categorical_features,
    )

    model_configuration = configuration.values["model"]
    metadata = {
        "configuration_sha256": configuration.sha256,
        "data_path": str(DATA_PATH),
        "dataset_identity": dataset_identity,
        "training_split": "train",
        "training_rows": len(train),
        "target": {
            "description": "next-observation poor-QoE forecasting",
            "column": target_column,
            "horizon_steps": configuration.values["data"]["target"][
                "horizon_steps"
            ],
            "poor_mos_rule": "future_mos < poor_mos_threshold",
            "poor_mos_threshold": configuration.values["data"]["target"][
                "poor_mos_threshold"
            ],
        },
        "random_seed": model_configuration["experiment"]["random_seed"],
        "model_configuration": {
            "logistic_regression": dict(
                model_configuration["models"]["logistic_regression"]
            ),
            "random_forest": dict(
                model_configuration["models"]["random_forest"]
            ),
        },
        "network_features": network_features,
        "cross_numeric_features": cross_numeric_features,
        "categorical_features": categorical_features,
        "artifacts": {
            "persistence": str(persistence_path),
            "network_logistic": str(logistic_path),
            "cross_layer_random_forest": str(random_forest_path),
        },
    }
    metadata_path = RESULT_DIRECTORY / "final_training_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"- {persistence_path}")
    print(f"- {logistic_path}")
    print(f"- {random_forest_path}")
    print(f"- {metadata_path}")


if __name__ == "__main__":
    main()
