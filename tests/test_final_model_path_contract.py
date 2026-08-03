"""Regression tests for the final Random Forest artifact contract."""

import subprocess
from pathlib import Path
from unittest.mock import Mock

import joblib
import pytest

import scripts.apply_trust as prototype_inference
import scripts.calibrate_models_final as final_calibration
from scripts.calibrate_models_final import (
    required_final_random_forest_artifact_path,
)
from scripts.train_models_final import (
    final_random_forest_artifact_path,
    save_final_random_forest,
)

FINAL_RF_FILENAME = (
    "cross_layer_random_forest_final.joblib"
)
PROTOTYPE_RF_FILENAME = (
    "cross_layer_random_forest_prototype.joblib"
)
PROTOTYPE_SCRIPTS = (
    "scripts/train_models.py",
    "scripts/calibrate_models.py",
    "scripts/apply_trust.py",
)


def _write_prototype_artifact(directory: Path) -> Path:
    artifact_path = directory / PROTOTYPE_RF_FILENAME
    joblib.dump(
        {"artifact_scope": "prototype"},
        artifact_path,
    )
    return artifact_path


def test_final_training_writes_exact_calibration_input(
    tmp_path: Path,
) -> None:
    saved_path = save_final_random_forest(
        {"artifact_scope": "final"},
        tmp_path,
    )

    required_path = (
        required_final_random_forest_artifact_path(
            tmp_path
        )
    )

    assert saved_path == required_path
    assert required_path == (
        final_random_forest_artifact_path(tmp_path)
    )
    assert required_path == tmp_path / FINAL_RF_FILENAME
    assert joblib.load(required_path) == {
        "artifact_scope": "final"
    }


def test_final_requirement_rejects_prototype_only_directory(
    tmp_path: Path,
) -> None:
    prototype_path = _write_prototype_artifact(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match=r"cross_layer_random_forest_final\.joblib",
    ):
        required_final_random_forest_artifact_path(
            tmp_path
        )

    assert prototype_path.is_file()
    assert not (tmp_path / FINAL_RF_FILENAME).exists()


def test_final_calibration_main_fails_before_reading_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prototype_path = _write_prototype_artifact(tmp_path)
    read_parquet = Mock(
        side_effect=AssertionError(
            "pd.read_parquet must not run before RF validation"
        )
    )

    monkeypatch.setattr(
        final_calibration,
        "MODEL_DIRECTORY",
        tmp_path,
    )
    monkeypatch.setattr(
        final_calibration.pd,
        "read_parquet",
        read_parquet,
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"cross_layer_random_forest_final\.joblib",
    ):
        final_calibration.main()

    read_parquet.assert_not_called()
    assert prototype_path.is_file()
    assert not (tmp_path / FINAL_RF_FILENAME).exists()


def test_prototype_scripts_are_unchanged_according_to_git() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            *PROTOTYPE_SCRIPTS,
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )


def test_prototype_inference_still_targets_prototype_rf() -> None:
    assert prototype_inference.MODEL_PATH == Path(
        "models/uncalibrated"
    ) / PROTOTYPE_RF_FILENAME
