"""Regression tests for the Random Forest artifact contract."""

from pathlib import Path
from unittest.mock import Mock

import joblib
import pytest

import scripts.apply_trust as final_inference
import scripts.calibrate_models as final_calibration
from scripts.calibrate_models import (
    required_final_random_forest_artifact_path,
)
from scripts.train_models import (
    final_random_forest_artifact_path,
    save_final_random_forest,
)

FINAL_RF_FILENAME = (
    "cross_layer_random_forest_final.joblib"
)
UNRELATED_RF_FILENAME = "unrelated_random_forest.joblib"


def _write_unrelated_artifact(directory: Path) -> Path:
    artifact_path = directory / UNRELATED_RF_FILENAME
    joblib.dump(
        {"artifact_scope": "unrelated"},
        artifact_path,
    )
    return artifact_path


def test_training_writes_exact_calibration_input(
    tmp_path: Path,
) -> None:
    saved_path = save_final_random_forest(
        {"artifact_scope": "frozen"},
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
        "artifact_scope": "frozen"
    }


def test_requirement_rejects_unrelated_artifact(
    tmp_path: Path,
) -> None:
    unrelated_path = _write_unrelated_artifact(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match=r"cross_layer_random_forest_final\.joblib",
    ):
        required_final_random_forest_artifact_path(
            tmp_path
        )

    assert unrelated_path.is_file()
    assert not (tmp_path / FINAL_RF_FILENAME).exists()


def test_calibration_main_fails_before_reading_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated_path = _write_unrelated_artifact(tmp_path)
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
    assert unrelated_path.is_file()
    assert not (tmp_path / FINAL_RF_FILENAME).exists()


def test_inference_targets_required_random_forest() -> None:
    assert final_inference.MODEL_PATH == Path(
        "models/uncalibrated"
    ) / FINAL_RF_FILENAME
