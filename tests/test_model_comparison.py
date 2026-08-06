"""Synthetic tests for the sealed three-model comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.final_evaluation as final_evaluation
from qoe_twin.features import get_network_feature_names


class _Persistence:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        poor = frame["current_mos"].lt(3.0).to_numpy(dtype=float)
        return np.column_stack([1.0 - poor, poor])


class _NetworkLogistic:
    def __init__(self, probability: list[float]) -> None:
        self.probability = np.asarray(probability, dtype=float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        assert list(frame.columns) == get_network_feature_names()
        return np.column_stack([1.0 - self.probability, self.probability])


class _IdentityCalibrator:
    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.asarray(probability, dtype=float)


def _comparison_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "session_id": ["s-1", "s-1", "s-2"],
            "user_id": [1, 1, 2],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01T00:00:00",
                    "2025-01-01T00:00:01",
                    "2025-01-01T00:00:02",
                ]
            ),
            "actual_future_poor_qoe": [0, 1, 1],
            "current_mos": [4.0, 2.0, 4.0],
            "calibrated_probability": [0.1, 0.8, 0.4],
            "predicted_poor_qoe": [0, 1, 0],
            "decision_threshold": [0.5, 0.5, 0.5],
        }
    )
    for feature in get_network_feature_names():
        if feature not in frame:
            frame[feature] = 1.0
    return frame


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_network_logistic_selection_resolves_exact_frozen_calibrator(
    tmp_path: Path,
    method: str,
) -> None:
    configuration_hash = "a" * 64
    selection_path = tmp_path / "selected.json"
    selection_path.write_text(
        json.dumps(
            {
                "configuration_sha256": configuration_hash,
                "network_logistic": {
                    "calibration_method": method,
                    "decision_threshold": 0.37,
                },
            }
        ),
        encoding="utf-8",
    )
    calibrator_path = (
        tmp_path / f"network_logistic_{method}_final.joblib"
    )
    calibrator_path.write_bytes(b"synthetic calibrator")

    selected_method, selected_path, threshold = (
        final_evaluation.load_network_logistic_selection(
            selection_path,
            tmp_path,
            expected_configuration_sha256=configuration_hash,
        )
    )

    assert selected_method == method
    assert selected_path == calibrator_path
    assert threshold == pytest.approx(0.37)


def test_network_logistic_selection_rejects_unfrozen_configuration(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selected.json"
    selection_path.write_text(
        json.dumps(
            {
                "configuration_sha256": "b" * 64,
                "network_logistic": {
                    "calibration_method": "sigmoid",
                    "decision_threshold": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        final_evaluation.load_network_logistic_selection(
            selection_path,
            tmp_path,
            expected_configuration_sha256="a" * 64,
        )


def test_evaluation_dataset_reads_only_exact_sealed_test_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _comparison_frame()
    predictions = frame[
        [
            *final_evaluation.ROW_KEY_COLUMNS,
            "actual_future_poor_qoe",
            "calibrated_probability",
            "predicted_poor_qoe",
            "decision_threshold",
        ]
    ].copy()
    predictions["abstain"] = False
    predictions["trust_level"] = "high"

    features = frame.drop(
        columns=[
            "actual_future_poor_qoe",
            "calibrated_probability",
            "predicted_poor_qoe",
            "decision_threshold",
        ]
    ).copy()
    features["split"] = "test"
    features["future_poor_qoe"] = [0, 1, 1]
    features["base_station"] = "BS1"
    features["prb"] = 25
    features["bitrate_kbps"] = 2_000
    features["resolution"] = "720p"
    features["throughput_mbps"] = 5.0
    features["plr_percent"] = 0.0
    features["prediction_lead_seconds"] = 1.0
    features["throughput_to_bitrate_ratio"] = 2.5
    features["capacity_margin_mbps"] = 3.0
    features["mos_change_1"] = 0.0
    features["throughput_change_1"] = 0.0

    calls: list[tuple[Path, dict[str, Any]]] = []

    def read_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
        calls.append((path, kwargs))
        if path == final_evaluation.TRUSTED_PREDICTION_PATH:
            return predictions.copy()
        assert path == final_evaluation.FEATURE_PATH
        return features.loc[:, kwargs["columns"]].copy()

    monkeypatch.setattr(final_evaluation.pd, "read_parquet", read_parquet)

    loaded = final_evaluation.load_evaluation_dataset()

    assert len(loaded) == 3
    assert calls[1][1]["filters"] == [("split", "==", "test")]
    assert loaded["actual_future_poor_qoe"].tolist() == [0, 1, 1]


def test_three_models_use_identical_test_rows_and_exact_metrics() -> None:
    predictions = final_evaluation.build_three_model_test_predictions(
        _comparison_frame(),
        persistence_model=_Persistence(),
        network_logistic_model=_NetworkLogistic([0.2, 0.8, 0.7]),
        network_logistic_calibrator=_IdentityCalibrator(),
        network_logistic_method="sigmoid",
        network_logistic_threshold=0.6,
        random_forest_method="isotonic",
        random_forest_threshold=0.5,
    )
    metrics = final_evaluation.evaluate_three_model_test_predictions(
        predictions
    ).set_index("model")

    assert len(predictions) == 9
    for _, model_predictions in predictions.groupby("model"):
        assert model_predictions[
            final_evaluation.ROW_KEY_COLUMNS
        ].reset_index(drop=True).equals(
            _comparison_frame()[
                final_evaluation.ROW_KEY_COLUMNS
            ].reset_index(drop=True)
        )
        assert model_predictions["actual_future_poor_qoe"].tolist() == [0, 1, 1]

    assert metrics.loc["network_logistic", "observations"] == 3
    assert metrics.loc["network_logistic", "accuracy"] == pytest.approx(1.0)
    assert metrics.loc["network_logistic", "f1"] == pytest.approx(1.0)
    assert metrics.loc["persistence", "accuracy"] == pytest.approx(2 / 3)
    assert metrics.loc["persistence", "f1"] == pytest.approx(2 / 3)
    assert metrics.loc[
        "cross_layer_random_forest", "accuracy"
    ] == pytest.approx(2 / 3)


def test_reliability_figure_uses_selected_calibration_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str | None] = []

    class _Axis:
        def plot(self, *_: Any, **kwargs: Any) -> None:
            labels.append(kwargs.get("label"))

        def set_xlabel(self, _: str) -> None:
            pass

        def set_ylabel(self, _: str) -> None:
            pass

        def set_title(self, _: str) -> None:
            pass

        def set_xlim(self, *_: float) -> None:
            pass

        def set_ylim(self, *_: float) -> None:
            pass

        def legend(self) -> None:
            pass

    class _Figure:
        def tight_layout(self) -> None:
            pass

        def savefig(self, *_: Any, **__: Any) -> None:
            pass

    figure = _Figure()
    axis = _Axis()
    monkeypatch.setattr(
        final_evaluation.plt,
        "subplots",
        lambda **_: (figure, axis),
    )
    monkeypatch.setattr(final_evaluation.plt, "close", lambda _: None)
    monkeypatch.setattr(final_evaluation, "FIGURE_DIRECTORY", tmp_path)

    final_evaluation.save_reliability_diagram(
        pd.DataFrame(
            {
                "mean_probability": [0.2, 0.8],
                "observed_frequency": [0.1, 0.9],
            }
        ),
        "sigmoid",
    )

    assert labels[1] == "Sigmoid calibrated model"
