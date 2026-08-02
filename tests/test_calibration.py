"""Tests for calibration utilities."""

import numpy as np

from qoe_twin.calibration import (
    ProbabilityCalibrator,
    calibration_table,
    expected_calibration_error,
)


def test_sigmoid_calibration_returns_probabilities() -> None:
    raw_probability = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )
    target = np.array([0, 0, 1, 1])

    calibrator = ProbabilityCalibrator(
        method="sigmoid"
    )

    calibrator.fit(
        raw_probability,
        target,
    )

    calibrated = calibrator.predict(
        raw_probability
    )

    assert calibrated.shape == (4,)
    assert np.all(calibrated >= 0)
    assert np.all(calibrated <= 1)


def test_isotonic_calibration_returns_probabilities() -> None:
    raw_probability = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )
    target = np.array([0, 0, 1, 1])

    calibrator = ProbabilityCalibrator(
        method="isotonic"
    )

    calibrator.fit(
        raw_probability,
        target,
    )

    calibrated = calibrator.predict(
        raw_probability
    )

    assert calibrated.shape == (4,)
    assert np.all(calibrated >= 0)
    assert np.all(calibrated <= 1)


def test_expected_calibration_error_is_bounded() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    error = expected_calibration_error(
        target,
        probability,
    )

    assert 0.0 <= error <= 1.0


def test_calibration_table_contains_observations() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    table = calibration_table(
        target,
        probability,
    )

    assert table["count"].sum() == 4
