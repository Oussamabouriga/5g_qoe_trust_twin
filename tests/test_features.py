"""Tests for leakage-safe temporal features."""

import pandas as pd

from qoe_twin.features import build_temporal_features


def create_test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "base_station": [1, 1, 1],
            "prb": [10, 10, 10],
            "bitrate_kbps": [4000, 4000, 4000],
            "user_id": [1, 1, 1],
            "session_id": ["s1", "s1", "s1"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:08Z",
                    "2026-01-01T00:00:16Z",
                ]
            ),
            "throughput_mbps": [2.0, 4.0, 8.0],
            "latitude": [48.0, 48.0001, 48.0002],
            "longitude": [2.0, 2.0001, 2.0002],
            "gap_seconds": [float("nan"), 8.0, 8.0],
            "plr_percent": [20.0, 10.0, 0.0],
            "current_mos": [2.0, 3.0, 4.0],
        }
    )


def test_lag_uses_previous_observation() -> None:
    featured = build_temporal_features(
        create_test_frame()
    )

    assert pd.isna(featured.loc[0, "throughput_lag1"])
    assert featured.loc[1, "throughput_lag1"] == 2.0
    assert featured.loc[2, "throughput_lag1"] == 4.0


def test_past_mean_excludes_current_observation() -> None:
    featured = build_temporal_features(
        create_test_frame()
    )

    assert featured.loc[1, "throughput_past_mean_3"] == 2.0
    assert featured.loc[2, "throughput_past_mean_3"] == 3.0


def test_capacity_ratio() -> None:
    featured = build_temporal_features(
        create_test_frame()
    )

    assert featured.loc[0, "throughput_to_bitrate_ratio"] == 0.5
    assert featured.loc[2, "throughput_to_bitrate_ratio"] == 2.0
