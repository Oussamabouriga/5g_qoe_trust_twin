"""Leakage-safe temporal feature engineering for the QoE digital twin."""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088

ORDER_COLUMNS = [
    "base_station",
    "prb",
    "bitrate_kbps",
    "user_id",
    "session_id",
    "timestamp",
]


def _haversine_distance_km(
    latitude_current: pd.Series,
    longitude_current: pd.Series,
    latitude_previous: pd.Series,
    longitude_previous: pd.Series,
) -> pd.Series:
    """Calculate vectorized distance between consecutive GPS points."""
    latitude_current_rad = np.radians(latitude_current)
    longitude_current_rad = np.radians(longitude_current)
    latitude_previous_rad = np.radians(latitude_previous)
    longitude_previous_rad = np.radians(longitude_previous)

    delta_latitude = latitude_current_rad - latitude_previous_rad
    delta_longitude = longitude_current_rad - longitude_previous_rad

    haversine_value = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude_previous_rad)
        * np.cos(latitude_current_rad)
        * np.sin(delta_longitude / 2) ** 2
    )

    distance = (
        2
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(haversine_value.clip(0, 1)))
    )

    return pd.Series(distance, index=latitude_current.index)


def _past_rolling_feature(
    frame: pd.DataFrame,
    value_column: str,
    window: int,
    statistic: str,
) -> pd.Series:
    """
    Calculate a rolling statistic using observations strictly before time t.

    The shift prevents the current and future values from entering
    the historical feature.
    """
    shifted = frame.groupby(
        "session_id",
        sort=False,
    )[value_column].shift(1)

    grouped = shifted.groupby(
        frame["session_id"],
        sort=False,
    )

    rolling = grouped.rolling(
        window=window,
        min_periods=1,
    )

    if statistic == "mean":
        result = rolling.mean()
    elif statistic == "std":
        result = rolling.std()
    elif statistic == "min":
        result = rolling.min()
    elif statistic == "max":
        result = rolling.max()
    else:
        raise ValueError(
            f"Unsupported rolling statistic: {statistic}"
        )

    return result.reset_index(
        level=0,
        drop=True,
    )


def build_temporal_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build current-state and historical features without future leakage."""
    result = frame.sort_values(
        ORDER_COLUMNS
    ).reset_index(drop=True)

    session_group = result.groupby(
        "session_id",
        sort=False,
    )

    # Previous observations
    result["throughput_lag1"] = session_group[
        "throughput_mbps"
    ].shift(1)

    result["throughput_lag2"] = session_group[
        "throughput_mbps"
    ].shift(2)

    result["mos_lag1"] = session_group[
        "current_mos"
    ].shift(1)

    result["plr_lag1"] = session_group[
        "plr_percent"
    ].shift(1)

    result["latitude_lag1"] = session_group[
        "latitude"
    ].shift(1)

    result["longitude_lag1"] = session_group[
        "longitude"
    ].shift(1)

    # Changes known at current time
    result["throughput_change_1"] = (
        result["throughput_mbps"]
        - result["throughput_lag1"]
    )

    result["mos_change_1"] = (
        result["current_mos"]
        - result["mos_lag1"]
    )

    result["plr_change_1"] = (
        result["plr_percent"]
        - result["plr_lag1"]
    )

    # Historical rolling features, excluding the current row
    for window in (3, 5):
        result[f"throughput_past_mean_{window}"] = (
            _past_rolling_feature(
                result,
                "throughput_mbps",
                window,
                "mean",
            )
        )

        result[f"mos_past_mean_{window}"] = (
            _past_rolling_feature(
                result,
                "current_mos",
                window,
                "mean",
            )
        )

        result[f"plr_past_mean_{window}"] = (
            _past_rolling_feature(
                result,
                "plr_percent",
                window,
                "mean",
            )
        )

    result["throughput_past_std_5"] = (
        _past_rolling_feature(
            result,
            "throughput_mbps",
            5,
            "std",
        )
    )

    result["mos_past_std_5"] = (
        _past_rolling_feature(
            result,
            "current_mos",
            5,
            "std",
        )
    )

    # Video/network compatibility
    result["bitrate_mbps"] = (
        result["bitrate_kbps"] / 1000.0
    )

    result["throughput_to_bitrate_ratio"] = (
        result["throughput_mbps"]
        / result["bitrate_mbps"]
    )

    result["capacity_margin_mbps"] = (
        result["throughput_mbps"]
        - result["bitrate_mbps"]
    )

    result["capacity_insufficient"] = (
        result["throughput_mbps"]
        < result["bitrate_mbps"]
    ).astype("int8")

    # Mobility
    result["movement_distance_km"] = (
        _haversine_distance_km(
            result["latitude"],
            result["longitude"],
            result["latitude_lag1"],
            result["longitude_lag1"],
        )
    )

    valid_gap = result["gap_seconds"].where(
        result["gap_seconds"] > 0
    )

    result["movement_speed_kmh"] = (
        result["movement_distance_km"]
        / valid_gap
        * 3600
    )

    # Position within a replay session
    result["session_step"] = (
        result.groupby(
            "session_id",
            sort=False,
        )
        .cumcount()
        .astype("int32")
    )

    result["elapsed_session_seconds"] = (
        result["timestamp"]
        - result.groupby(
            "session_id",
            sort=False,
        )["timestamp"].transform("min")
    ).dt.total_seconds()

    return result


def get_network_feature_names() -> list[str]:
    """Features available to the network-only Logistic Regression."""
    return [
        "throughput_mbps",
        "prb",
        "base_station",
        "latitude",
        "longitude",
        "gap_seconds",
        "throughput_lag1",
        "throughput_lag2",
        "throughput_change_1",
        "throughput_past_mean_3",
        "throughput_past_mean_5",
        "throughput_past_std_5",
        "movement_distance_km",
        "movement_speed_kmh",
        "session_step",
        "elapsed_session_seconds",
    ]


def get_cross_layer_feature_names() -> list[str]:
    """Features available to the cross-layer tree model."""
    return [
        *get_network_feature_names(),
        "bitrate_kbps",
        "resolution",
        "plr_percent",
        "current_mos",
        "mos_lag1",
        "plr_lag1",
        "mos_change_1",
        "plr_change_1",
        "mos_past_mean_3",
        "mos_past_mean_5",
        "mos_past_std_5",
        "plr_past_mean_3",
        "plr_past_mean_5",
        "bitrate_mbps",
        "throughput_to_bitrate_ratio",
        "capacity_margin_mbps",
        "capacity_insufficient",
    ]
