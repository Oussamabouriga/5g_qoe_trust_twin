"""Causal parity checks for interleaved trace replay."""

from collections import defaultdict

import pandas as pd

from qoe_twin.features import (
    build_temporal_features,
    get_cross_layer_feature_names,
)
from qoe_twin.replay import TraceDrivenQoETwin


class PrefixFeatureRecordingTwin(TraceDrivenQoETwin):
    """Build features from only the active session prefix at each step."""

    def __init__(self) -> None:
        super().__init__()
        self.prefixes: dict[str, list[pd.Series]] = defaultdict(list)
        self.feature_rows: list[pd.DataFrame] = []

    def predict_current_row(self, row: pd.Series) -> None:
        session_id = str(row["session_id"])
        self.prefixes[session_id].append(row.copy())
        prefix = pd.DataFrame(self.prefixes[session_id])
        featured = build_temporal_features(prefix)
        current = featured.loc[
            featured["row_id"].eq(row["row_id"]),
            ["row_id", *get_cross_layer_feature_names()],
        ]
        self.feature_rows.append(current)


def create_interleaved_rows() -> pd.DataFrame:
    """Create shuffled rows from two temporally interleaved sessions."""
    rows = pd.DataFrame(
        {
            "row_id": ["s1-0", "s2-0", "s1-1", "s2-1", "s1-2", "s2-2"],
            "session_id": ["s1", "s2", "s1", "s2", "s1", "s2"],
            "user_id": [1, 2, 1, 2, 1, 2],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:04Z",
                    "2026-01-01T00:00:08Z",
                    "2026-01-01T00:00:12Z",
                    "2026-01-01T00:00:16Z",
                    "2026-01-01T00:00:20Z",
                ]
            ),
            "base_station": [1] * 6,
            "prb": [10] * 6,
            "latitude": [48.0, 49.0, 48.0001, 49.0001, 48.0002, 49.0002],
            "longitude": [2.0, 3.0, 2.0001, 3.0001, 2.0002, 3.0002],
            "throughput_mbps": [2.0, 9.0, 4.0, 6.0, 8.0, 3.0],
            "bitrate_kbps": [4000] * 6,
            "resolution": ["1080p"] * 6,
            "gap_seconds": [float("nan"), float("nan"), 8.0, 8.0, 8.0, 8.0],
            "plr_percent": [20.0, 1.0, 10.0, 5.0, 0.0, 12.0],
            "current_mos": [2.0, 4.5, 3.0, 3.5, 4.0, 2.5],
            "session_step": [0, 0, 1, 1, 2, 2],
            "elapsed_session_seconds": [0.0, 0.0, 8.0, 8.0, 16.0, 16.0],
        }
    )
    return rows.iloc[[4, 1, 3, 0, 5, 2]].reset_index(drop=True)


def _features_by_row(frame: pd.DataFrame) -> pd.DataFrame:
    columns = get_cross_layer_feature_names()
    return (
        build_temporal_features(frame)
        .set_index("row_id")[columns]
        .sort_index()
    )


def test_interleaved_prefix_replay_matches_causal_batch_features() -> None:
    rows = create_interleaved_rows()
    expected = _features_by_row(rows)
    twin = PrefixFeatureRecordingTwin()

    twin.replay(rows)

    replayed = (
        pd.concat(twin.feature_rows, ignore_index=True)
        .set_index("row_id")
        .sort_index()
    )
    pd.testing.assert_frame_equal(
        replayed,
        expected,
        check_dtype=False,
    )


def test_future_row_mutation_cannot_change_past_features() -> None:
    rows = create_interleaved_rows()
    original = _features_by_row(rows).loc[["s1-0", "s1-1"]]
    mutated_rows = rows.copy()
    future = mutated_rows["row_id"].eq("s1-2")
    mutated_rows.loc[future, "throughput_mbps"] = 1000.0
    mutated_rows.loc[future, "current_mos"] = 1.0
    mutated_rows.loc[future, "plr_percent"] = 99.0
    mutated_rows.loc[future, "latitude"] = 55.0
    mutated_rows.loc[future, "longitude"] = 10.0

    after_mutation = _features_by_row(mutated_rows).loc[["s1-0", "s1-1"]]

    pd.testing.assert_frame_equal(original, after_mutation)
