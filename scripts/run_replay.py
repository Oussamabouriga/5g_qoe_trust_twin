"""Run chronological digital-twin replay on a prototype sample."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from qoe_twin.baselines import (
    CurrentQoEPersistenceBaseline,
)
from qoe_twin.replay import TraceDrivenQoETwin


INPUT_PATH = Path(
    "data/processed/qoe_features_prototype.parquet"
)

OUTPUT_PATH = Path(
    "results/predictions/replay_prototype_predictions.parquet"
)

REPLAY_SAMPLE_SIZE = 50_000


def main() -> None:
    """Replay a chronological test sample."""
    print(f"Reading {INPUT_PATH}")

    frame = pd.read_parquet(INPUT_PATH)

    test_frame = frame[
        frame["split"] == "test"
    ].copy()

    # Keep complete sessions until approximately the desired size.
    session_sizes = (
        test_frame.groupby("session_id")
        .size()
        .sort_index()
    )

    cumulative_rows = session_sizes.cumsum()

    selected_sessions = cumulative_rows[
        cumulative_rows <= REPLAY_SAMPLE_SIZE
    ].index

    if len(selected_sessions) == 0:
        selected_sessions = session_sizes.index[:1]

    replay_frame = test_frame[
        test_frame["session_id"].isin(
            selected_sessions
        )
    ].copy()

    replay_frame = replay_frame.sort_values(
        ["session_id", "timestamp"]
    )

    print(
        f"Replaying {len(replay_frame):,} rows "
        f"from {replay_frame['session_id'].nunique():,} sessions"
    )

    model = CurrentQoEPersistenceBaseline(
        poor_mos_threshold=3.0
    )

    twin = TraceDrivenQoETwin(
        model=model,
        feature_names=["current_mos"],
        decision_threshold=0.5,
        history_size=5,
    )

    predictions = twin.replay(replay_frame)

    valid = predictions[
        predictions["actual_future_poor_qoe"].notna()
    ].copy()

    y_true = valid[
        "actual_future_poor_qoe"
    ].astype(int)

    y_pred = valid[
        "predicted_poor_qoe"
    ].astype(int)

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nREPLAY SUMMARY")
    print("=" * 60)
    print(f"Rows replayed: {len(predictions):,}")
    print(
        f"Sessions replayed: "
        f"{replay_frame['session_id'].nunique():,}"
    )
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nConfusion matrix")
    print(confusion)

    print(f"\nPredictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
