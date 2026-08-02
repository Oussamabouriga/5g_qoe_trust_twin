"""Analyze temporal gaps and potential session boundaries in 5G-QoERA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_DIRECTORY = Path("data/external/5G-QoERA/5G-QoERA")
OUTPUT_DIRECTORY = Path("results/tables")

CANDIDATE_THRESHOLDS = [30, 60, 120, 300]


def analyze_file(path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    """Analyze timestamp gaps for one dataset file."""
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["User_ID", "Timestamp"],
    )

    frame["Timestamp"] = pd.to_datetime(
        frame["Timestamp"],
        errors="raise",
        utc=True,
    )

    frame = frame.sort_values(
        ["User_ID", "Timestamp"]
    ).reset_index(drop=True)

    frame["gap_seconds"] = (
        frame.groupby("User_ID")["Timestamp"]
        .diff()
        .dt.total_seconds()
    )

    valid_gaps = frame["gap_seconds"].dropna()

    summary: dict[str, object] = {
        "filename": path.name,
        "rows": len(frame),
        "users": frame["User_ID"].nunique(),
        "gap_count": len(valid_gaps),
        "minimum_gap_seconds": valid_gaps.min(),
        "p25_gap_seconds": valid_gaps.quantile(0.25),
        "median_gap_seconds": valid_gaps.median(),
        "p75_gap_seconds": valid_gaps.quantile(0.75),
        "p90_gap_seconds": valid_gaps.quantile(0.90),
        "p95_gap_seconds": valid_gaps.quantile(0.95),
        "p99_gap_seconds": valid_gaps.quantile(0.99),
        "maximum_gap_seconds": valid_gaps.max(),
    }

    for threshold in CANDIDATE_THRESHOLDS:
        session_starts = (
            frame["gap_seconds"].isna()
            | (frame["gap_seconds"] > threshold)
        )

        session_count = int(session_starts.sum())

        summary[f"sessions_gap_{threshold}s"] = session_count
        summary[f"breaks_gap_{threshold}s"] = int(
            (frame["gap_seconds"] > threshold).sum()
        )

    return summary, frame


def main() -> None:
    """Analyze unique mobility traces without repeating every PRB scenario."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Timestamp trajectories are identical across PRBs.
    # Analyze one PRB file per base station.
    files = sorted(
        DATASET_DIRECTORY.glob(
            "MOS_BS_*_Driver_PRB_5.tsv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No PRB 5 files were found."
        )

    summaries = []
    gap_frames = []

    for path in files:
        print(f"Analyzing {path.name}")

        summary, frame = analyze_file(path)
        summaries.append(summary)

        frame = frame[
            ["User_ID", "Timestamp", "gap_seconds"]
        ].copy()

        frame["filename"] = path.name
        gap_frames.append(frame)

    summary_frame = pd.DataFrame(summaries)

    summary_path = (
        OUTPUT_DIRECTORY
        / "session_gap_summary.csv"
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    all_gaps = pd.concat(
        gap_frames,
        ignore_index=True,
    )

    valid_gaps = all_gaps["gap_seconds"].dropna()

    gap_distribution = pd.DataFrame(
        {
            "statistic": [
                "minimum",
                "p25",
                "median",
                "p75",
                "p90",
                "p95",
                "p99",
                "maximum",
            ],
            "gap_seconds": [
                valid_gaps.min(),
                valid_gaps.quantile(0.25),
                valid_gaps.median(),
                valid_gaps.quantile(0.75),
                valid_gaps.quantile(0.90),
                valid_gaps.quantile(0.95),
                valid_gaps.quantile(0.99),
                valid_gaps.max(),
            ],
        }
    )

    distribution_path = (
        OUTPUT_DIRECTORY
        / "global_gap_distribution.csv"
    )

    gap_distribution.to_csv(
        distribution_path,
        index=False,
    )

    print("\nSESSION GAP SUMMARY")
    print("=" * 100)
    print(summary_frame.to_string(index=False))

    print("\nGLOBAL GAP DISTRIBUTION")
    print("=" * 50)
    print(gap_distribution.to_string(index=False))

    print("\nBREAK COUNTS BY THRESHOLD")
    print("=" * 50)

    for threshold in CANDIDATE_THRESHOLDS:
        breaks = int(
            summary_frame[
                f"breaks_gap_{threshold}s"
            ].sum()
        )

        sessions = int(
            summary_frame[
                f"sessions_gap_{threshold}s"
            ].sum()
        )

        print(
            f"Gap > {threshold:>3}s: "
            f"{breaks:,} breaks, "
            f"{sessions:,} sessions"
        )

    print("\nGenerated:")
    print(f"- {summary_path}")
    print(f"- {distribution_path}")


if __name__ == "__main__":
    main()
