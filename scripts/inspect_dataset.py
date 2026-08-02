from pathlib import Path

import pandas as pd


DATASET_DIRECTORY = Path(
    "data/external/5G-QoERA/5G-QoERA"
)


def main() -> None:
    files = sorted(DATASET_DIRECTORY.glob("*.tsv"))

    if not files:
        raise FileNotFoundError(
            f"No TSV files found in {DATASET_DIRECTORY}"
        )

    print(f"Number of TSV files: {len(files)}")

    summaries = []

    for path in files:
        frame = pd.read_csv(path, sep="\t")

        timestamps = pd.to_datetime(
            frame["Timestamp"],
            errors="coerce",
            utc=True,
        )

        summaries.append(
            {
                "filename": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "users": frame["User_ID"].nunique(),
                "missing_values": int(
                    frame.isna().sum().sum()
                ),
                "duplicates": int(
                    frame.duplicated().sum()
                ),
                "start_time": timestamps.min(),
                "end_time": timestamps.max(),
                "minimum_tput": frame["Tput"].min(),
                "maximum_tput": frame["Tput"].max(),
                "mean_tput": frame["Tput"].mean(),
            }
        )

        print(
            f"Inspected {path.name}: "
            f"{len(frame):,} rows"
        )

    summary = pd.DataFrame(summaries)

    output_directory = Path("results/tables")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "dataset_file_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print("\nDataset summary")
    print("-" * 60)
    print(f"Total files: {len(summary)}")
    print(
        f"Total rows: "
        f"{summary['rows'].sum():,}"
    )
    print(
        f"Total missing values: "
        f"{summary['missing_values'].sum():,}"
    )
    print(
        f"Total duplicates: "
        f"{summary['duplicates'].sum():,}"
    )
    print(
        f"Global start: "
        f"{summary['start_time'].min()}"
    )
    print(
        f"Global end: "
        f"{summary['end_time'].max()}"
    )
    print(
        f"\nSaved summary to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
