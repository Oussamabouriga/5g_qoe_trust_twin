"""Generate grounded OpenAI explanations for diverse final test cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from qoe_twin.evidence_builder import build_evidence
from qoe_twin.grounding_validator import GroundingValidator
from qoe_twin.openai_client import OpenAIExplanationClient
from qoe_twin.prompt_builder import build_prompt


PREDICTION_PATH = Path(
    "results/predictions/final_trusted_predictions.parquet"
)

FEATURE_PATH = Path(
    "data/processed/qoe_features_final.parquet"
)

OUTPUT_DIRECTORY = Path("results/explanations")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=int,
        default=30,
        help="Number of explanations to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    raise TypeError(
        f"Object of type {type(value)} is not JSON serializable."
    )


def select_diverse_cases(
    frame: pd.DataFrame,
    number_of_cases: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample across trust level, prediction and actual class."""
    grouping_columns = [
        "trust_level",
        "predicted_poor_qoe",
        "actual_future_poor_qoe",
    ]

    groups = list(
        frame.groupby(
            grouping_columns,
            dropna=False,
        )
    )

    if not groups:
        raise ValueError(
            "No prediction groups are available."
        )

    cases_per_group = max(
        number_of_cases // len(groups),
        1,
    )

    selected_parts: list[pd.DataFrame] = []

    for _, group in groups:
        selected_parts.append(
            group.sample(
                n=min(
                    cases_per_group,
                    len(group),
                ),
                random_state=random_seed,
            )
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            "session_id",
            "user_id",
            "timestamp",
        ]
    )

    if len(selected) < number_of_cases:
        remaining = frame.merge(
            selected[
                [
                    "session_id",
                    "user_id",
                    "timestamp",
                ]
            ],
            on=[
                "session_id",
                "user_id",
                "timestamp",
            ],
            how="left",
            indicator=True,
        )

        remaining = remaining[
            remaining["_merge"].eq("left_only")
        ].drop(columns="_merge")

        additional_count = min(
            number_of_cases - len(selected),
            len(remaining),
        )

        if additional_count > 0:
            additional = remaining.sample(
                n=additional_count,
                random_state=random_seed + 1,
            )

            selected = pd.concat(
                [selected, additional],
                ignore_index=True,
            )

    return selected.head(
        number_of_cases
    ).reset_index(drop=True)


def build_row_evidence(
    row: pd.Series,
) -> dict[str, Any]:
    """Map final prediction columns to the LLM evidence schema."""
    if bool(row["abstain"]):
        prediction = "abstain"
    elif int(row["predicted_poor_qoe"]) == 1:
        prediction = "future_poor_qoe"
    else:
        prediction = "future_acceptable_qoe"

    raw = {
        "prediction": prediction,
        "prediction_probability": round(
            float(row["calibrated_probability"]),
            4,
        ),
        "trust_score": round(
            float(row["trust_score"]),
            4,
        ),
        "trust_level": str(row["trust_level"]),
        "current_mos": round(
            float(row["current_mos"]),
            4,
        ),
        "throughput_mbps": round(
            float(row["throughput_mbps"]),
            4,
        ),
        "bitrate_kbps": int(
            row["bitrate_kbps"]
        ),
        "capacity_margin_mbps": round(
            float(row["capacity_margin_mbps"]),
            4,
        ),
        "throughput_to_bitrate_ratio": round(
            float(
                row[
                    "throughput_to_bitrate_ratio"
                ]
            ),
            4,
        ),
        "plr_percent": round(
            float(row["plr_percent"]),
            4,
        ),
        "lead_time_seconds": round(
            float(row["prediction_lead_seconds"]),
            4,
        ),
    }

    return build_evidence(raw)


def main() -> None:
    load_dotenv()
    args = parse_arguments()

    print(f"Reading {PREDICTION_PATH}")
    predictions = pd.read_parquet(
        PREDICTION_PATH
    )

    feature_columns = [
        "session_id",
        "user_id",
        "timestamp",
        "current_mos",
        "throughput_mbps",
        "bitrate_kbps",
        "capacity_margin_mbps",
        "throughput_to_bitrate_ratio",
        "plr_percent",
        "prediction_lead_seconds",
    ]

    print(f"Reading {FEATURE_PATH}")
    features = pd.read_parquet(
        FEATURE_PATH,
        columns=feature_columns,
    )

    frame = predictions.merge(
        features,
        on=[
            "session_id",
            "user_id",
            "timestamp",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )

    if "prediction_lead_seconds_feature" in frame.columns:
        frame["prediction_lead_seconds"] = frame[
            "prediction_lead_seconds_feature"
        ]

    selected = select_diverse_cases(
        frame=frame,
        number_of_cases=args.cases,
        random_seed=args.seed,
    )

    print(
        f"Selected {len(selected)} diverse cases."
    )

    client = OpenAIExplanationClient()
    validator = GroundingValidator()

    records: list[dict[str, Any]] = []

    for index, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        case_id = f"case_{index:03d}"
        evidence = build_row_evidence(row)
        prompt = build_prompt(evidence)

        started_at = time.perf_counter()

        explanation_data: dict[str, Any] | None = None
        errors: list[str] = []
        schema_valid = False
        grounding_valid = False
        status = "rejected"

        try:
            explanation = client.generate_json(
                prompt
            )

            schema_valid = True

            grounding_valid, errors = (
                validator.validate(
                    explanation,
                    evidence,
                )
            )

            explanation_data = (
                explanation.model_dump()
            )

            status = (
                "accepted"
                if grounding_valid
                else "rejected"
            )

        except Exception as error:
            errors = [
                f"{type(error).__name__}: {error}"
            ]

        latency_seconds = (
            time.perf_counter()
            - started_at
        )

        record = {
            "case_id": case_id,
            "session_id": str(
                row["session_id"]
            ),
            "user_id": int(
                row["user_id"]
            ),
            "timestamp": row["timestamp"],
            "actual_future_poor_qoe": int(
                row["actual_future_poor_qoe"]
            ),
            "abstain": bool(row["abstain"]),
            "evidence": evidence,
            "explanation": explanation_data,
            "schema_valid": schema_valid,
            "grounding_valid": grounding_valid,
            "status": status,
            "validation_errors": errors,
            "latency_seconds": latency_seconds,
        }

        records.append(record)

        print(
            f"[{index:02d}/{len(selected):02d}] "
            f"{case_id}: {status} "
            f"({latency_seconds:.2f}s)"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_DIRECTORY
        / "final_explanations.json"
    )

    parquet_path = (
        OUTPUT_DIRECTORY
        / "final_explanations.parquet"
    )

    metrics_path = (
        OUTPUT_DIRECTORY
        / "llm_evaluation.json"
    )

    json_path.write_text(
        json.dumps(
            records,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    flat_records = []

    for record in records:
        flat_records.append(
            {
                "case_id": record["case_id"],
                "session_id": record["session_id"],
                "user_id": record["user_id"],
                "timestamp": record["timestamp"],
                "actual_future_poor_qoe": (
                    record[
                        "actual_future_poor_qoe"
                    ]
                ),
                "abstain": record["abstain"],
                "schema_valid": (
                    record["schema_valid"]
                ),
                "grounding_valid": (
                    record["grounding_valid"]
                ),
                "status": record["status"],
                "latency_seconds": (
                    record["latency_seconds"]
                ),
                "evidence_json": json.dumps(
                    record["evidence"]
                ),
                "explanation_json": json.dumps(
                    record["explanation"]
                ),
                "validation_errors_json": (
                    json.dumps(
                        record[
                            "validation_errors"
                        ]
                    )
                ),
            }
        )

    pd.DataFrame(
        flat_records
    ).to_parquet(
        parquet_path,
        index=False,
    )

    total = len(records)
    schema_passes = sum(
        record["schema_valid"]
        for record in records
    )
    grounding_passes = sum(
        record["grounding_valid"]
        for record in records
    )
    accepted = sum(
        record["status"] == "accepted"
        for record in records
    )

    metrics = {
        "requested_cases": args.cases,
        "generated_cases": total,
        "schema_valid_cases": schema_passes,
        "schema_valid_rate": (
            schema_passes / total
            if total
            else 0.0
        ),
        "grounding_valid_cases": (
            grounding_passes
        ),
        "grounding_pass_rate": (
            grounding_passes / total
            if total
            else 0.0
        ),
        "accepted_cases": accepted,
        "rejected_cases": total - accepted,
        "acceptance_rate": (
            accepted / total
            if total
            else 0.0
        ),
        "mean_latency_seconds": float(
            pd.Series(
                [
                    record["latency_seconds"]
                    for record in records
                ]
            ).mean()
        ),
        "median_latency_seconds": float(
            pd.Series(
                [
                    record["latency_seconds"]
                    for record in records
                ]
            ).median()
        ),
    }

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nLLM BATCH EVALUATION")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(f"- {json_path}")
    print(f"- {parquet_path}")
    print(f"- {metrics_path}")


if __name__ == "__main__":
    main()
