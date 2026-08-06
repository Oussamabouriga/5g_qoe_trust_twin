"""Generate grounded OpenAI explanations for diverse final test cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from qoe_twin.artifact_lineage import (
    load_resolved_configuration,
    load_selected_final_calibrator,
    validate_prediction_chain,
)
from qoe_twin.evidence_builder import build_evidence
from qoe_twin.explanation_schema import Explanation
from qoe_twin.features import get_cross_layer_feature_names
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
OFFLINE_PROMPT_PATH = (
    OUTPUT_DIRECTORY / "explanation_prompts.jsonl"
)
COMPLETED_PROMPT_PATH = (
    OUTPUT_DIRECTORY / "final_explanations.jsonl"
)
MANUAL_REVIEW_PATH = (
    OUTPUT_DIRECTORY / "manual_review_template.csv"
)
CONFIG_DIRECTORY = Path("configs")
SELECTION_PATH = Path(
    "results/metrics/selected_calibration_configuration_final.json"
)
MODEL_PATH = Path(
    "models/uncalibrated/"
    "cross_layer_random_forest_final.joblib"
)
CALIBRATED_DIRECTORY = Path("models/calibrated")
ROW_ID_COLUMNS = [
    "session_id",
    "user_id",
    "timestamp",
]


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

    parser.add_argument(
        "--offline-prompt-export",
        action="store_true",
        help=(
            "Export the fixed review prompt batch without constructing "
            "an API client."
        ),
    )

    parser.add_argument(
        "--import-completed-jsonl",
        action="store_true",
        help=(
            "Validate the completed explanation JSONL and create the review CSV."
        ),
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


def select_review_cases(
    frame: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Select the fixed 10/10/10 decision strata for human review."""
    required_columns = {
        *ROW_ID_COLUMNS,
        "abstain",
        "predicted_poor_qoe",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(
            "Explanation case selection is missing required columns: "
            + ", ".join(missing)
        )
    if frame[list(required_columns)].isna().any().any():
        raise ValueError(
            "Explanation case selection requires non-null row identities and decisions."
        )

    predicted = frame["predicted_poor_qoe"].astype(int)
    if not predicted.isin([0, 1]).all():
        raise ValueError(
            "predicted_poor_qoe must contain only zero or one."
        )
    abstained = frame["abstain"].astype(bool)
    category_masks = (
        (
            "accepted_poor_qoe",
            ~abstained & predicted.eq(1),
        ),
        (
            "accepted_acceptable_qoe",
            ~abstained & predicted.eq(0),
        ),
        (
            "abstention",
            abstained,
        ),
    )

    selected_parts: list[pd.DataFrame] = []
    for category, mask in category_masks:
        candidates = frame.loc[mask].sort_values(
            ROW_ID_COLUMNS,
            kind="mergesort",
        )
        if len(candidates) < 10:
            raise ValueError(
                f"Review category {category!r} has {len(candidates)} rows; "
                "at least 10 are required."
            )
        selected = candidates.sample(
            n=10,
            random_state=random_seed,
        ).sort_values(
            ROW_ID_COLUMNS,
            kind="mergesort",
        )
        selected = selected.copy()
        selected["category"] = category
        selected_parts.append(selected)

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )


def build_offline_prompt_records(
    selected: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build prompt-only records through the configured evidence path."""
    records: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        evidence = build_row_evidence(row)
        records.append(
            {
                "case_id": f"case_{index:03d}",
                "category": str(row["category"]),
                "session_id": str(row["session_id"]),
                "user_id": int(row["user_id"]),
                "timestamp": row["timestamp"],
                "prediction": evidence["prediction"],
                "probability": evidence[
                    "prediction_probability"
                ],
                "trust_score": evidence["trust_score"],
                "abstain": bool(row["abstain"]),
                "evidence": evidence,
                "prompt": build_prompt(evidence),
                "explanation": None,
            }
        )
    return records


def write_offline_prompt_export(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write deterministic one-record-per-line JSON for external completion."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    lines = [
        json.dumps(
            record,
            default=json_default,
            sort_keys=True,
            allow_nan=False,
        )
        for record in records
    ]
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_jsonl_records(
    path: Path,
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Read a JSONL batch and reject malformed or non-object records."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required {label} JSONL file is missing: {path}"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {label} line {line_number}: {error}"
            ) from error
        if type(record) is not dict:
            raise ValueError(
                f"{label} line {line_number} must contain one JSON object."
            )
        records.append(record)
    return records


def _schema_error_messages(
    error: ValidationError,
) -> list[str]:
    messages: list[str] = []
    for detail in error.errors(
        include_url=False,
        include_input=False,
    ):
        location = ".".join(
            str(part)
            for part in detail.get("loc", ())
        )
        prefix = (
            f"Schema validation at {location}: "
            if location
            else "Schema validation: "
        )
        messages.append(
            prefix + str(detail["msg"])
        )
    return messages


def import_completed_explanations(
    prompt_path: Path,
    completed_path: Path,
    review_path: Path,
) -> dict[str, int]:
    """Validate returned explanations and create a blank human-review sheet."""
    prompts = read_jsonl_records(
        prompt_path,
        label="offline prompt export",
    )
    completed = read_jsonl_records(
        completed_path,
        label="completed explanation",
    )
    if len(prompts) != 30 or len(completed) != 30:
        raise ValueError(
            "Explanation import requires exactly 30 prompt and completed records."
        )

    expected_case_ids = [
        f"case_{index:03d}"
        for index in range(1, 31)
    ]
    if [record.get("case_id") for record in prompts] != expected_case_ids:
        raise ValueError(
            "Offline prompt export case IDs or order are not canonical."
        )
    if [record.get("case_id") for record in completed] != expected_case_ids:
        raise ValueError(
            "Completed explanation case IDs or order do not match the export."
        )

    validator = GroundingValidator()
    review_rows: list[dict[str, Any]] = []
    schema_valid_cases = 0
    grounding_valid_cases = 0

    for prompt_record, completed_record in zip(
        prompts,
        completed,
        strict=True,
    ):
        prompt_fields = {
            key: value
            for key, value in prompt_record.items()
            if key != "explanation"
        }
        completed_fields = {
            key: value
            for key, value in completed_record.items()
            if key != "explanation"
        }
        case_id = str(prompt_record["case_id"])
        if completed_fields != prompt_fields:
            raise ValueError(
                f"Completed record {case_id} changed exported fields other than "
                "explanation."
            )

        explanation_payload = completed_record.get(
            "explanation"
        )
        validation_errors: list[str] = []
        try:
            explanation = Explanation.model_validate(
                explanation_payload
            )
        except ValidationError as error:
            validation_errors.extend(
                _schema_error_messages(error)
            )
        else:
            schema_valid_cases += 1
            grounding_valid, grounding_errors = (
                validator.validate(
                    explanation,
                    completed_record["evidence"],
                )
            )
            validation_errors.extend(
                grounding_errors
            )
            if grounding_valid:
                grounding_valid_cases += 1

        review_rows.append(
            {
                "case_id": case_id,
                "prediction": completed_record[
                    "prediction"
                ],
                "evidence_json": json.dumps(
                    completed_record["evidence"],
                    sort_keys=True,
                    allow_nan=False,
                ),
                "prompt": completed_record["prompt"],
                "explanation_json": json.dumps(
                    explanation_payload,
                    sort_keys=True,
                    allow_nan=False,
                ),
                "automatic_validation_errors": json.dumps(
                    validation_errors,
                    ensure_ascii=False,
                ),
                "reviewer_id": "",
                "correctness": "",
                "completeness": "",
                "unsupported_statement_present": "",
                "contradiction_present": "",
                "review_notes": "",
            }
        )

    review_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pd.DataFrame(review_rows).to_csv(
        review_path,
        index=False,
        lineterminator="\n",
    )
    return {
        "total_cases": len(review_rows),
        "schema_valid_cases": schema_valid_cases,
        "grounding_valid_cases": grounding_valid_cases,
        "automatic_rejection_cases": sum(
            bool(
                json.loads(
                    row["automatic_validation_errors"]
                )
            )
            for row in review_rows
        ),
    }


def validate_frozen_prediction_chain() -> None:
    """Authenticate the exact frozen prediction chain before reading it."""
    configuration = load_resolved_configuration(
        CONFIG_DIRECTORY
    )
    _, calibrator_path, _ = load_selected_final_calibrator(
        SELECTION_PATH,
        CALIBRATED_DIRECTORY,
        configuration=configuration,
    )
    categorical_features = ["resolution"]
    numeric_features = [
        feature
        for feature in get_cross_layer_feature_names()
        if feature not in categorical_features
    ]
    validate_prediction_chain(
        MODEL_PATH,
        calibrator_path,
        PREDICTION_PATH,
        configuration=configuration,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


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
    }

    return build_evidence(raw)


def main() -> None:
    load_dotenv()
    args = parse_arguments()

    if (
        args.offline_prompt_export
        and args.import_completed_jsonl
    ):
        raise ValueError(
            "Choose either offline prompt export or completed JSONL import."
        )
    if args.import_completed_jsonl:
        summary = import_completed_explanations(
            OFFLINE_PROMPT_PATH,
            COMPLETED_PROMPT_PATH,
            MANUAL_REVIEW_PATH,
        )
        print("Explanation batch automatic validation:")
        for key, value in summary.items():
            print(f"- {key}: {value}")
        print(f"Saved manual-review CSV to {MANUAL_REVIEW_PATH}")
        return

    validate_frozen_prediction_chain()

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
    ]

    print(f"Reading {FEATURE_PATH}")
    features = pd.read_parquet(
        FEATURE_PATH,
        columns=feature_columns,
        filters=[("split", "==", "test")],
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

    if (
        len(frame) != len(predictions)
        or len(frame) != len(features)
    ):
        raise ValueError(
            "Frozen predictions and sealed test features do not have exact "
            "one-to-one row identities."
        )

    if args.offline_prompt_export:
        if args.cases != 30:
            raise ValueError(
                "Offline explanation export requires exactly 30 cases."
            )
        selected = select_review_cases(
            frame=frame,
            random_seed=args.seed,
        )
        records = build_offline_prompt_records(
            selected
        )
        write_offline_prompt_export(
            records,
            OFFLINE_PROMPT_PATH,
        )
        print(
            f"Exported {len(records)} offline prompts to "
            f"{OFFLINE_PROMPT_PATH}"
        )
        return

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
            "prompt": prompt,
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
                "prediction": record["evidence"][
                    "prediction"
                ],
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
                "prompt": record["prompt"],
                "explanation_json": json.dumps(
                    record["explanation"]
                ),
                "automatic_validation_errors": (
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
