"""Validate and summarize the completed CP9 human review."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qoe_twin.artifact_lineage import sha256_file
from qoe_twin.explanation_schema import Explanation
from qoe_twin.grounding_validator import GroundingValidator

COMPLETED_EXPLANATION_PATH = Path(
    "results/explanations/cp9_phase_a_completed.jsonl"
)
COMPLETED_REVIEW_PATH = Path(
    "results/explanations/cp9_phase_a_manual_review_completed.csv"
)
SUMMARY_PATH = Path("results/metrics/cp9_human_evaluation.json")
TABLE_PATH = Path("results/tables/cp9_human_evaluation.csv")

CATEGORY_PREDICTIONS = {
    "accepted_poor_qoe": "future_poor_qoe",
    "accepted_acceptable_qoe": "future_acceptable_qoe",
    "abstention": "abstain",
}
HUMAN_FIELDS = (
    "reviewer_id",
    "correctness",
    "completeness",
    "unsupported_statement_present",
    "contradiction_present",
    "review_notes",
)
TABLE_COLUMNS = (
    "prediction_category",
    "reviewed_cases",
    "factual_correct_cases",
    "factual_correctness_rate",
    "complete_cases",
    "completeness_rate",
    "unsupported_statement_cases",
    "unsupported_statement_rate",
    "contradiction_cases",
    "contradiction_rate",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL objects without altering their values."""
    if not path.is_file():
        raise FileNotFoundError(f"Completed explanation JSONL is missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid completed JSONL line {line_number}: {error}"
            ) from error
        if type(record) is not dict:
            raise ValueError(f"Completed JSONL line {line_number} must be an object.")
        records.append(record)
    return records


def read_review_csv(path: Path) -> list[dict[str, str]]:
    """Read review values as strings so empty judgments remain detectable."""
    if not path.is_file():
        raise FileNotFoundError(f"Completed manual-review CSV is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "case_id",
            "prediction",
            "evidence_json",
            "prompt",
            "explanation_json",
            "automatic_validation_errors",
            *HUMAN_FIELDS,
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                "Completed manual-review CSV is missing columns: "
                + ", ".join(missing)
            )
        return list(reader)


def _unique_case_map(
    records: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = record.get("case_id")
        if type(case_id) is not str or not case_id:
            raise ValueError(f"{label} contains a missing or invalid case_id.")
        if case_id in by_id:
            raise ValueError(f"{label} contains duplicate case_id {case_id!r}.")
        by_id[case_id] = record
    return by_id


def _json_cell(review: dict[str, str], field: str, case_id: str) -> Any:
    raw_value = review.get(field)
    if type(raw_value) is not str:
        raise ValueError(f"{case_id} has a missing value in {field}.")
    try:
        return json.loads(raw_value)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{case_id} has invalid JSON in {field}: {error}") from error


def _binary(value: str, field: str, case_id: str) -> int:
    normalized = value.strip().lower()
    mapping = {"0": 0, "1": 1, "false": 0, "true": 1}
    if normalized not in mapping:
        raise ValueError(f"{case_id} field {field} must be 0/1 or false/true.")
    return mapping[normalized]


def _summary(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    reviewed = len(rows)
    if not reviewed:
        raise ValueError("Every prediction category requires reviewed cases.")
    correct = sum(row["correctness"] for row in rows)
    complete = sum(row["completeness"] for row in rows)
    unsupported = sum(row["unsupported_statement_present"] for row in rows)
    contradictions = sum(row["contradiction_present"] for row in rows)
    return {
        "reviewed_cases": reviewed,
        "factual_correct_cases": correct,
        "factual_correctness_rate": correct / reviewed,
        "complete_cases": complete,
        "completeness_rate": complete / reviewed,
        "unsupported_statement_cases": unsupported,
        "unsupported_statement_rate": unsupported / reviewed,
        "contradiction_cases": contradictions,
        "contradiction_rate": contradictions / reviewed,
    }


def evaluate_human_review(
    explanation_path: Path, review_path: Path
) -> dict[str, Any]:
    """Validate the exact 30-case batch and calculate reviewer rates."""
    explanations = read_jsonl(explanation_path)
    reviews = read_review_csv(review_path)
    if len(explanations) != 30 or len(reviews) != 30:
        raise ValueError(
            "CP9 human evaluation requires exactly 30 explanation and review rows."
        )

    explanation_map = _unique_case_map(explanations, "Completed explanation JSONL")
    review_map = _unique_case_map(reviews, "Completed manual-review CSV")
    if set(explanation_map) != set(review_map):
        raise ValueError("Completed explanation and review case IDs do not match.")

    category_counts = Counter(record.get("category") for record in explanations)
    if category_counts != Counter({category: 10 for category in CATEGORY_PREDICTIONS}):
        raise ValueError(
            "CP9 categories must contain exactly 10 accepted poor-QoE, "
            "10 accepted acceptable-QoE and 10 abstention cases."
        )

    validator = GroundingValidator()
    evaluated: list[dict[str, Any]] = []
    for case_id, record in explanation_map.items():
        review = review_map[case_id]
        category = record["category"]
        expected_prediction = CATEGORY_PREDICTIONS[category]
        if record.get("prediction") != expected_prediction:
            raise ValueError(
                f"{case_id} prediction does not match category {category!r}."
            )
        evidence = record.get("evidence")
        if type(evidence) is not dict:
            raise ValueError(f"{case_id} evidence must be an object.")
        if evidence.get("prediction") != expected_prediction:
            raise ValueError(
                f"{case_id} evidence prediction does not match its category."
            )

        try:
            explanation = Explanation.model_validate(record.get("explanation"))
        except ValidationError as error:
            raise ValueError(
                f"{case_id} fails the existing explanation schema: {error}"
            ) from error
        if explanation.prediction != expected_prediction:
            raise ValueError(
                f"{case_id} explanation prediction does not match its category."
            )
        grounded, errors = validator.validate(explanation, evidence)
        if not grounded:
            raise ValueError(
                f"{case_id} fails grounding validation: " + "; ".join(errors)
            )

        empty_fields = [
            field
            for field in HUMAN_FIELDS
            if type(review.get(field)) is not str or not review[field].strip()
        ]
        if empty_fields:
            raise ValueError(
                f"{case_id} human-review field {empty_fields[0]} is empty."
            )
        if review["prediction"] != expected_prediction:
            raise ValueError(
                f"{case_id} review prediction does not match its category."
            )
        if _json_cell(review, "evidence_json", case_id) != record["evidence"]:
            raise ValueError(f"{case_id} review evidence does not match the JSONL.")
        if review["prompt"] != record["prompt"]:
            raise ValueError(f"{case_id} review prompt does not match the JSONL.")
        if _json_cell(review, "explanation_json", case_id) != record["explanation"]:
            raise ValueError(f"{case_id} review explanation does not match the JSONL.")
        if _json_cell(review, "automatic_validation_errors", case_id) != []:
            raise ValueError(f"{case_id} retains automatic validation errors.")

        evaluated.append(
            {
                "category": category,
                "reviewer_id": review["reviewer_id"].strip(),
                "correctness": _binary(review["correctness"], "correctness", case_id),
                "completeness": _binary(
                    review["completeness"], "completeness", case_id
                ),
                "unsupported_statement_present": _binary(
                    review["unsupported_statement_present"],
                    "unsupported_statement_present",
                    case_id,
                ),
                "contradiction_present": _binary(
                    review["contradiction_present"],
                    "contradiction_present",
                    case_id,
                ),
            }
        )

    by_category = {
        category: _summary(
            [row for row in evaluated if row["category"] == category]
        )
        for category in CATEGORY_PREDICTIONS
    }
    reviewer_count = len({row["reviewer_id"] for row in evaluated})
    return {
        "schema_version": 1,
        "source_artifacts": {
            "completed_explanations": {
                "path": explanation_path.as_posix(),
                "sha256": sha256_file(explanation_path),
            },
            "completed_manual_review": {
                "path": review_path.as_posix(),
                "sha256": sha256_file(review_path),
            },
        },
        "validation": {
            "matching_unique_case_ids": 30,
            "schema_valid_cases": 30,
            "grounding_valid_cases": 30,
            "completed_human_review_cases": 30,
            "reviewer_count": reviewer_count,
            "category_counts": dict(category_counts),
        },
        "overall": _summary(evaluated),
        "by_prediction_category": by_category,
        "limitations": [
            f"Human judgments were supplied by {reviewer_count} reviewer(s).",
            "No inter-rater reliability estimate is available.",
            "The balanced 30-case review is small and not prevalence weighted.",
            "The external LLM model and request metadata were not retained.",
        ],
    }


def write_evaluation_outputs(
    evaluation: dict[str, Any], summary_path: Path, table_path: Path
) -> None:
    """Write deterministic JSON and a publication-ready CSV table."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    rows = [
        {"prediction_category": "overall", **evaluation["overall"]},
        *(
            {
                "prediction_category": category,
                **evaluation["by_prediction_category"][category],
            }
            for category in CATEGORY_PREDICTIONS
        ),
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: f"{value:.4f}" if field.endswith("_rate") else value
                    for field, value in row.items()
                }
            )


def main() -> None:
    """Validate completed reviews and save the CP9 summary artifacts."""
    evaluation = evaluate_human_review(
        COMPLETED_EXPLANATION_PATH, COMPLETED_REVIEW_PATH
    )
    write_evaluation_outputs(evaluation, SUMMARY_PATH, TABLE_PATH)
    print(json.dumps(evaluation["overall"], indent=2))
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {TABLE_PATH}")


if __name__ == "__main__":
    main()
