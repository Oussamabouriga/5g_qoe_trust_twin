"""Focused synthetic checks for human-review aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.evaluate_llm as evaluation
from qoe_twin.prompt_builder import build_prompt


def _write_review_inputs(
    root: Path,
) -> tuple[Path, Path]:
    explanation_path = root / "completed.jsonl"
    review_path = root / "review.csv"
    explanation_records: list[dict[str, object]] = []
    review_rows: list[dict[str, str]] = []
    categories = (
        ("accepted_poor_qoe", "future_poor_qoe", "medium"),
        ("accepted_acceptable_qoe", "future_acceptable_qoe", "high"),
        ("abstention", "abstain", "low"),
    )
    for category, prediction, confidence in categories:
        for _ in range(10):
            case_id = f"case_{len(explanation_records) + 1:03d}"
            evidence = {
                "prediction": prediction,
                "prediction_probability": 0.7,
                "trust_score": 0.8,
                "trust_level": confidence,
                "current_mos": 3.5,
            }
            explanation = {
                "prediction": prediction,
                "confidence": confidence,
                "summary": "Insufficient evidence supports a specific cause.",
                "likely_causes": [],
                "recommended_operator_checks": [
                    "Review the cited operational evidence."
                ],
                "limitations": "Insufficient evidence supports a specific cause.",
            }
            record = {
                "case_id": case_id,
                "category": category,
                "prediction": prediction,
                "evidence": evidence,
                "prompt": build_prompt(evidence),
                "explanation": explanation,
            }
            explanation_records.append(record)
            review_rows.append(
                {
                    "case_id": case_id,
                    "prediction": prediction,
                    "evidence_json": json.dumps(evidence),
                    "prompt": str(record["prompt"]),
                    "explanation_json": json.dumps(explanation),
                    "automatic_validation_errors": "[]",
                    "reviewer_id": "reviewer",
                    "correctness": "1",
                    "completeness": "1",
                    "unsupported_statement_present": "0",
                    "contradiction_present": "0",
                    "review_notes": "Reviewed.",
                }
            )

    explanation_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in explanation_records
        )
        + "\n",
        encoding="utf-8",
    )
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(review_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(review_rows)
    return explanation_path, review_path


def _rewrite_reviews(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_human_review_rates_are_calculated_by_category(
    tmp_path: Path,
) -> None:
    explanation_path, review_path = _write_review_inputs(tmp_path)
    review_rows = evaluation.read_review_csv(review_path)
    review_rows[0]["correctness"] = "0"
    review_rows[1]["completeness"] = "0"
    review_rows[10]["unsupported_statement_present"] = "1"
    review_rows[20]["contradiction_present"] = "1"
    _rewrite_reviews(review_path, review_rows)

    result = evaluation.evaluate_human_review(
        explanation_path,
        review_path,
    )

    assert result["overall"]["reviewed_cases"] == 30
    assert result["overall"]["factual_correct_cases"] == 29
    assert result["overall"]["factual_correctness_rate"] == 29 / 30
    assert result["overall"]["complete_cases"] == 29
    assert result["overall"]["completeness_rate"] == 29 / 30
    assert result["overall"]["unsupported_statement_cases"] == 1
    assert result["overall"]["unsupported_statement_rate"] == 1 / 30
    assert result["overall"]["contradiction_cases"] == 1
    assert result["overall"]["contradiction_rate"] == 1 / 30
    assert result["validation"]["reviewer_count"] == 1
    assert {
        category: values["reviewed_cases"]
        for category, values in result["by_prediction_category"].items()
    } == {
        "accepted_poor_qoe": 10,
        "accepted_acceptable_qoe": 10,
        "abstention": 10,
    }
    poor = result["by_prediction_category"]["accepted_poor_qoe"]
    acceptable = result["by_prediction_category"]["accepted_acceptable_qoe"]
    abstention = result["by_prediction_category"]["abstention"]
    assert (poor["factual_correct_cases"], poor["complete_cases"]) == (9, 9)
    assert poor["factual_correctness_rate"] == 0.9
    assert poor["completeness_rate"] == 0.9
    assert acceptable["unsupported_statement_rate"] == 0.1
    assert abstention["contradiction_rate"] == 0.1

    summary_path = tmp_path / "summary.json"
    table_path = tmp_path / "summary.csv"
    evaluation.write_evaluation_outputs(
        result,
        summary_path,
        table_path,
    )
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result
    with table_path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        table_rows = list(csv.DictReader(handle))
    assert [row["prediction_category"] for row in table_rows] == [
        "overall",
        "accepted_poor_qoe",
        "accepted_acceptable_qoe",
        "abstention",
    ]
    assert table_rows[0]["factual_correctness_rate"] == "0.9667"
    assert table_rows[0]["unsupported_statement_rate"] == "0.0333"
    assert table_rows[1]["factual_correctness_rate"] == "0.9000"
    assert table_rows[2]["unsupported_statement_rate"] == "0.1000"
    assert table_rows[3]["contradiction_rate"] == "0.1000"


def test_human_review_rejects_an_empty_judgment(
    tmp_path: Path,
) -> None:
    explanation_path, review_path = _write_review_inputs(tmp_path)
    rows = evaluation.read_review_csv(review_path)
    rows[0]["correctness"] = ""
    _rewrite_reviews(review_path, rows)

    with pytest.raises(
        ValueError,
        match="human-review field correctness is empty",
    ):
        evaluation.evaluate_human_review(
            explanation_path,
            review_path,
        )


def test_human_review_rejects_category_evidence_prediction_mismatch(
    tmp_path: Path,
) -> None:
    explanation_path, review_path = _write_review_inputs(tmp_path)
    records = evaluation.read_jsonl(explanation_path)
    records[0]["evidence"]["prediction"] = "future_acceptable_qoe"
    records[0]["explanation"]["prediction"] = "future_acceptable_qoe"
    explanation_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="evidence prediction does not match its category",
    ):
        evaluation.evaluate_human_review(
            explanation_path,
            review_path,
        )
