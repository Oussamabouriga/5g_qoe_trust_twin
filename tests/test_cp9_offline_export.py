"""Focused synthetic checks for the CP9 Phase A offline prompt export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

import scripts.generate_batch_explanations as batch
from qoe_twin.prompt_builder import build_prompt


def _candidate_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    category_inputs = (
        ("poor", False, 1),
        ("acceptable", False, 0),
        ("abstain", True, 1),
    )
    for category_index, (label, abstain, prediction) in enumerate(
        category_inputs
    ):
        for row_index in range(12):
            rows.append(
                {
                    "source_id": f"{label}-{row_index:02d}",
                    "session_id": f"session-{category_index}-{row_index:02d}",
                    "user_id": category_index,
                    "timestamp": pd.Timestamp("2025-01-01")
                    + pd.Timedelta(seconds=row_index),
                    "abstain": abstain,
                    "predicted_poor_qoe": prediction,
                    "calibrated_probability": 0.8 if prediction else 0.2,
                    "trust_score": 0.7,
                    "trust_level": "medium",
                    "current_mos": 3.5,
                    "throughput_mbps": 8.0,
                    "bitrate_kbps": 4000,
                    "capacity_margin_mbps": 4.0,
                    "throughput_to_bitrate_ratio": 2.0,
                    "plr_percent": 0.1,
                }
            )
    return pd.DataFrame(rows).sample(
        frac=1.0,
        random_state=7,
    )


def test_phase_a_selection_is_exact_and_deterministic() -> None:
    frame = _candidate_frame()

    first = batch.select_phase_a_cases(frame, random_seed=42)
    second = batch.select_phase_a_cases(frame, random_seed=42)

    assert first["source_id"].tolist() == second["source_id"].tolist()
    assert first["category"].value_counts(sort=False).to_dict() == {
        "accepted_poor_qoe": 10,
        "accepted_acceptable_qoe": 10,
        "abstention": 10,
    }
    assert first["category"].tolist() == (
        ["accepted_poor_qoe"] * 10
        + ["accepted_acceptable_qoe"] * 10
        + ["abstention"] * 10
    )
    assert not first[batch.ROW_ID_COLUMNS].duplicated().any()


def test_offline_export_uses_cp7_prompt_path_and_empty_explanations(
    tmp_path: Path,
) -> None:
    selected = batch.select_phase_a_cases(
        _candidate_frame(),
        random_seed=42,
    )
    records = batch.build_offline_prompt_records(selected)
    output_path = tmp_path / "prompts.jsonl"

    batch.write_offline_prompt_export(records, output_path)

    exported = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(exported) == 30
    assert all(record["explanation"] is None for record in exported)
    assert all(
        {"session_id", "user_id", "timestamp", "probability"}
        <= record.keys()
        for record in exported
    )
    assert all(
        record["prompt"] == build_prompt(record["evidence"])
        for record in exported
    )
    assert {
        record["prediction"]
        for record in exported[:10]
    } == {"future_poor_qoe"}
    assert {
        record["prediction"]
        for record in exported[10:20]
    } == {"future_acceptable_qoe"}
    assert {
        record["prediction"]
        for record in exported[20:]
    } == {"abstain"}


def _valid_explanation(record: dict[str, object]) -> dict[str, object]:
    prediction = str(record["prediction"])
    confidence = (
        "low"
        if prediction == "abstain"
        else str(record["evidence"]["trust_level"])  # type: ignore[index]
    )
    return {
        "prediction": prediction,
        "confidence": confidence,
        "summary": "Insufficient evidence supports a specific cause.",
        "likely_causes": [],
        "recommended_operator_checks": [
            "Review the cited operational evidence."
        ],
        "limitations": "Insufficient evidence supports a specific cause.",
    }


def test_completed_import_validates_and_leaves_human_fields_empty(
    tmp_path: Path,
) -> None:
    selected = batch.select_phase_a_cases(
        _candidate_frame(),
        random_seed=42,
    )
    prompts = batch.build_offline_prompt_records(selected)
    completed = [
        {
            **record,
            "explanation": _valid_explanation(record),
        }
        for record in prompts
    ]
    prompt_path = tmp_path / "prompts.jsonl"
    completed_path = tmp_path / "completed.jsonl"
    review_path = tmp_path / "review.csv"
    batch.write_offline_prompt_export(prompts, prompt_path)
    batch.write_offline_prompt_export(completed, completed_path)

    summary = batch.import_completed_explanations(
        prompt_path,
        completed_path,
        review_path,
    )

    assert summary == {
        "total_cases": 30,
        "schema_valid_cases": 30,
        "grounding_valid_cases": 30,
        "automatic_rejection_cases": 0,
    }
    with review_path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    human_fields = {
        "reviewer_id",
        "correctness",
        "completeness",
        "unsupported_statement_present",
        "contradiction_present",
        "review_notes",
    }
    assert all(
        all(row[field] == "" for field in human_fields)
        for row in rows
    )
    assert all(
        json.loads(row["automatic_validation_errors"]) == []
        for row in rows
    )


def test_completed_import_records_schema_errors_without_human_judgment(
    tmp_path: Path,
) -> None:
    selected = batch.select_phase_a_cases(
        _candidate_frame(),
        random_seed=42,
    )
    prompts = batch.build_offline_prompt_records(selected)
    completed = [
        {
            **record,
            "explanation": _valid_explanation(record),
        }
        for record in prompts
    ]
    completed[-1]["explanation"] = {
        **completed[-1]["explanation"],  # type: ignore[dict-item]
        "summary": "The system withheld its decision.",
        "limitations": "No diagnosis is provided.",
    }
    prompt_path = tmp_path / "prompts.jsonl"
    completed_path = tmp_path / "completed.jsonl"
    review_path = tmp_path / "review.csv"
    batch.write_offline_prompt_export(prompts, prompt_path)
    batch.write_offline_prompt_export(completed, completed_path)

    summary = batch.import_completed_explanations(
        prompt_path,
        completed_path,
        review_path,
    )

    assert summary["schema_valid_cases"] == 29
    assert summary["automatic_rejection_cases"] == 1
    with review_path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert "insufficient evidence" in rows[-1][
        "automatic_validation_errors"
    ]
    assert rows[-1]["correctness"] == ""
