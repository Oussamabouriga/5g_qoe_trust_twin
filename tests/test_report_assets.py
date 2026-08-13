"""Consistency tests for publication assets and the IEEE report source."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.generate_report_assets import (
    CALIBRATION_PATH,
    HUMAN_EVALUATION_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    SPLIT_MANIFEST_PATH,
    load_json,
    validate_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_report_sources_are_consistent() -> None:
    validate_sources(
        load_json(METRICS_PATH),
        load_json(MODEL_COMPARISON_PATH),
        load_json(CALIBRATION_PATH),
        load_json(HUMAN_EVALUATION_PATH),
        load_json(SPLIT_MANIFEST_PATH),
    )


def test_publication_model_table_matches_final_metrics() -> None:
    final = load_json(METRICS_PATH)
    table_path = ROOT / "results/publication_tables/table3_model_comparison.csv"
    with table_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["model"] for row in rows] == [
        "persistence",
        "network_logistic",
        "cross_layer_random_forest",
    ]
    for table_row, metric_row in zip(
        rows,
        final["three_model_test_metrics"],
        strict=True,
    ):
        assert float(table_row["f1"]) == pytest.approx(metric_row["f1"])
        assert float(table_row["pr_auc"]) == pytest.approx(metric_row["pr_auc"])
        assert int(table_row["false_positives"]) == metric_row["false_positives"]


def test_llm_publication_summary_uses_retained_validated_batch() -> None:
    summary_path = ROOT / "results/publication_tables/llm_evaluation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["reviewed_cases"] == 30
    assert summary["schema_valid_cases"] == 30
    assert summary["grounding_valid_cases"] == 30
    assert summary["factual_correct_cases"] == 30
    assert summary["unsupported_statement_cases"] == 0
    assert summary["contradiction_cases"] == 0


def test_report_source_has_required_sections_and_no_stale_claims() -> None:
    source = (ROOT / "report/ieee_report.tex").read_text(encoding="utf-8")
    for section in (
        "State of the Art",
        "Work Description and Research Design",
        "Methodology",
        "Actions Performed and Software Implementation",
        "Results",
        "Suggestions for Improvement",
        "Requirement Compliance Audit",
    ):
        assert f"\\section{{{section}}}" in source

    for stale in (
        "0.8970 accuracy",
        "0.8779 F1",
        "15,522",
        "96.67\\%",
        "must be added before final submission",
        "exact numerical MOS threshold must be verified",
        "GPT-4o mini",
    ):
        assert stale not in source
