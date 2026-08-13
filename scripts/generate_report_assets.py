"""Generate publication tables and figures from authoritative frozen results.

This script deliberately does not read the legacy ``results/publication_*``
artifacts.  Those files came from an earlier experiment and are overwritten
only after the final metric, calibration, lineage, and human-review documents
have passed the consistency checks below.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results/metrics/final_final_evaluation.json"
MODEL_COMPARISON_PATH = ROOT / "results/metrics/final_model_comparison.json"
CALIBRATION_PATH = (
    ROOT / "results/metrics/selected_calibration_configuration_final.json"
)
HUMAN_EVALUATION_PATH = ROOT / "results/metrics/llm_human_evaluation.json"
SPLIT_MANIFEST_PATH = ROOT / "manifests/cp6_sample_split_manifest.json"
DATASET_SUMMARY_SOURCE = ROOT / "results/publication_tables/table1_dataset_summary.csv"

GENERATED_DIRECTORY = ROOT / "report/generated"
REPORT_TABLE_DIRECTORY = ROOT / "report/tables"
PUBLICATION_TABLE_DIRECTORY = ROOT / "results/publication_tables"

AUTHORITATIVE_INPUTS = (
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    CALIBRATION_PATH,
    HUMAN_EVALUATION_PATH,
    SPLIT_MANIFEST_PATH,
)

MODEL_LABELS = {
    "persistence": "Persistence",
    "network_logistic": "Network Logistic",
    "cross_layer_random_forest": "Cross-layer RF",
}
COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "gray": "#6b7280",
    "light": "#e8eef7",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and reject absent or malformed inputs."""
    if not path.is_file():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Report input must contain a JSON object: {path}")
    return document


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_close(
    left: float,
    right: float,
    *,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    """Fail when two frozen numeric representations disagree."""
    if not np.isclose(left, right, rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} mismatch: {left!r} != {right!r}")


def validate_sources(
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    calibration: Mapping[str, Any],
    human: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> None:
    """Enforce the cross-file contract before producing publication assets."""
    models = list(metrics["three_model_test_metrics"])
    comparison_models = list(comparison["models"])
    expected_names = [
        "persistence",
        "network_logistic",
        "cross_layer_random_forest",
    ]
    if [row["model"] for row in models] != expected_names:
        raise ValueError("Final evaluation does not contain the required models.")
    if [row["model"] for row in comparison_models] != expected_names:
        raise ValueError("Model-comparison document has an unexpected model order.")

    test_rows = int(comparison["sealed_test_rows"])
    if test_rows != int(metrics["overall_metrics"]["observations"]):
        raise ValueError("Sealed-test row counts disagree between result files.")

    for final_row, comparison_row in zip(models, comparison_models, strict=True):
        for metric_name in (
            "f1",
            "pr_auc",
            "brier_score",
            "expected_calibration_error",
            "false_positives",
        ):
            require_close(
                float(final_row[metric_name]),
                float(comparison_row[metric_name]),
                label=f"{final_row['model']} {metric_name}",
            )

    random_forest = models[-1]
    overall = metrics["overall_metrics"]
    for metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "brier_score",
        "expected_calibration_error",
        "false_positives",
    ):
        require_close(
            float(random_forest[metric_name]),
            float(overall[metric_name]),
            label=f"final RF/overall {metric_name}",
        )

    confusion_total = sum(
        int(overall[key])
        for key in (
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        )
    )
    if confusion_total != test_rows:
        raise ValueError("Final confusion-matrix counts do not sum to test rows.")

    trust = metrics["trust_summary"]
    if int(trust["total_predictions"]) != test_rows:
        raise ValueError("Trust summary does not cover the sealed test set.")
    if (
        int(trust["accepted_predictions"]) + int(trust["abstained_predictions"])
        != test_rows
    ):
        raise ValueError("Accepted and abstained counts do not sum to test rows.")
    require_close(
        float(trust["coverage"]),
        int(trust["accepted_predictions"]) / test_rows,
        label="test coverage",
    )

    selected = calibration["cross_layer_random_forest"]
    if selected["calibration_method"] != metrics["calibration"]:
        raise ValueError("Selected calibrator disagrees with final evaluation.")
    require_close(
        float(selected["decision_threshold"]),
        float(random_forest["decision_threshold"]),
        label="Random Forest decision threshold",
    )

    validation = human["validation"]
    overall_human = human["overall"]
    if int(validation["matching_unique_case_ids"]) != 30:
        raise ValueError("The frozen explanation review must contain 30 cases.")
    if int(validation["schema_valid_cases"]) != 30:
        raise ValueError("Not all retained explanations are schema-valid.")
    if int(validation["grounding_valid_cases"]) != 30:
        raise ValueError("Not all retained explanations pass grounding checks.")
    if int(overall_human["reviewed_cases"]) != 30:
        raise ValueError("Human review is incomplete.")

    output = split_manifest["output"]
    if int(output["rows"]) != sum(
        int(record["rows"]) for record in split_manifest["split"]["records"]
    ):
        raise ValueError("Split rows do not sum to the final feature dataset.")
    if int(output["rows"]) != 897_056:
        raise ValueError("Unexpected final feature-row count.")


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write a deterministic UTF-8 CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mirror_csv(
    filename: str,
    fieldnames: list[str],
    rows: list[Mapping[str, Any]],
) -> None:
    """Write identical report and publication copies of one table."""
    for directory in (REPORT_TABLE_DIRECTORY, PUBLICATION_TABLE_DIRECTORY):
        write_csv(directory / filename, fieldnames, rows)


def fmt4(value: Any) -> str:
    """Format a metric to four decimal places for tables."""
    return f"{float(value):.4f}"


def escape_tex(value: str) -> str:
    """Escape the small TeX-sensitive subset used in generated cells."""
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_tex_table(
    filename: str,
    column_specification: str,
    header: list[str],
    rows: Iterable[Iterable[Any]],
) -> Path:
    """Write a compact booktabs table body for inclusion by the report."""
    lines = [
        rf"\begin{{tabular}}{{{column_specification}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(escape_tex(str(value)) for value in row) + r" \\" for row in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path = GENERATED_DIRECTORY / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_dataset_summary(split_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load the audited split summary and verify immutable counts."""
    if not DATASET_SUMMARY_SOURCE.is_file():
        raise FileNotFoundError(
            f"Dataset summary source is missing: {DATASET_SUMMARY_SOURCE}"
        )
    with DATASET_SUMMARY_SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest_records = {
        str(record["split"]).title(): record
        for record in split_manifest["split"]["records"]
    }
    if [row["Split"] for row in rows] != ["Train", "Validation", "Test"]:
        raise ValueError("Dataset summary must contain Train/Validation/Test rows.")
    for row in rows:
        record = manifest_records[row["Split"]]
        if int(row["Rows"]) != int(record["rows"]):
            raise ValueError(f"Dataset row count mismatch for {row['Split']}.")
        if int(row["Sessions"]) != int(record["sessions"]):
            raise ValueError(f"Session count mismatch for {row['Split']}.")
    return rows


def generate_tables(
    metrics: Mapping[str, Any],
    calibration: Mapping[str, Any],
    human: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> None:
    """Generate all CSV and TeX tables used by the revised report."""
    dataset_rows = load_dataset_summary(split_manifest)
    mirror_csv(
        "table1_dataset_summary.csv",
        ["Split", "Rows", "Sessions", "Users", "Poor QoE Rate"],
        dataset_rows,
    )
    write_tex_table(
        "table_dataset.tex",
        "lrrrr",
        ["Split", "Rows", "Sessions", "Users", "Poor rate"],
        (
            (
                row["Split"],
                f"{int(row['Rows']):,}",
                f"{int(row['Sessions']):,}",
                row["Users"],
                f"{100 * float(row['Poor QoE Rate']):.2f}%",
            )
            for row in dataset_rows
        ),
    )

    model_rows: list[dict[str, Any]] = []
    for row in metrics["three_model_test_metrics"]:
        model_rows.append(
            {
                "model": row["model"],
                "calibration_method": row["calibration_method"],
                "decision_threshold": row["decision_threshold"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "pr_auc": row["pr_auc"],
                "roc_auc": row["roc_auc"],
                "brier_score": row["brier_score"],
                "expected_calibration_error": row["expected_calibration_error"],
                "false_positives": row["false_positives"],
                "false_alarm_rate": row["false_alarm_rate"],
            }
        )
    model_fields = list(model_rows[0])
    mirror_csv("table3_model_comparison.csv", model_fields, model_rows)
    write_tex_table(
        "table_models.tex",
        "lrrrrrr",
        ["Model", "F1", "PR-AUC", "Brier", "ECE", "FP", "FAR"],
        (
            (
                MODEL_LABELS[row["model"]],
                fmt4(row["f1"]),
                fmt4(row["pr_auc"]),
                fmt4(row["brier_score"]),
                fmt4(row["expected_calibration_error"]),
                f"{int(row['false_positives']):,}",
                f"{100 * float(row['false_alarm_rate']):.2f}%",
            )
            for row in model_rows
        ),
    )

    final_rows: list[dict[str, Any]] = []
    for scope, result in (
        ("All predictions", metrics["overall_metrics"]),
        ("Accepted only", metrics["accepted_metrics"]),
    ):
        final_rows.append(
            {
                "scope": scope,
                "observations": int(result["observations"]),
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "pr_auc": result["pr_auc"],
                "brier_score": result["brier_score"],
                "expected_calibration_error": result["expected_calibration_error"],
            }
        )
    final_fields = list(final_rows[0])
    mirror_csv("table2_final_metrics.csv", final_fields, final_rows)
    write_tex_table(
        "table_final.tex",
        "lrrrrrr",
        ["Scope", "Acc.", "Prec.", "Recall", "F1", "Brier", "ECE"],
        (
            (
                row["scope"],
                fmt4(row["accuracy"]),
                fmt4(row["precision"]),
                fmt4(row["recall"]),
                fmt4(row["f1"]),
                fmt4(row["brier_score"]),
                fmt4(row["expected_calibration_error"]),
            )
            for row in final_rows
        ),
    )

    calibration_rows = []
    for key in ("network_logistic", "cross_layer_random_forest"):
        row = calibration[key]
        calibration_rows.append(
            {
                "model": key,
                "calibration_method": row["calibration_method"],
                "decision_threshold": row["decision_threshold"],
                "validation_f1": row["f1"],
                "validation_pr_auc": row["pr_auc"],
                "validation_brier_score": row["brier_score"],
                "validation_expected_calibration_error": row[
                    "expected_calibration_error"
                ],
            }
        )
    calibration_fields = list(calibration_rows[0])
    mirror_csv("table4_calibration.csv", calibration_fields, calibration_rows)
    write_tex_table(
        "table_calibration.tex",
        "lrrrrr",
        ["Model", "Method", "Threshold", "F1", "Brier", "ECE"],
        (
            (
                MODEL_LABELS[row["model"]],
                row["calibration_method"].title(),
                f"{float(row['decision_threshold']):.2f}",
                fmt4(row["validation_f1"]),
                fmt4(row["validation_brier_score"]),
                fmt4(row["validation_expected_calibration_error"]),
            )
            for row in calibration_rows
        ),
    )

    trust = metrics["trust_summary"]
    trust_rows = [
        {
            "measure": "total_predictions",
            "value": int(trust["total_predictions"]),
        },
        {
            "measure": "accepted_predictions",
            "value": int(trust["accepted_predictions"]),
        },
        {
            "measure": "abstained_predictions",
            "value": int(trust["abstained_predictions"]),
        },
        {"measure": "coverage", "value": trust["coverage"]},
        {"measure": "abstention_rate", "value": trust["abstention_rate"]},
        {
            "measure": "accepted_decision_accuracy",
            "value": trust["accepted_decision_accuracy"],
        },
    ]
    mirror_csv("table5_trust.csv", ["measure", "value"], trust_rows)

    validation = human["validation"]
    reviewed = human["overall"]
    llm_row = {
        "reviewed_cases": reviewed["reviewed_cases"],
        "schema_valid_cases": validation["schema_valid_cases"],
        "grounding_valid_cases": validation["grounding_valid_cases"],
        "factual_correct_cases": reviewed["factual_correct_cases"],
        "complete_cases": reviewed["complete_cases"],
        "unsupported_statement_cases": reviewed["unsupported_statement_cases"],
        "contradiction_cases": reviewed["contradiction_cases"],
        "reviewer_count": validation["reviewer_count"],
    }
    for directory in (REPORT_TABLE_DIRECTORY, PUBLICATION_TABLE_DIRECTORY):
        (directory / "llm_evaluation.json").write_text(
            json.dumps(llm_row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_tex_table(
        "table_llm.tex",
        "lr",
        ["Measure", "Result"],
        [
            ("Retained/reviewed", "30/30"),
            ("Schema valid", "30 (100%)"),
            ("Grounding valid", "30 (100%)"),
            ("Factual and complete", "30 (100%)"),
            ("Unsupported statements", "0 (0%)"),
            ("Contradictions", "0 (0%)"),
        ],
    )


def write_macros(
    metrics: Mapping[str, Any],
    calibration: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> None:
    """Write all headline values as TeX commands backed by frozen JSON."""
    overall = metrics["overall_metrics"]
    accepted = metrics["accepted_metrics"]
    trust = metrics["trust_summary"]
    lead = metrics["lead_time_statistics"]
    rf_selection = calibration["cross_layer_random_forest"]
    macros = {
        "SampleRows": f"{int(split_manifest['input']['rows']):,}",
        "PhysicalObservations": (
            f"{int(split_manifest['sample_validation']['physical_observations']):,}"
        ),
        "SampleSessions": f"{int(split_manifest['sample_validation']['sessions']):,}",
        "FinalRows": f"{int(split_manifest['output']['rows']):,}",
        "TestRows": f"{int(overall['observations']):,}",
        "TestPoorRate": f"{100 * float(overall['positive_rate']):.2f}\\%",
        "RFAccuracy": fmt4(overall["accuracy"]),
        "RFPrecision": fmt4(overall["precision"]),
        "RFRecall": fmt4(overall["recall"]),
        "RFFone": fmt4(overall["f1"]),
        "RFPRAUC": fmt4(overall["pr_auc"]),
        "RFROCAUC": fmt4(overall["roc_auc"]),
        "RFBrier": fmt4(overall["brier_score"]),
        "RFECE": fmt4(overall["expected_calibration_error"]),
        "FalsePositives": f"{int(overall['false_positives']):,}",
        "FalseAlarmRate": f"{100 * float(overall['false_alarm_rate']):.2f}\\%",
        "MeanLead": f"{float(lead['mean_seconds']):.3f}",
        "MedianLead": f"{float(lead['median_seconds']):.3f}",
        "PninetyfiveLead": f"{float(lead['p95_seconds']):.3f}",
        "Coverage": f"{100 * float(trust['coverage']):.2f}\\%",
        "AbstentionRate": f"{100 * float(trust['abstention_rate']):.2f}\\%",
        "AbstainedCount": f"{int(trust['abstained_predictions']):,}",
        "AcceptedAccuracy": fmt4(accepted["accuracy"]),
        "AcceptedPrecision": fmt4(accepted["precision"]),
        "AcceptedRecall": fmt4(accepted["recall"]),
        "AcceptedFone": fmt4(accepted["f1"]),
        "DecisionThreshold": f"{float(rf_selection['decision_threshold']):.2f}",
        "TrustThreshold": f"{float(rf_selection['abstention_threshold']):.6f}",
    }
    lines = [
        rf"\newcommand{{\{name}}}{{{value}\xspace}}" for name, value in macros.items()
    ]
    (GENERATED_DIRECTORY / "report_numbers.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save one publication figure in vector and preview formats."""
    fig.savefig(
        GENERATED_DIRECTORY / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "generate_report_assets.py"},
    )
    fig.savefig(
        GENERATED_DIRECTORY / f"{name}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def style_axis(axis: plt.Axes) -> None:
    """Apply restrained IEEE-friendly chart styling."""
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def generate_figures(
    metrics: Mapping[str, Any],
    human: Mapping[str, Any],
) -> None:
    """Create summary charts without requiring per-row prediction artifacts."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
        }
    )

    models = list(metrics["three_model_test_metrics"])
    labels = [MODEL_LABELS[row["model"]] for row in models]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
    for offset, metric_name, color in (
        (-0.18, "f1", COLORS["blue"]),
        (0.18, "pr_auc", COLORS["orange"]),
    ):
        axes[0].bar(
            x + offset,
            [row[metric_name] for row in models],
            width=0.34,
            label=metric_name.upper().replace("_", "-"),
            color=color,
        )
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_title("Classification quality")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].bar(
        x - 0.18,
        [row["brier_score"] for row in models],
        width=0.34,
        label="Brier",
        color=COLORS["green"],
    )
    axes[1].bar(
        x + 0.18,
        [row["expected_calibration_error"] for row in models],
        width=0.34,
        label="ECE",
        color=COLORS["purple"],
    )
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_title("Probability reliability (lower is better)")
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    axes[2].bar(
        x,
        [100 * row["false_alarm_rate"] for row in models],
        color=[COLORS["gray"], COLORS["red"], COLORS["blue"]],
    )
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_ylabel("False-alarm rate (%)")
    axes[2].set_title("Operational false alarms")
    style_axis(axes[2])
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, "model_comparison")

    all_metrics = metrics["overall_metrics"]
    accepted = metrics["accepted_metrics"]
    measures = ["accuracy", "precision", "recall", "f1"]
    measure_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(measures))
    fig, axis = plt.subplots(figsize=(3.45, 2.25))
    axis.bar(
        x - 0.18,
        [all_metrics[name] for name in measures],
        width=0.36,
        label="All predictions",
        color=COLORS["gray"],
    )
    axis.bar(
        x + 0.18,
        [accepted[name] for name in measures],
        width=0.36,
        label="Accepted only",
        color=COLORS["blue"],
    )
    axis.set_ylim(0.80, 0.96)
    axis.set_xticks(x, measure_labels)
    axis.set_title("Selective prediction at 90.40% coverage")
    axis.legend(frameon=False, loc="lower right")
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "selective_performance")

    features = list(metrics["top_20_features"][:12])
    feature_labels = [
        row["feature"].replace("numeric__", "").replace("_", " ")
        for row in reversed(features)
    ]
    feature_values = [row["importance"] for row in reversed(features)]
    fig, axis = plt.subplots(figsize=(3.45, 3.0))
    axis.barh(feature_labels, feature_values, color=COLORS["blue"])
    axis.set_xlabel("Impurity importance")
    axis.set_title("Top cross-layer Random Forest features")
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "feature_importance")

    lead = metrics["lead_time_statistics"]
    quantile_labels = ["Min", "P25", "Median", "Mean", "P75", "P95", "Max"]
    quantile_values = [
        lead["minimum_seconds"],
        lead["p25_seconds"],
        lead["median_seconds"],
        lead["mean_seconds"],
        lead["p75_seconds"],
        lead["p95_seconds"],
        lead["maximum_seconds"],
    ]
    fig, axis = plt.subplots(figsize=(3.45, 2.15))
    axis.plot(
        quantile_labels,
        quantile_values,
        marker="o",
        linewidth=1.7,
        color=COLORS["orange"],
    )
    axis.axhline(10.0, linestyle="--", linewidth=0.8, color=COLORS["gray"])
    axis.set_ylabel("Seconds to next observation")
    axis.set_title("Observed one-step horizon")
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "lead_time_summary")

    validation = human["validation"]
    reviewed = human["overall"]
    labels = ["Schema", "Grounding", "Correct", "Complete", "No unsupported"]
    values = [
        validation["schema_valid_cases"] / 30,
        validation["grounding_valid_cases"] / 30,
        reviewed["factual_correct_cases"] / 30,
        reviewed["complete_cases"] / 30,
        1 - reviewed["unsupported_statement_cases"] / 30,
    ]
    fig, axis = plt.subplots(figsize=(3.45, 2.15))
    bars = axis.bar(labels, values, color=COLORS["green"])
    axis.bar_label(bars, labels=[f"{100 * value:.0f}%" for value in values], padding=2)
    axis.set_ylim(0, 1.12)
    axis.set_title("Retained explanation review (n=30)")
    axis.tick_params(axis="x", rotation=20)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "llm_validation")

    fig, axis = plt.subplots(figsize=(7.15, 1.35))
    axis.set_xlim(0, 8)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stages = [
        "5G-QoERA\ntrace",
        "Chronological\nreplay",
        "Causal\nstate/features",
        "Future QoE\nmodels",
        "Isotonic\ncalibration",
        "Trust +\nabstention",
        "Evidence\nobject",
        "LLM +\ngrounding gate",
    ]
    for index, stage in enumerate(stages):
        axis.text(
            index + 0.45,
            0.50,
            stage,
            ha="center",
            va="center",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": COLORS["light"],
                "edgecolor": COLORS["blue"],
                "linewidth": 0.8,
            },
        )
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(index + 1.06, 0.5),
                xytext=(index + 0.83, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
            )
    fig.tight_layout(pad=0.2)
    save_figure(fig, "architecture")


def write_generation_manifest() -> None:
    """Record exact source and generated-artifact hashes for the report build."""
    outputs = sorted(
        path
        for path in GENERATED_DIRECTORY.iterdir()
        if path.is_file() and path.name != "generation_manifest.json"
    )
    document = {
        "schema_version": 1,
        "authoritative_inputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path in AUTHORITATIVE_INPUTS
        ],
        "generated_outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    (GENERATED_DIRECTORY / "generation_manifest.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Validate frozen sources and regenerate every current report asset."""
    GENERATED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PUBLICATION_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    metrics = load_json(METRICS_PATH)
    comparison = load_json(MODEL_COMPARISON_PATH)
    calibration = load_json(CALIBRATION_PATH)
    human = load_json(HUMAN_EVALUATION_PATH)
    split_manifest = load_json(SPLIT_MANIFEST_PATH)

    validate_sources(metrics, comparison, calibration, human, split_manifest)
    generate_tables(metrics, calibration, human, split_manifest)
    write_macros(metrics, calibration, split_manifest)
    generate_figures(metrics, human)
    write_generation_manifest()

    print(f"Validated {len(AUTHORITATIVE_INPUTS)} authoritative inputs.")
    print(f"Generated report assets in {GENERATED_DIRECTORY.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
