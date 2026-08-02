import json
import pandas as pd
from pathlib import Path

with open(
    "results/metrics/final_final_evaluation.json",
    "r",
    encoding="utf-8",
) as f:
    metrics = json.load(f)

overall = metrics["overall_metrics"]
accepted = metrics["accepted_metrics"]

table = pd.DataFrame([
    {
        "Metric": "Accuracy",
        "Overall": overall["accuracy"],
        "Accepted": accepted["accuracy"],
    },
    {
        "Metric": "Precision",
        "Overall": overall["precision"],
        "Accepted": accepted["precision"],
    },
    {
        "Metric": "Recall",
        "Overall": overall["recall"],
        "Accepted": accepted["recall"],
    },
    {
        "Metric": "F1-score",
        "Overall": overall["f1"],
        "Accepted": accepted["f1"],
    },
    {
        "Metric": "PR-AUC",
        "Overall": overall["pr_auc"],
        "Accepted": accepted["pr_auc"],
    },
    {
        "Metric": "ROC-AUC",
        "Overall": overall["roc_auc"],
        "Accepted": accepted["roc_auc"],
    },
    {
        "Metric": "Brier Score",
        "Overall": overall["brier_score"],
        "Accepted": accepted["brier_score"],
    },
    {
        "Metric": "ECE",
        "Overall": overall["expected_calibration_error"],
        "Accepted": accepted["expected_calibration_error"],
    },
])

output = Path(
    "results/publication_tables/table2_final_metrics.csv"
)

table.to_csv(
    output,
    index=False,
)

print(table)
print(f"\nSaved: {output}")
