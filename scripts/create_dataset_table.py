from pathlib import Path
import pandas as pd

df = pd.read_parquet(
    "data/processed/qoe_features_final.parquet"
)

rows = []

for split in ["train", "validation", "test"]:

    subset = df[df["split"] == split]

    rows.append({
        "Split": split.capitalize(),
        "Rows": len(subset),
        "Sessions": subset["session_id"].nunique(),
        "Users": subset["user_id"].nunique(),
        "Poor QoE Rate": round(
            subset["future_poor_qoe"].mean(),
            4,
        ),
    })

table = pd.DataFrame(rows)

output = Path(
    "results/publication_tables/table1_dataset_summary.csv"
)

table.to_csv(
    output,
    index=False,
)

print(table)
print(f"\nSaved: {output}")
