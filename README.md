# Trust-Aware 5G QoE Digital Twin

A research prototype for **next-observation poor-QoE forecasting in 5G New Radio video streaming**.

The project combines chronological session replay, Digital Twin state reconstruction, temporal and cross-layer feature engineering, future poor-QoE prediction, probability calibration, trust scoring and selective abstention, OpenAI-based structured explanations, schema validation, grounding validation, and publication-ready evaluation outputs.


---

## 1. Project objective

The system addresses the following research question:

> Can a chronological, trust-aware Digital Twin forecast poor video-streaming QoE at the next observed row while producing calibrated probabilities, abstaining from unreliable predictions, and generating evidence-grounded operator explanations?

```text
5G-QoERA dataset
        ↓
Data inspection and preparation
        ↓
Long-format modelling dataset
        ↓
Complete-session sampling
        ↓
Chronological Digital Twin replay
        ↓
Temporal and cross-layer features
        ↓
Prediction models
        ↓
Probability calibration
        ↓
Trust score and abstention
        ↓
Final test evaluation
        ↓
Structured evidence
        ↓
OpenAI GPT-4o mini explanation
        ↓
Pydantic schema validation
        ↓
Grounding validation
        ↓
Accepted or rejected explanation
```

---

## 2. Historical submitted results (superseded)

> The values below belong to the pre-correction submission and are not results
> of the frozen CP5 experiment. They must not be used as evidence for the
> corrected pipeline. CP8 will replace them after one untouched-test run.

The final resource-safe sample contains **994,496 rows** and preserves complete sessions across all base stations, Physical Resource Block allocations, and selected video bitrates.

After feature construction and chronological-boundary filtering:

| Split | Rows | Sessions | Users | Poor-QoE rate |
|---|---:|---:|---:|---:|
| Train | 537,504 | 51,232 | 216 | 30.18% |
| Validation | 180,224 | 17,152 | 167 | 37.35% |
| Test | 179,328 | 18,144 | 181 | 41.43% |

Final calibrated cross-layer Random Forest results:

| Metric | All predictions | Accepted predictions |
|---|---:|---:|
| Accuracy | 0.8970 | 0.9335 |
| Precision | 0.8626 | 0.9469 |
| Recall | 0.8937 | 0.8866 |
| F1-score | 0.8779 | 0.9158 |
| PR-AUC | 0.9538 | 0.9650 |
| ROC-AUC | 0.9633 | 0.9726 |
| Brier score | 0.0716 | 0.0555 |
| Expected Calibration Error | 0.0084 | 0.0074 |

Trust results:

- coverage: **91.34%**;
- abstention rate: **8.66%**;
- high-trust accuracy: **96.41%**;
- accepted-decision accuracy: **93.35%**.

LLM evaluation on 30 diverse cases:

- schema-valid rate: **100%**;
- grounding pass rate: **96.67%**;
- explanation acceptance rate: **96.67%**;
- mean generation latency: approximately **2.87 seconds**.

---

## 3. Project structure

```text
5g_qoe_trust_twin/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── trust.yaml
│   └── llm.yaml
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   ├── raw/
│   └── llm_evaluation/
├── docs/
├── models/
│   ├── baseline/
│   ├── calibrated/
│   └── uncalibrated/
├── notebooks/
├── paper/
├── report/
│   ├── figures/
│   ├── references/
│   ├── tables/
│   └── ieee_report.tex
├── results/
│   ├── evidence/
│   ├── explanations/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   ├── publication_figures/
│   ├── publication_tables/
│   └── tables/
├── scripts/
├── src/
│   └── qoe_twin/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

### Important source modules

| Module | Purpose |
|---|---|
| `src/qoe_twin/features.py` | Temporal and cross-layer feature generation |
| `src/qoe_twin/replay.py` | Chronological Digital Twin replay |
| `src/qoe_twin/models.py` | Logistic Regression and Random Forest pipelines |
| `src/qoe_twin/calibration.py` | Sigmoid/isotonic calibration and ECE |
| `src/qoe_twin/metrics.py` | Classification and threshold metrics |
| `src/qoe_twin/trust.py` | Trust score and abstention |
| `src/qoe_twin/evidence_builder.py` | Controlled evidence construction |
| `src/qoe_twin/prompt_builder.py` | LLM prompt construction |
| `src/qoe_twin/openai_client.py` | OpenAI Responses API integration |
| `src/qoe_twin/explanation_schema.py` | Strict Pydantic explanation schema |
| `src/qoe_twin/grounding_validator.py` | Numeric and evidence grounding checks |
| `src/qoe_twin/evaluation.py` | Final evaluation utilities |

---

## 4. Requirements

Recommended environment:

- macOS, Linux, or Windows;
- Python **3.11**;
- at least 8 GB RAM;
- 16 GB RAM recommended for the final sample;
- sufficient disk space for the external dataset, processed Parquet files, and models;
- an OpenAI API key only for the explanation stage.

The project was developed and tested with Python 3.11.15.

Check the active interpreter:

```bash
which python
python --version
```

Expected on macOS/Linux:

```text
.../5g_qoe_trust_twin/.venv/bin/python
Python 3.11.x
```

---

## 5. Installation

### 5.1 Open the project

```bash
cd ~/Documents/5g_qoe_trust_twin
```

### 5.2 Create and activate the virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5.3 Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional exact environment:

```bash
python -m pip install -r requirements-lock.txt
```

### 5.4 Verify installation

```bash
python -c "
import numpy
import pandas
import sklearn
import pydantic
from openai import OpenAI

print('NumPy:', numpy.__version__)
print('Pandas:', pandas.__version__)
print('Scikit-learn:', sklearn.__version__)
print('Pydantic:', pydantic.__version__)
print('OpenAI SDK imported successfully')
"
```

---

## 6. Environment variables

Create `.env`:

```bash
cp .env.example .env
```

Example:

```dotenv
OPENAI_API_KEY=replace_with_your_real_key
OPENAI_MODEL=gpt-4o-mini

PROJECT_NAME=5g_qoe_trust_twin
RANDOM_SEED=42
LOG_LEVEL=INFO

DATASET_DIRECTORY=data/external/5G-QoERA/5G-QoERA
RAW_DATA_DIRECTORY=data/raw
PROCESSED_DATA_DIRECTORY=data/processed
MODEL_DIRECTORY=models
RESULTS_DIRECTORY=results
```

Important:

- never commit `.env`;
- never expose `OPENAI_API_KEY`;
- `.env.example` must contain placeholders only;
- the ML pipeline does not require the API key;
- only explanation scripts call OpenAI.

Test the explanation layer:

```bash
python scripts/test_explanation_generator.py
```

---

## 7. Download the dataset

```bash
git clone \
  https://github.com/telematics-lab/5G-QoERA.git \
  data/external/5G-QoERA
```

Verify 32 TSV files:

```bash
find data/external/5G-QoERA/5G-QoERA \
  -type f \
  -name "*.tsv" \
  | wc -l
```

Expected:

```text
32
```

Example:

```text
MOS_BS_1_Driver_PRB_10.tsv
```

Meaning:

- `BS_1`: base station 1;
- `Driver`: driver mobility scenario;
- `PRB_10`: 10 Physical Resource Blocks.

TSV loading:

```python
pd.read_csv(path, sep="\t")
```

---

## 8. Validate the project

Run all tests:

```bash
python -m pytest -v
```

The validated project state contains **37 passing tests**.

Tests cover:

- environment and dataset availability;
- target generation;
- feature causality;
- chronological splitting;
- cross-boundary label removal;
- Digital Twin replay;
- model probabilities;
- threshold selection;
- calibration and ECE;
- trust scoring and abstention;
- evaluation utilities;
- evidence and prompt construction;
- explanation schema;
- grounding validation.

---

## 9. Dataset inspection

```bash
python scripts/inspect_dataset.py
python scripts/analyze_dataset_structure.py
python scripts/analyze_sessions.py
python scripts/analyze_application_columns.py
```

These scripts generate exploratory summaries but do not train the final model.

---

## 10. Prepare the modelling dataset

```bash
python scripts/prepare_data.py
```

This stage:

1. reads the 32 TSV files;
2. parses scenario metadata from filenames;
3. transforms wide application and MOS columns to long format;
4. selects configured video bitrates;
5. creates session identifiers;
6. sorts observations chronologically;
7. creates future poor-QoE labels;
8. records the elapsed time to the next observation;
9. writes the processed dataset.

Output:

```text
data/processed/qoe_modeling_dataset.parquet
```

The complete long-format dataset contains approximately 16.59 million observations.

The target is **next-observation poor-QoE forecasting**. For each row, the
future row is the immediately following observation in the same session. The
label is one exactly when that future MOS is strictly below `3.0`; MOS `3.0`
is not poor QoE. The horizon is one observation, so the elapsed time can vary.
This is neither degradation-onset detection nor a fixed-ten-second warning;
already-poor to poor transitions remain positive examples.

---

## 11. Validate the resource-safe final sample

CP6 does not rebuild the 16.6-million-row intermediate on a 16 GB machine.
It requires the locally audited, immutable input:

```text
data/processed/friend_snapshot_quarantine/qoe_modeling_final_sample.parquet
SHA-256: 68ee536b88d294b9faa97b5e2c7d3eec4c46c07dc756d163b95ce32fa1d09100
```

The source remains labelled `quarantine` because its historical sampling
command is unavailable. Before using it, the CP6 build verifies its exact
bytes and schema, checks every row against the trusted raw TSV content, and
discards all inherited session and future-target calculations.

Validated input:

- 994,496 rows;
- 97,248 complete sessions;
- 245 users;
- four base stations;
- PRB values from 5 to 40;
- bitrates 2,000, 4,000, 6,000, and 8,000 kbps.

This is an earliest-time, group-balanced sample rather than a representative
random sample. Four bitrate rows also share each physical radio observation.

---

## 12. Build final Digital Twin features

```bash
python scripts/build_features_final.py
```

This stage:

- validates the pinned sample and its trusted-raw compatibility;
- rebuilds sessions and next-observation targets from base measurements;
- computes lag and past-only rolling features;
- derives mobility, capacity, packet-loss, and MOS features;
- performs a global timestamp-based 60/20/20 train/validation/test split;
- removes targets crossing split boundaries;
- stores rows in stable global timestamp order.

Outputs:

```text
data/processed/qoe_features_final.parquet
manifests/cp6_sample_split_manifest.json
```

Future MOS values are never used as prediction inputs.

---

## 13. Train the models

```bash
python scripts/train_models_final.py
```

Models:

1. persistence baseline;
2. network-only Logistic Regression;
3. cross-layer Random Forest.

Outputs:

```text
models/uncalibrated/
results/metrics/final_training_metadata.json
```

The learned estimators and preprocessing pipelines are fitted only on training
rows. This stage does not inspect validation or test outcomes and does not
select decision thresholds. Validation roles are reserved for calibration.

---

## 14. Calibrate probabilities

```bash
python scripts/calibrate_models_final.py
```

Compared methods:

- sigmoid;
- isotonic.

The validation partition is divided chronologically. Its first half fits both
calibrators, after removing labels that cross the internal midpoint. Its second
half selects the calibration method and decision threshold. The test partition
remains sealed until the single CP8 final run. CP5 does not preselect a method
or threshold.

Outputs:

```text
models/calibrated/
results/tables/final_calibration_comparison.csv
results/metrics/selected_calibration_configuration_final.json
```

---

## 15. Apply trust scoring and abstention

```bash
python scripts/apply_trust_final.py
```

Validation quality:

```text
validation_quality =
    0.70 × validation_F1
    + 0.30 × (1 − validation_ECE)
```

Trust score:

```text
trust_score =
    0.45 × probability_confidence
    + 0.35 × validation_quality
    + 0.10 × data_quality
    + 0.10 × prediction_stability
```

Trust levels:

```text
High:   trust_score ≥ 0.80
Medium: 0.55 ≤ trust_score < 0.80
Low:    trust_score < 0.55
```

Decision:

```text
High or Medium → accept
Low            → abstain
```

Outputs:

```text
results/predictions/final_trusted_predictions.parquet
results/tables/final_trust_summary.csv
```

---

## 16. Final evaluation

```bash
python scripts/final_evaluation_final.py
```

Outputs:

```text
results/metrics/final_final_evaluation.json
results/tables/
results/figures/
```

Generated final figures:

```text
final_accuracy_by_trust.pdf
final_confusion_matrix.pdf
final_feature_importance.pdf
final_lead_time_histogram.pdf
final_precision_recall_curve.pdf
final_reliability_diagram.pdf
final_trust_distribution.pdf
```

---

## 17. Generate grounded LLM explanations

Confirm `.env`:

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Smoke test:

```bash
python scripts/generate_batch_explanations.py --cases 3
```

Inspect:

```bash
cat results/explanations/llm_evaluation.json
```

Final evaluation batch:

```bash
python scripts/generate_batch_explanations.py --cases 30
```

Outputs:

```text
results/explanations/final_explanations.json
results/explanations/final_explanations.parquet
results/explanations/llm_evaluation.json
```

The LLM receives only approved evidence fields. It does not receive the target label, training data, or unrestricted raw histories.

---

## 18. Create publication tables

```bash
python scripts/create_dataset_table.py
python scripts/create_final_metrics_table.py
```

Outputs:

```text
results/publication_tables/
├── table1_dataset_summary.csv
├── table2_final_metrics.csv
├── table3_model_comparison.csv
├── table4_calibration.csv
├── table5_trust.csv
└── llm_evaluation.json
```

Copy final publication material into the report:

```bash
cp results/publication_figures/* report/figures/
cp results/publication_tables/* report/tables/
```

---

## 19. Recommended complete execution order

```bash
source .venv/bin/activate

python -m pytest -v

python scripts/build_features_final.py
python scripts/train_models_final.py
python scripts/calibrate_models_final.py
python scripts/apply_trust_final.py
python scripts/final_evaluation_final.py

python scripts/generate_batch_explanations.py --cases 3
python scripts/generate_batch_explanations.py --cases 30

python scripts/create_dataset_table.py
python scripts/create_final_metrics_table.py

python -m pytest -v
```

Run expensive stages one at a time.

---

## 20. Run only the already-trained project

```bash
source .venv/bin/activate

python scripts/apply_trust_final.py
python scripts/final_evaluation_final.py
python scripts/generate_batch_explanations.py --cases 3
```

Regenerate publication tables:

```bash
python scripts/create_dataset_table.py
python scripts/create_final_metrics_table.py
```

Run tests only:

```bash
python -m pytest -v
```

---

## 21. Main output paths

| Artifact | Path |
|---|---|
| Complete processed dataset | `data/processed/qoe_modeling_dataset.parquet` |
| Audited manageable input | `data/processed/friend_snapshot_quarantine/qoe_modeling_final_sample.parquet` |
| Final feature dataset | `data/processed/qoe_features_final.parquet` |
| CP6 sample/split manifest | `manifests/cp6_sample_split_manifest.json` |
| Uncalibrated models | `models/uncalibrated/` |
| Calibrated models | `models/calibrated/` |
| Final trusted predictions | `results/predictions/final_trusted_predictions.parquet` |
| Final metrics | `results/metrics/final_final_evaluation.json` |
| Model comparison | `results/tables/final_model_comparison.csv` |
| Calibration comparison | `results/tables/final_calibration_comparison.csv` |
| Trust summary | `results/tables/final_trust_summary.csv` |
| Explanations | `results/explanations/final_explanations.json` |
| LLM evaluation | `results/explanations/llm_evaluation.json` |
| Final figures | `results/figures/` |
| Publication figures | `results/publication_figures/` |
| Publication tables | `results/publication_tables/` |
| IEEE report | `report/ieee_report.tex` |

---

## 22. VS Code workflow

```bash
code .
```

In VS Code:

1. open the Command Palette;
2. select **Python: Select Interpreter**;
3. select `.venv/bin/python`;
4. open the integrated terminal;
5. activate the environment.

```bash
source .venv/bin/activate
python --version
python -m pytest -v
```

Jupyter is optional. The whole project can be run from normal Python scripts.

---

## 23. Troubleshooting

### Pytest uses the wrong Python version

```bash
deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

cd ~/Documents/5g_qoe_trust_twin
source .venv/bin/activate

which python
python --version
python -m pytest -v
```

Use `python -m pytest`, not a global `pytest`.

### `deactivate: command not found`

The environment is already inactive:

```bash
source .venv/bin/activate
```

### Permission denied for a Parquet path

A Parquet path is a file, not a command.

Wrong:

```bash
data/processed/qoe_features_final.parquet
```

Correct:

```bash
ls -lh data/processed/qoe_features_final.parquet
```

### No TSV files found

```bash
find data/external/5G-QoERA/5G-QoERA \
  -type f \
  -name "*.tsv" \
  | wc -l
```

Expected: `32`.

### OpenAI key missing

Set:

```dotenv
OPENAI_API_KEY=your_real_key
OPENAI_MODEL=gpt-4o-mini
```

Then run:

```bash
python scripts/test_explanation_generator.py
```

### Explanation schema failure

Do not use unrelated JSON such as `{"status": "working"}`. Use:

```bash
python scripts/test_explanation_generator.py
```

### Grounding rejection

A rejection means an unsupported number or inconsistency was detected. Inspect:

```text
results/explanations/final_explanations.json
```

### Memory pressure

Run `scripts/build_features_final.py` by itself with other memory-intensive
applications closed. Do not construct the full 16.6-million-row intermediate
on a 16 GB machine. Keep Random Forest resource limits enabled for CP8.

### Homebrew warnings

MongoDB/FVM trust warnings are unrelated to this project. The `tree` command is not required:

```bash
find . -maxdepth 2
```

---

## 24. Reproducibility rules

- keep chronological splitting;
- never use a random row split;
- never calculate temporal features with future values;
- fit preprocessing only on training data;
- fit calibrators only on the first chronological validation half;
- select calibration methods and thresholds only on the second half;
- keep test labels and outcomes sealed until the frozen CP8 evaluation;
- retain complete sessions when sampling;
- never tune on the test set;
- never expose the target label to the LLM;
- reject explanations failing schema or grounding checks;
- record dependency versions and random seeds.

---

## 25. Known limitations

- The target is a one-observation-ahead state forecast, not degradation onset;
  the elapsed interval between observations is irregular.
- The final sample is smaller than the full long-format dataset.
- Evaluation covers one dataset and one family of historical scenarios.
- Random Forest feature importance is associative, not causal.
- The LLM grounding evaluation includes 30 automatic cases.
- GPT-4o mini is an external API dependency.
- This is a research decision-support prototype, not a production network-control system.

---

## 26. Future work

- additional 5G and 6G datasets;
- online recalibration;
- risk–coverage curves;
- alternative abstention strategies;
- temporal neural networks;
- drift detection;
- live stream ingestion;
- human operator evaluation;
- comparison of explanation models;
- retrieval-grounded telecom knowledge;
- API and dashboard deployment.

---

## 27. Citation

```bibtex
@misc{bouriga2026trustawareqoe,
  author       = {Mohamed Oussama Bouriga},
  title        = {A Trust-Aware Digital Twin for Early Prediction and
                  Explainable Detection of QoE Degradation in 5G Video Streaming},
  year         = {2026},
  institution  = {Aivancity School for Technology, Business and Society}
}
```

Also cite the original 5G-QoERA publication and repository.

---

## 28. Security and data notice

- Never commit `.env` or API keys.
- The 5G-QoERA dataset remains subject to its original license.
- Verify the repository license before public release.
- Do not redistribute credentials or private files.

---

## 29. Quick start

Existing generated project:

```bash
cd ~/Documents/5g_qoe_trust_twin
source .venv/bin/activate

python --version
python -m pytest -v
python scripts/final_evaluation_final.py
python scripts/generate_batch_explanations.py --cases 3
```

Corrected experiment from the audited manageable sample:

```bash
python scripts/build_features_final.py
python scripts/train_models_final.py
python scripts/calibrate_models_final.py
python scripts/apply_trust_final.py
python scripts/final_evaluation_final.py
python scripts/generate_batch_explanations.py --cases 30
python -m pytest -v
```
