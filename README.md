# Trust-Aware 5G QoE Digital Twin

A research implementation for **next-observation poor-QoE forecasting in 5G New Radio video streaming**.

The project combines offline chronological session reconstruction, past-only
temporal and cross-layer feature engineering, synthetic replay-parity checks,
future poor-QoE prediction, probability calibration, trust scoring and selective
abstention, structured LLM explanations, schema validation, grounding
validation, and publication-ready evaluation outputs.


---

## 1. Project objective

The system addresses the following research question:

> Can a trust-aware Digital Twin forecast poor video-streaming QoE at
> the next observed row from past-only features while producing calibrated
> probabilities, abstaining from unreliable predictions, and generating
> evidence-grounded operator explanations?

```text
5G-QoERA dataset
        ↓
Data inspection and preparation
        ↓
Long-format modelling dataset
        ↓
Audited resource-safe sample
        ↓
Offline chronological session reconstruction
        ↓
Past-only temporal and cross-layer features
        ↓
Synthetic replay-parity validation
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
Structured LLM explanation
        ↓
Pydantic schema validation
        ↓
Grounding validation
        ↓
Accepted or rejected explanation
```

---

## 2. Frozen results

The values in this section come from the frozen experiment and the validated
explanation review artifacts retained in this repository.

The pinned resource-safe input contains **994,496 rows**. Rebuilding sessions,
the next-observation target, prediction-time-available features, and split
boundaries produced **897,056 labelled rows**:

| Split | Rows | Sessions | Users | Poor-QoE rate |
|---|---:|---:|---:|---:|
| Train | 537,504 | 51,232 | 216 | 30.18% |
| Validation | 180,224 | 17,152 | 167 | 37.35% |
| Test | 179,328 | 18,144 | 181 | 41.43% |

The validation partition was divided chronologically: its first half fitted
sigmoid and isotonic calibrators, and its second half selected calibration and
decision thresholds. The Random Forest selected isotonic calibration and a
decision threshold of **0.37** before the test set was accessed.

All three models were evaluated on the same 179,328 sealed test rows:

| Model | Accuracy | F1 | PR-AUC | ROC-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.8675 | 0.8403 | 0.7719 | 0.8637 | 0.1325 | 0.1325 |
| Network Logistic Regression | 0.6749 | 0.6784 | 0.6459 | 0.7607 | 0.1941 | 0.0216 |
| Cross-layer Random Forest | **0.9012** | **0.8804** | **0.9548** | **0.9637** | **0.0715** | **0.0119** |

Final calibrated Random Forest results:

| Metric | All predictions | Accepted predictions |
|---|---:|---:|
| Accuracy | 0.9012 | 0.9344 |
| Precision | 0.8827 | 0.9305 |
| Recall | 0.8780 | 0.9095 |
| F1-score | 0.8804 | 0.9199 |
| PR-AUC | 0.9548 | 0.9681 |
| ROC-AUC | 0.9637 | 0.9747 |
| Brier score | 0.0715 | 0.0540 |
| Expected Calibration Error | 0.0119 | 0.0095 |

Trust results:

- validation-selected abstention threshold: **0.670912**;
- test coverage: **90.31%**;
- test abstention rate: **9.69%**;
- accepted-decision accuracy: **93.44%**;
- accepted-decision selective risk: **6.56%**.

The trust score is a heuristic reliability score, not a probability of
correctness.

Seed 42 selected ten accepted poor-QoE predictions, ten accepted acceptable-QoE
predictions, and ten abstentions. The prompts were exported for offline LLM
completion. Every retained abstention explanation states that evidence is
insufficient to support a specific cause or choose between poor and acceptable
QoE. The retained batch passed the same schema and grounding validators used by
the normal generation path:

- schema-valid rate: **100%**;
- grounding pass rate: **100%**;
- manually reviewed cases: **30**;
- factual-correctness rate: **100%**;
- completeness rate: **100%**;
- unsupported-statement rate: **0%**;
- contradiction rate: **0%**.

These human results come from one reviewer and a small, deliberately balanced
case set. They do not establish inter-rater reliability or population-wide LLM
performance. The external model and request metadata used for offline
completion were not retained, so the report does not attribute these outputs
to a specific provider or model.

---

## 3. Project structure

```text
5g_qoe_trust_twin/
├── configs/
├── data/
├── manifests/
├── models/
├── notebooks/
├── provenance/
├── report/
├── results/
├── scripts/
│   ├── prepare_data.py
│   ├── build_features.py
│   ├── train_models.py
│   ├── calibrate_models.py
│   ├── apply_trust.py
│   ├── final_evaluation.py
│   ├── generate_batch_explanations.py
│   └── evaluate_llm.py
├── src/qoe_twin/
├── templates/
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
- an OpenAI API key only for the optional online explanation client; offline
  prompt export and validation do not require one.

The experiment was developed and tested with Python 3.11.

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

Optional pinned dependency snapshot. The lock contains no developer-local
project path and uses platform markers for OS-specific packages; install the
project itself separately:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
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

Test the explanation layer without an API call:

```bash
python -m pytest \
  tests/test_explanation_generator.py \
  tests/test_prompt_builder.py \
  tests/test_grounding.py -q
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

## 11. Validate the resource-safe sample

The resource-safe workflow does not rebuild the 16.6-million-row intermediate
on a 16 GB machine. It requires the locally audited, immutable input:

```text
data/processed/friend_snapshot_quarantine/qoe_modeling_final_sample.parquet
SHA-256: 68ee536b88d294b9faa97b5e2c7d3eec4c46c07dc756d163b95ce32fa1d09100
```

The source remains labelled `quarantine` because its producing sampling command
is unavailable. Before using it, the feature build verifies its exact bytes and
schema, checks every row against the trusted raw TSV content, and discards all
inherited session and future-target calculations.

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

## 12. Build Digital Twin features

```bash
python scripts/build_features.py
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
python scripts/train_models.py
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
python scripts/calibrate_models.py
```

Compared methods:

- sigmoid;
- isotonic.

The validation partition is divided chronologically. Its first half fits both
calibrators, after removing labels that cross the internal midpoint. Its second
half selects the calibration method and decision threshold. In the frozen run,
isotonic calibration was selected for both learned models. The selected
decision thresholds were `0.35` for Network Logistic Regression and `0.37` for
the Random Forest. These choices were frozen before the test partition was
accessed.

Outputs:

```text
models/calibrated/
results/tables/final_calibration_comparison.csv
results/metrics/selected_calibration_configuration_final.json
```

---

## 15. Apply trust scoring and abstention

This stage ran after validation selected and froze both the model decision
threshold and the abstention threshold:

```bash
python scripts/apply_trust.py
```

Probability confidence is the normalized distance from the selected model
decision threshold. It is zero at that threshold and approaches one toward
either probability endpoint; it is not calculated around a fixed `0.50`.

The only authoritative trust equation is the four-term policy in
`configs/trust.yaml`:

```text
validation_performance = validation_F1
calibration_quality = 1 - validation_ECE

trust_score =
    0.40 * threshold_aware_probability_confidence
    + 0.30 * validation_performance
    + 0.20 * calibration_quality
    + 0.10 * data_quality
```

The weights are loaded from the strictly validated YAML configuration. There
is no alternative hardcoded production formula. The resulting score is a
**heuristic reliability score**, not a probability that a prediction is
correct.

The abstention threshold was selected using only validation
predictions, minimizing selective risk subject to at least 90% coverage. The
frozen threshold was `0.670911977555425`; it achieved 90.06% coverage on the
selection half and 90.31% coverage on the sealed test set. Trust-level labels
are descriptive and do not substitute for the validation-selected abstention
decision.

Outputs:

```text
results/predictions/final_trusted_predictions.parquet
results/tables/final_trust_summary.csv
```

---

## 16. Final evaluation

```bash
python scripts/final_evaluation.py
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

The file named `final_lead_time_histogram.pdf` summarizes elapsed time to the
next observation; it is not evidence of degradation-onset warning time.

---

## 17. Generate grounded LLM explanations

The explanation layer uses three separate prompts: poor QoE,
acceptable QoE, and abstention. Operational evidence contains only approved
fields available at prediction time. In particular, it excludes
`prediction_lead_seconds`, future timestamps, future MOS, targets, and other
future-derived information.

Every proposed cause must cite the exact evidence keys that support it. When
the evidence is insufficient, the explanation may state that limitation and
return an empty cause list. Grounding validation rejects obvious prediction
polarity contradictions, confident causal claims for abstentions, unsupported
numbers, incorrect field attribution, and incompatible units.

The explanation review used frozen predictions and deterministic seed 42 to
select exactly ten accepted poor-QoE predictions, ten accepted acceptable-QoE
predictions, and ten abstentions. No API was called by the repository during
this evaluation. The normal evidence and prompt builders exported the batch:

```bash
python scripts/generate_batch_explanations.py \
  --offline-prompt-export \
  --cases 30 \
  --seed 42
```

Only the `explanation` field was completed externally. The completed JSONL was
then imported and checked with the existing schema and grounding validators:

```bash
python scripts/generate_batch_explanations.py --import-completed-jsonl
```

The generated review sheet was completed by one human reviewer. Its binary
fields use `1` for correct/complete and `0` otherwise; unsupported-statement
and contradiction fields use `1` when the problem is present.

Outputs:

```text
results/explanations/explanation_prompts.jsonl
results/explanations/final_explanations.jsonl
results/explanations/manual_review_template.csv
results/explanations/final_manual_review.csv
results/metrics/llm_human_evaluation.json
results/tables/llm_human_evaluation.csv
```
The exported prompt contains only approved decision-time evidence fields and
excludes the target label, training data, unrestricted raw histories, and
retrospective lead-time information.

---

## 18. Publication material

Regenerate the report tables and figures from the authoritative frozen JSON:

```bash
python scripts/generate_report_assets.py
```

The generator validates consistency across final evaluation, three-model
comparison, calibration selection, split provenance, and the retained 30-case
human review before overwriting publication tables. This prevents stale values
from an earlier experiment from appearing in the report.

Build and audit the final IEEE PDF with Tectonic:

```bash
python scripts/build_report.py
```

If Tectonic is not on `PATH`, pass its executable explicitly:

```bash
python scripts/build_report.py --tectonic /path/to/tectonic
```

The build fails if the body before References exceeds 15 pages, required
sections or headline metrics are absent, or superseded metrics are detected.
The final PDF is written to:

```text
output/pdf/ieee_report_Bouriga_BenAissa_final.pdf
```

Use the frozen experiment and validated explanation-review artifacts directly:

```text
results/tables/final_model_comparison.csv
results/tables/final_final_metrics.csv
results/tables/final_trust_summary.csv
results/tables/final_rf_validation_risk_coverage.csv
results/tables/llm_human_evaluation.csv
results/metrics/final_final_evaluation.json
results/metrics/selected_calibration_configuration_final.json
results/metrics/llm_human_evaluation.json
results/figures/
```

---

## 19. Recommended complete execution order

```bash
source .venv/bin/activate

python -m pytest -v

python scripts/build_features.py
python scripts/train_models.py
python scripts/calibrate_models.py
python scripts/apply_trust.py
python scripts/final_evaluation.py

python scripts/generate_batch_explanations.py \
  --offline-prompt-export --cases 30 --seed 42
# Save externally completed explanations as final_explanations.jsonl, then:
python scripts/generate_batch_explanations.py --import-completed-jsonl
# Review manual_review_template.csv and save final_manual_review.csv, then:
python scripts/evaluate_llm.py

python scripts/generate_report_assets.py
python scripts/build_report.py

python -m pytest -v
```

Run expensive stages one at a time.

---

## 20. Inspect the frozen project

```bash
source .venv/bin/activate

python -m pytest -v
```

To validate a returned offline explanation batch without calling an API:

```bash
python scripts/generate_batch_explanations.py --import-completed-jsonl
python scripts/evaluate_llm.py
```

---

## 21. Main output paths

| Artifact | Path |
|---|---|
| Complete processed dataset | `data/processed/qoe_modeling_dataset.parquet` |
| Audited manageable input | `data/processed/friend_snapshot_quarantine/qoe_modeling_final_sample.parquet` |
| Final feature dataset | `data/processed/qoe_features_final.parquet` |
| Sample/split manifest | `manifests/cp6_sample_split_manifest.json` |
| Uncalibrated models | `models/uncalibrated/` |
| Calibrated models | `models/calibrated/` |
| Final trusted predictions | `results/predictions/final_trusted_predictions.parquet` |
| Final metrics | `results/metrics/final_final_evaluation.json` |
| Model comparison | `results/tables/final_model_comparison.csv` |
| Calibration comparison | `results/tables/final_calibration_comparison.csv` |
| Trust summary | `results/tables/final_trust_summary.csv` |
| Completed explanation cases | `results/explanations/final_explanations.jsonl` |
| Completed human-review sheet | `results/explanations/final_manual_review.csv` |
| Human-evaluation metrics | `results/metrics/llm_human_evaluation.json` |
| Human-evaluation table | `results/tables/llm_human_evaluation.csv` |
| Final figures | `results/figures/` |
| Final audited IEEE report | `output/pdf/ieee_report_Bouriga_BenAissa_final.pdf` |

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

Offline prompt export/import and review validation do not require an API key.
Only optional online explanation generation requires:

```dotenv
OPENAI_API_KEY=your_real_key
OPENAI_MODEL=gpt-4o-mini
```

Then run:

```bash
python scripts/generate_batch_explanations.py --cases 1
```

### Explanation schema failure

Do not use unrelated JSON such as `{"status": "working"}`. Validate the
explanation contract with:

```bash
python -m pytest \
  tests/test_explanation_schema.py \
  tests/test_grounding.py -q
```

### Grounding rejection

A rejection means an unsupported number or inconsistency was detected. Inspect:

```text
results/explanations/final_explanations.jsonl
```

### Memory pressure

Run `scripts/build_features.py` by itself with other memory-intensive
applications closed. Do not construct the full 16.6-million-row intermediate
on a 16 GB machine. Keep the configured Random Forest resource limits enabled.

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
- keep test labels and outcomes sealed until the frozen evaluation;
- retain complete sessions when sampling;
- never tune on the test set;
- never expose the target label to the LLM;
- never expose future-derived evidence to the LLM;
- treat trust as a heuristic reliability score, not probability of correctness;
- select abstention on validation with at least 90% coverage before test use;
- reject explanations failing schema or grounding checks;
- record dependency versions and random seeds.

---

## 25. Known limitations

- The target is a one-observation-ahead state forecast, not degradation onset;
  the elapsed interval between observations is irregular.
- The pinned sample is an earliest-time, group-balanced subset whose producing
  sampling command is unavailable; it is not a representative random sample.
- Four bitrate alternatives share each physical radio observation, so modelling
  rows are correlated and must not be treated as independent measurements.
- The global chronological split measures temporal generalization, not
  independent generalization to unseen users, sessions, or cells.
- Final features are reconstructed offline. Synthetic interleaved-session tests
  demonstrate batch/replay parity and future-invariance on a small test fixture,
  not through an end-to-end replay of every test row.
- Evaluation covers one dataset and one family of recorded scenarios.
- Random Forest feature importance is associative, not causal.
- The LLM evaluation covers 30 deliberately balanced cases reviewed by one
  person; there is no inter-rater study.
- The external model and request metadata for offline explanation completion
  were not retained, limiting generation reproducibility.
- This is a research decision-support prototype, not a production network-control system.

---

## 26. Future work

- additional 5G and 6G datasets;
- online recalibration;
- richer selective-risk analysis beyond the current risk-versus-coverage table;
- alternative abstention strategies;
- temporal neural networks;
- drift detection;
- live stream ingestion;
- inter-rater studies beyond the current manual review;
- comparison of explanation models;
- retrieval-grounded telecom knowledge;
- API and dashboard deployment.

---

## 27. Citation

```bibtex
@misc{bouriga2026trustawareqoe,
  author       = {Mohamed Oussama Bouriga},
  title        = {A Trust-Aware Digital Twin Prototype for Next-Observation
                  Poor-QoE Forecasting and Grounded Explanations in
                  5G Video Streaming},
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

Validate the frozen project and completed human review without rerunning the
experiment or calling an API:

```bash
cd ~/Documents/5g_qoe_trust_twin
source .venv/bin/activate

python --version
python -m pytest -v
python scripts/evaluate_llm.py
```

The full experiment is resource-intensive and should be rerun only
when an intentional reproduction is required:

```bash
python scripts/build_features.py
python scripts/train_models.py
python scripts/calibrate_models.py
python scripts/apply_trust.py
python scripts/final_evaluation.py
python scripts/generate_batch_explanations.py \
  --offline-prompt-export --cases 30 --seed 42
# Complete only each explanation field externally, then:
python scripts/generate_batch_explanations.py --import-completed-jsonl
python -m pytest -v
```
