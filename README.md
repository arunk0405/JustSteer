# Dynamic Data Ingestion Engine

Phase 1, sub-steps 1.1–1.6 of the route-optimization MLOps pipeline.

Accepts **any** tabular dataset (CSV, JSON, Parquet, or Excel), automatically
figures out what kind of data is in each column, cleans it appropriately for
that type, validates it against quality gates, and hands off a clean
DataFrame plus a full attribute-metadata JSON file — everything a model
training step (or a later drift-monitoring step) needs to know about the
data, without you hand-writing a schema for every new dataset.

## What it actually does, in order

1. **Load** — reads the file based on its extension, with CSV encoding
   fallback for messy scraped/Kaggle files.
2. **Profile** — for every column: pandas dtype, inferred semantic type
   (`numeric` / `categorical` / `boolean` / `datetime` / `identifier` /
   `text`), missing %, cardinality, sample values, and a suggested role
   (`feature` / `target` / `identifier` / `drop`).
3. **Clean** — type-appropriate missing-value imputation (median for
   numeric, mode for categorical), duplicate-row removal, optional
   IQR-based outlier capping. Identifier columns are never touched.
4. **Validate** — hard-fails on too few rows or a column missing beyond an
   unrecoverable threshold; warns on moderate missingness or schema drift
   versus a previously ingested version of the same kind of data.
5. **Export** — writes `processed_data.csv` and `schema_metadata.json` to
   the output directory.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Generate the bundled synthetic sample dataset (optional — one is already included)
python sample_data/generate_sample_data.py

# Run the engine on it
python main.py --input sample_data/sample_delivery_data.csv

# Run it on YOUR dataset
python main.py --input path/to/your_dataset.csv --target-col your_label_column
```

Output lands in `output/processed_data.csv` and `output/schema_metadata.json`
by default (override with `--output-dir`).

## Running the tests

```bash
python -m pytest -v
```

23 tests covering every module individually plus one end-to-end test that
runs the whole pipeline against the bundled sample dataset and checks the
real output files. All should pass out of the box — this was verified
before packaging.

## Configuration

Everything tunable lives in `configs/default_config.yaml` — thresholds for
what counts as an identifier vs. a category, missing-value imputation
strategy, quality gate limits, etc. Copy it and point `--config` at your
copy rather than editing `src/ingestion/config.py` directly:

```bash
python main.py --input data.csv --config configs/my_config.yaml
```

Key settings:

| Setting | What it controls |
|---|---|
| `id_cardinality_ratio` | How unique a column must be to be treated as an identifier |
| `categorical_max_unique` | Above this many distinct values, a text column is treated as free text, not a category |
| `numeric_impute_strategy` | `median` / `mean` / `drop_rows` |
| `categorical_impute_strategy` | `mode` / `constant` / `drop_rows` |
| `min_rows` | Hard-fails ingestion if the dataset is smaller than this |
| `max_missing_threshold_hard_fail` | A column missing more than this fraction fails validation outright, rather than getting imputed |
| `target_column` | Set explicitly to skip name-based auto-detection |
| `target_name_hints` | Column-name substrings used to auto-detect the target when not set explicitly |

## Project structure

```
.
├── main.py                          # CLI entry point
├── requirements.txt
├── pytest.ini
├── configs/
│   └── default_config.yaml
├── sample_data/
│   ├── generate_sample_data.py      # regenerates the bundled synthetic dataset
│   └── sample_delivery_data.csv
├── src/ingestion/
│   ├── __init__.py                  # public API: IngestionConfig, run_ingestion
│   ├── config.py                    # all tunable thresholds/strategies
│   ├── loaders.py                   # sub-step 1.1 — multi-format file loading
│   ├── profiler.py                  # sub-step 1.2 — attribute type/role inference
│   ├── cleaner.py                   # sub-step 1.3 — type-aware cleaning
│   ├── validator.py                 # sub-step 1.4 — quality gates + schema drift check
│   ├── exporter.py                  # writes processed_data.csv + schema_metadata.json
│   └── pipeline.py                  # sub-step 1.5 — orchestrates all of the above
└── tests/
    ├── test_loaders.py
    ├── test_profiler.py
    ├── test_cleaner.py
    ├── test_validator.py
    └── test_pipeline.py             # end-to-end test
```

## Using it from Python instead of the CLI

```python
from ingestion import IngestionConfig, run_ingestion

config = IngestionConfig(target_column="risk_label", output_dir="output")
result = run_ingestion("data/my_dataset.csv", config)

result.dataframe        # the cleaned pandas DataFrame — hand this to your model
result.profiles         # list[ColumnProfile] — attribute metadata per column
result.warnings         # any soft validation warnings
result.output_paths     # where processed_data.csv / schema_metadata.json were written
```

## Known, deliberate limitations

- **Auto-detected target column relies on name hints** (`risk`, `label`,
  `target`, `class`, `y`). For a dataset where the label doesn't match any
  hint, pass `--target-col` explicitly — don't rely on auto-detection for
  an unfamiliar dataset without checking `schema_metadata.json` afterward.
- **Type inference is heuristic, not guaranteed.** Always check the
  generated `schema_metadata.json` before training on a genuinely new
  dataset — it's designed to be read, not just trusted blindly.
- **Free-text columns are dropped, not vectorized.** If you need NLP
  features from a text column, that's a separate feature-engineering step
  downstream of this engine, not something it does automatically.
- **Schema-drift comparison is against the single most recent
  `schema_metadata.json`** in the output directory, not a full version
  history. Keep your own copies (or DVC-track the output directory) if you
  need to compare across many historical ingestions.

## How this fits into the larger MLOps pipeline

`schema_metadata.json`'s `feature_columns` list is exactly the column set
your training script should use, and `processed_data.csv` is what gets
DVC-tracked as `data/reference_training_data.csv` for the closed-loop
retraining pipeline described elsewhere in this project — the drift check
(`check_drift.py`) compares live data against this exact reference set and
feature list.
