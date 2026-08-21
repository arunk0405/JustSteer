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
python ingest_cli.py --input sample_data/sample_delivery_data.csv

# Run it on YOUR dataset
python ingest_cli.py --input path/to/your_dataset.csv --target-col your_label_column
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
python ingest_cli.py --input data.csv --config configs/my_config.yaml
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
├── ingest_cli.py                    # CLI entry point
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

---

## Phase 2 — Context Enrichment (Zone-level Weather & Traffic)

Phase 2 adds area-level weather and road-corridor traffic context to every
delivery record, giving the model spatial and environmental signals it needs
for accurate route-risk prediction.

### What it adds

| New columns | Source |
|---|---|
| `zone_name`, `zone_code`, `zone_type`, `zone_tags` | Zone registry (static config) |
| `corridor_name`, `corridor_code` | Zone registry (nearest corridor) |
| `weather_condition`, `temp_celsius`, `humidity_pct` | OpenWeatherMap (OWM) |
| `wind_speed_mps`, `visibility_m`, `rain_1h_mm` | OWM |
| `rain_risk_score` | Computed 0–1 score (rain + NE monsoon multiplier) |
| `is_raining`, `is_foggy`, `is_high_wind`, `is_ne_monsoon` | Derived flags |
| `rain_forecast_hours` | OWM 3-hourly forecast (how many upcoming slots have rain) |
| `congestion_index`, `congestion_level` | TomTom Traffic Flow API |
| `current_speed_kmph`, `freeflow_speed_kmph` | TomTom |
| `travel_time_ratio` | Computed and clamped at 5.0 |
| `traffic_source` | `"tomtom"` or `"heuristic"` |

### Zone registry — Chennai

35 zones defined in `configs/zones/chennai.yaml`:
- **25 area zones** — Anna Nagar, T. Nagar, Adyar, Velachery, Chromepet,
  Sholinganallur, Ambattur, Tambaram, and more.
- **10 road corridors** — GST Road, Outer Ring Road, Anna Salai, OMR,
  ECR, IRR, Poonamallee High Road, NH48, Sardar Patel Road,
  Medavakkam–Sholinganallur Road.

Each zone has a centroid lat/lon, influence radius, semantic type, and
tags (e.g. `waterlogging_risk`, `peak_hour_congestion`). Corridors have
ordered waypoints used for traffic queries.

### Quick start

```bash
# 1. Run Phase 1 ingestion first (produces output/processed_data.csv)
python ingest_cli.py --input sample_data/sample_delivery_data.csv

# 2. Enrich with context (no API keys needed — uses dry_run automatically)
python -m src.context.enricher

# Output: output/enriched_data.csv
```

With real API keys set in `.env`:
```
OWM_API_KEY=your_openweathermap_key
TOMTOM_API_KEY=your_tomtom_key
```

then re-run the same command — it auto-detects the keys and switches to
live data.

### Configuration

All thresholds live in `configs/context_config.yaml`. Key settings:

| Setting | What it controls |
|---|---|
| `weather.cache_ttl_seconds` | How long a zone's weather is cached (default: 900s / 15 min) |
| `weather.rain_risk.*` | mm/hour thresholds for low / moderate / high / severe rain risk |
| `weather.ne_monsoon.risk_multiplier` | Rain risk multiplier during Oct–Dec NE monsoon season |
| `weather.forecast_hours` | How many hours ahead to fetch for pre-shift planning |
| `traffic.cache_ttl_seconds` | Traffic cache TTL (default: 300s / 5 min) |
| `traffic.congestion_levels.*` | Speed-ratio thresholds for free_flow / moderate / heavy / severe |
| `traffic.travel_time_ratio_max` | Clamp for travel_time_ratio outliers (default: 5.0) |
| `traffic.peak_hour_heuristic.*` | Synthetic congestion when no API key is set |
| `zone_registry.corridor_max_distance_km` | Max km to assign a corridor to a delivery point |

### Using the context module from Python

```python
from src.context import ZoneRegistry, WeatherFetcher, TrafficFetcher, ContextAggregator

registry   = ZoneRegistry("configs/zones/chennai.yaml")
weather    = WeatherFetcher()          # reads OWM_API_KEY from env
traffic    = TrafficFetcher()          # reads TOMTOM_API_KEY from env
aggregator = ContextAggregator(registry, weather, traffic)

# Single coordinate lookup
ctx = aggregator.get_context(lat=13.0067, lon=80.2572)
print(ctx.zone_name)                   # "Adyar"
print(ctx.weather.rain_risk_score)     # 0.0 – 1.0
print(ctx.traffic.congestion_level)    # "free_flow" | "moderate" | "heavy" | "severe"

# Flat dict for ML features
features = ctx.to_feature_dict()       # 24 keys, ready for DataFrame row

# Batch enrichment
df = aggregator.to_dataframe([(13.04, 80.23), (12.98, 80.22)])
```

### Phase 2 project structure additions

```
configs/
  context_config.yaml          # all Phase 2 thresholds
  zones/
    chennai.yaml                # 25 area zones + 10 road corridors
src/context/
  __init__.py                   # public API
  zone_registry.py              # loads zones.yaml, Haversine spatial lookup
  weather_fetcher.py            # OWM integration + rain_risk + NE monsoon + forecast
  traffic_fetcher.py            # TomTom integration + peak-hour heuristic
  context_aggregator.py         # coordinate → zone → weather + traffic context
  enricher.py                   # reads processed_data.csv, writes enriched_data.csv
tests/
  test_zone_registry.py         # 39 unit tests for the zone registry
  test_context_integration.py   # integration tests for the full Phase 2 stack
```
