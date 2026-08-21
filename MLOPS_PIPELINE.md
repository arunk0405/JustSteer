# Self-Healing MLOps Pipeline for Route Optimization

## Overview

This is a production-grade, **fully local and open-source** MLOps pipeline that automatically maintains two machine learning models for dynamic route optimization:

1. **XGBoost Road Risk Classifier** — predicts risk level (`low`, `medium`, `high`) based on weather and traffic
2. **Ridge Edge Velocity Predictor** — predicts traffic speed (km/h) for A* graph edge weights

The pipeline implements a **self-healing closed loop**:
```
Daily APIs → SQLite → Drift Detection → GitHub Actions → Retrain → Deploy
     ↑                                                                  ↓
     └──────────────────────────────────────────────────────────────────┘
```

When Evidently AI detects distribution drift in the live data (p-value < 0.05), it automatically triggers a GitHub Actions workflow that retrains both models, logs metrics to MLflow, and commits the updated model files back to the repository.

---

## Architecture

### Components

| Script | Purpose | Runs |
|---|---|---|
| **src/mlops/data_ingestion.py** | Fetch weather (Open-Meteo) + traffic (TomTom) data → SQLite | Daily cron (2 AM) |
| **src/mlops/drift_monitor.py** | Kolmogorov-Smirnov test on `rainfall`, `traffic_speed_kmh` → webhook trigger | Daily cron (2:05 AM) |
| **src/mlops/train.py** | Train XGBoost + Ridge → MLflow + export models/ + new train.csv | On-demand via GitHub Actions |
| **.github/workflows/auto_retrain.yml** | GitHub Actions retraining pipeline | Triggered by `repository_dispatch` |

### Data Flow

```
┌─────────────────────┐
│  Open-Meteo API     │  Chennai weather (rainfall, temp, wind)
│  TomTom Traffic API │  Bounding-box traffic speeds
└──────────┬──────────┘
           │ python -m src.mlops.data_ingestion (daily)
           ▼
┌─────────────────────┐
│ data/db/daily_features.db │  SQLite with UNIQUE(timestamp, location_key)
└──────────┬──────────┘
           │ python -m src.mlops.drift_monitor (daily after ingestion)
           ▼
┌─────────────────────┐
│ Evidently AI        │  KS test: reference (data/raw/train.csv) vs current (last 7 days)
│ ColumnDriftMetric   │  p-value < 0.05? → DRIFT DETECTED
└──────────┬──────────┘
           │ repository_dispatch webhook
           ▼
┌─────────────────────┐
│ GitHub Actions      │  auto_retrain.yml workflow
│                     │  → train.py → MLflow → git commit models/
└─────────────────────┘
```

---

## Installation

### Prerequisites

- **Python 3.11+**
- **Git** (for GitHub Actions integration)
- **TomTom API key** (free tier: 2,500 requests/day) — sign up at [developer.tomtom.com](https://developer.tomtom.com/)
- **GitHub Fine-grained PAT** (for `drift_monitor.py` to trigger workflows) — create at [github.com/settings/tokens](https://github.com/settings/tokens)
  - Required permissions: `Contents: read+write`, `Actions: write`

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/JustSteer.git
cd JustSteer

# 2. Create virtual environment
python -m venv mlops
source mlops/bin/activate  # Windows: mlops\Scripts\activate

# 3. Install dependencies
pip install -r requirements-mlops.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env and fill in:
#   TOMTOM_API_KEY=your_tomtom_api_key
#   GITHUB_TOKEN=ghp_your_github_pat
#   GITHUB_REPO_OWNER=your_github_username
#   GITHUB_REPO_NAME=JustSteer

# 5. Test the pipeline locally
python -m src.mlops.data_ingestion        # Populate data/db/daily_features.db
python -m src.mlops.train                 # Train initial models → models/, data/raw/train.csv
python -m src.mlops.drift_monitor --dry-run  # Test drift detection without webhook
```

---

## Usage

### Local Development

#### Run data ingestion
```bash
python -m src.mlops.data_ingestion
```
Creates `data/db/daily_features.db` with hourly weather + traffic features.

#### Run drift monitoring (dry-run, no webhook)
```bash
python -m src.mlops.drift_monitor --dry-run
```
Generates HTML drift report in `drift_reports/` without triggering GitHub Actions.

#### Train models manually
```bash
python -m src.mlops.train --triggered-by local_dev
```
Outputs:
- `models/risk_classifier.json` — XGBoost model
- `models/velocity_predictor.pkl` — sklearn Pipeline
- `data/raw/train.csv` — reference dataset for next drift check
- `mlruns/` — MLflow tracking directory

View MLflow UI:
```bash
mlflow ui --backend-store-uri mlruns
# Open: http://localhost:5000
```

---

### Production Deployment

#### 1. Schedule data ingestion (Linux/Mac cron)
```bash
crontab -e
# Add:
0 2 * * * cd /path/to/JustSteer && /path/to/mlops/bin/python -m src.mlops.data_ingestion
5 2 * * * cd /path/to/JustSteer && /path/to/mlops/bin/python -m src.mlops.drift_monitor
```

#### 2. Configure GitHub Actions
Set these as **repository secrets** in GitHub (Settings → Secrets and variables → Actions):
- `TOMTOM_API_KEY` — your TomTom API key
- `DISPATCH_TOKEN` — your GitHub PAT (for `drift_monitor.py` to call the webhook)

The workflow `.github/workflows/auto_retrain.yml` uses GitHub's built-in `GITHUB_TOKEN` for git commits (no manual setup needed).

#### 3. Push the workflow file
```bash
git add .github/workflows/auto_retrain.yml
git commit -m "feat: add auto-retrain GitHub Actions workflow"
git push
```

#### 4. Test the end-to-end flow
Manually trigger drift detection:
```bash
export GITHUB_TOKEN=ghp_your_pat_here
export GITHUB_REPO_OWNER=your_username
export GITHUB_REPO_NAME=JustSteer
python -m src.mlops.drift_monitor
```
If drift is detected, check the GitHub Actions tab — `auto_retrain.yml` should start automatically.

---

## Configuration

### src/mlops/data_ingestion.py
| Variable | Default | Description |
|---|---|---|
| `CHENNAI_LAT`, `CHENNAI_LON` | 13.0827, 80.2707 | Chennai coordinates for Open-Meteo |
| `CHENNAI_BBOX` | 12.95–13.15, 80.15–80.35 | TomTom traffic flow bounding box |
| `MAX_RETRIES` | 3 | HTTP retry attempts with exponential backoff |

### src/mlops/drift_monitor.py
| Variable | Default | Description |
|---|---|---|
| `MONITORED_COLUMNS` | `["rainfall", "traffic_speed_kmh"]` | Features checked for drift |
| `DRIFT_P_VALUE_THRESHOLD` | 0.05 | KS test significance level |
| `LOOKBACK_DAYS` | 7 | Days of current data to compare |

### src/mlops/train.py
| Variable | Default | Description |
|---|---|---|
| `MIN_TRAIN_ROWS` | 50 | Minimum rows required to train |
| `TEST_SIZE` | 0.2 | Train/test split fraction |
| `FEATURE_COLS` | `rainfall`, `temperature_c`, `wind_speed_kmh`, `hour_of_day`, `day_of_week`, `is_weekend`, `rainfall_lag1` | Input features for both models |

---

## Statistical Drift Detection (Explained)

### Why Kolmogorov-Smirnov (KS) Test?

The KS test compares two distributions by measuring the maximum distance between their **empirical cumulative distribution functions (CDFs)**. It answers: *"Are these two samples drawn from the same population?"*

**Null hypothesis (H₀):** Reference and current data come from the same distribution.

**KS statistic:** Maximum vertical distance between the two CDFs (0 to 1, where 1 = completely different).

**p-value:** Probability of observing this large a KS statistic by random chance if H₀ is true.

**Decision rule:** If `p-value < 0.05`, reject H₀ → **drift detected** (95% confidence the distributions differ).

### Why p-value = 0.05?

This is the standard **α = 0.05 significance level** (5% false positive rate) used in statistical hypothesis testing:
- **Type I error (false positive):** Declaring drift when none exists → unnecessary retrain (cheap cost)
- **Type II error (false negative):** Missing real drift → model degrades silently (**expensive cost**)

A 5% false-positive rate is acceptable because:
1. Retraining is cheap (GitHub Actions is free for public repos, ~\$0.008/minute for private repos)
2. Model staleness is expensive (bad route recommendations erode user trust)

### Why monitor only `rainfall` and `traffic_speed_kmh`?

These are the **two most volatile inputs** to the route optimization system:
- **Rainfall** directly affects road risk (the primary target of the Risk Classifier)
- **Traffic speed** directly determines edge weights (the primary target of the Velocity Predictor)

Monitoring all columns increases false positives due to the **multiple testing problem** (each additional test adds a 5% chance of spurious drift).

---

## Model Details

### XGBoost Road Risk Classifier

**Task:** Multi-class classification (`low`, `medium`, `high`)

**Hyperparameters:**
```python
n_estimators=200          # Number of boosting rounds
max_depth=4               # Tree depth (conservative to avoid overfitting)
learning_rate=0.05        # Slow learning for stable convergence
subsample=0.8             # Row subsampling for regularization
colsample_bytree=0.8      # Column subsampling
```

**Class imbalance handling:** Sample weights inversely proportional to class frequency.

**Evaluation metrics:**
- Accuracy (overall correctness)
- Per-class F1 scores (low, medium, high)
- Macro-averaged F1 (equal weight to all classes)

### Ridge Edge Velocity Predictor

**Task:** Regression (predict `traffic_speed_kmh`)

**Model:** `Pipeline([StandardScaler, Ridge(alpha=1.0)])`

**Why Ridge (not Lasso/ElasticNet)?**
- L2 regularization shrinks all coefficients toward zero without eliminating any (Lasso does feature selection, which we don't need — our feature set is already curated)
- Computationally cheaper than ElasticNet
- Closed-form solution → very fast to train

**Evaluation metrics:**
- MAE (Mean Absolute Error) in km/h — interpretable error magnitude
- RMSE (Root Mean Squared Error) — penalizes large outliers
- R² (coefficient of determination) — % of variance explained

---

## GitHub Actions Workflow Details

### Trigger: `repository_dispatch` Event

`src/mlops/drift_monitor.py` fires this webhook when drift is detected:
```bash
POST https://api.github.com/repos/{owner}/{repo}/dispatches
{
  "event_type": "drift_detected",
  "client_payload": {
    "drift_details": { ... },
    "triggered_at": "2026-07-24T11:00:00Z"
  }
}
```

The workflow listens for `event_type: drift_detected`:
```yaml
on:
  repository_dispatch:
    types: [drift_detected]
```

### Workflow Steps

1. **Checkout** — `actions/checkout@v4` with `fetch-depth: 0` (full git history)
2. **Log drift payload** — Parse `github.event.client_payload` and extract which columns drifted
3. **Setup Python 3.11** — `actions/setup-python@v5` with pip caching keyed on `requirements-mlops.txt`
4. **Install dependencies** — `pip install -r requirements-mlops.txt`
5. **Restore DB from cache** — `actions/cache@v4` restores `data/db/daily_features.db` (avoids re-downloading API data)
6. **Top-up ingestion** — Run `python -m src.mlops.data_ingestion` to fetch today's latest data (best-effort)
7. **Train models** — `python -m src.mlops.train --triggered-by drift_monitor_auto`
8. **Post-retrain drift baseline** — `python -m src.mlops.drift_monitor --dry-run` (generates HTML report, no webhook)
9. **Upload MLflow artifacts** — `actions/upload-artifact@v4` stores `mlruns/`, `models/`, `drift_reports/` for 30 days
10. **Commit updated models** — `git add models/ data/raw/train.csv && git commit && git push`
11. **Write job summary** — Populate the GitHub Actions Summary tab with metrics

### Concurrency Control

```yaml
concurrency:
  group: retrain
  cancel-in-progress: true
```
Prevents two retrains from running in parallel (avoids git commit race conditions).

---

## Troubleshooting

### `src/mlops/drift_monitor.py` fails with "Authentication failed (HTTP 401)"
**Cause:** `GITHUB_TOKEN` doesn't have `repo:write` scope.

**Fix:** Regenerate your GitHub Fine-grained PAT at [github.com/settings/tokens](https://github.com/settings/tokens) with:
- Repository access: Select `JustSteer`
- Permissions: `Contents: read+write`, `Actions: write`

### `src/mlops/data_ingestion.py` hangs on TomTom API call
**Cause:** TomTom free-tier rate limit (2,500 requests/day) exceeded.

**Fix:** Either:
1. Wait until tomorrow (rate limits reset at midnight UTC)
2. The script will fall back to synthetic traffic data (`speed = 40 km/h`) automatically

### `src/mlops/train.py` fails with "Only X usable rows — need at least 50"
**Cause:** Not enough data collected yet.

**Fix:** Run `python -m src.mlops.data_ingestion` multiple times (or let it run daily for a few days) until you have 50+ rows in `data/db/daily_features.db`.

### GitHub Actions workflow doesn't trigger
**Cause 1:** `.github/workflows/auto_retrain.yml` isn't pushed to the repository.
```bash
git add .github/workflows/auto_retrain.yml
git commit -m "feat: add auto-retrain workflow"
git push
```

**Cause 2:** `src/mlops/drift_monitor.py` uses the wrong `event_type`.
Ensure the script sends `"event_type": "drift_detected"` (must match the workflow YAML).

### MLflow UI shows "No experiments found"
**Cause:** `mlruns/` directory doesn't exist or is empty.

**Fix:** Run `python train.py` at least once to create the experiment.

---

## License

MIT License — see LICENSE file for details.

---

## Contributing

Pull requests are welcome. For major changes:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests if applicable
4. Commit with conventional commit messages (`feat:`, `fix:`, `docs:`)
5. Push and open a pull request

---

## Contact

**Author:** Senior MLOps Engineer  
**Project:** Route Optimization Self-Healing Pipeline  
**Documentation Date:** July 24, 2026
