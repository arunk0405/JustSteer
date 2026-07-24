# Quick Start Guide — Self-Healing MLOps Pipeline

## 🚀 Get Running in 5 Minutes

### 1. Install dependencies
```bash
python -m venv mlops
source mlops/bin/activate  # Windows: mlops\Scripts\activate
pip install -r requirements-mlops.txt
```

### 2. Configure secrets
```bash
cp .env.example .env
# Edit .env with your API keys:
#   - TOMTOM_API_KEY (get free at developer.tomtom.com)
#   - GITHUB_TOKEN (create PAT at github.com/settings/tokens)
#   - GITHUB_REPO_OWNER and GITHUB_REPO_NAME
```

### 3. Run the pipeline locally
```bash
# Step 1: Ingest data
python data_ingestion.py
# → Creates daily_features.db with 24 hours of weather + traffic data

# Step 2: Train initial models
python train.py
# → Creates models/risk_classifier.json, models/velocity_predictor.pkl, train.csv
# → Logs metrics to mlruns/ (view with: mlflow ui)

# Step 3: Test drift detection (dry-run, no webhook)
python drift_monitor.py --dry-run
# → Generates drift_reports/drift_report_YYYYMMDD_HHMMSS.html
```

---

## 📊 What You Just Built

**Deliverable 1: Automated Data Ingestion** ✅
- `data_ingestion.py` — Queries Open-Meteo (Chennai weather) + TomTom (traffic speeds)
- Appends to `daily_features.db` (SQLite with deduplication)
- Retry logic, error handling, logging

**Deliverable 2: Drift Detection + Webhook** ✅
- `drift_monitor.py` — Evidently AI KS test on `rainfall` and `traffic_speed_kmh`
- p-value < 0.05 → triggers GitHub Actions `repository_dispatch` webhook
- Falls back to saving JSON locally if GITHUB_TOKEN not set

**Deliverable 3: GitHub Actions Retraining** ✅
- `.github/workflows/auto_retrain.yml` — Listens for `drift_detected` event
- Runs `train.py`, logs to MLflow, commits updated models/train.csv
- Job summary with drift details and metrics

---

## 🔄 The Self-Healing Loop (Visual)

```
┌────────────────────────────────────────────────────────────────┐
│                    DAILY AT 2:00 AM                            │
│                                                                │
│  data_ingestion.py → daily_features.db (new rows appended)    │
│          ↓                                                     │
│  drift_monitor.py (2:05 AM)                                    │
│    - Load train.csv (reference)                                │
│    - Load last 7 days from daily_features.db (current)         │
│    - Evidently ColumnDriftMetric (KS test)                     │
│    - If p-value < 0.05 for rainfall OR traffic_speed:         │
│        → POST github.com/repos/{owner}/{repo}/dispatches       │
│             { "event_type": "drift_detected", ... }            │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                  GITHUB ACTIONS (AUTO-TRIGGERED)               │
│                                                                │
│  .github/workflows/auto_retrain.yml                            │
│    1. Restore daily_features.db from cache                     │
│    2. python train.py --triggered-by drift_monitor_auto        │
│       - Train XGBoost RiskClassifier (low/medium/high)         │
│       - Train Ridge VelocityPredictor (traffic_speed_kmh)      │
│       - Log metrics to MLflow                                  │
│       - Export new train.csv (becomes reference for next drift)│
│    3. git commit models/ train.csv && git push                 │
└────────────────────────────────────────────────────────────────┘
                              ↓
                    MODELS AUTO-UPDATED ✅
```

---

## 🧪 Test the Webhook Trigger (Local → GitHub Actions)

```bash
# Ensure env vars are set:
export GITHUB_TOKEN=ghp_your_pat_here
export GITHUB_REPO_OWNER=your_username
export GITHUB_REPO_NAME=JustSteer

# Force a drift detection (will trigger if drift exists):
python drift_monitor.py

# If drift is detected, check:
# https://github.com/{owner}/{repo}/actions
# You should see "Auto Retrain on Drift" workflow running.
```

---

## 📁 Key Files Reference

| File | Purpose |
|---|---|
| `data_ingestion.py` | Daily API data fetch → SQLite |
| `drift_monitor.py` | Evidently KS test → webhook trigger |
| `train.py` | XGBoost + Ridge training → MLflow |
| `.github/workflows/auto_retrain.yml` | GitHub Actions retraining pipeline |
| `requirements-mlops.txt` | Python dependencies |
| `.env.example` | Secrets template (copy to .env) |
| `MLOPS_PIPELINE.md` | Full documentation |

---

## 🐛 Common Issues

### Issue: `drift_monitor.py` says "No risk_label column found"
**Fix:** Run `python train.py` first to generate `train.csv`.

### Issue: TomTom API returns 403 Forbidden
**Fix:** Set `TOMTOM_API_KEY` in `.env`. Script will fall back to synthetic traffic data if key is missing.

### Issue: GitHub Actions workflow doesn't start
**Fix:** Ensure `.github/workflows/auto_retrain.yml` is pushed to GitHub, and the PAT has `Actions: write` permission.

---

## 📖 Next Steps

1. **Schedule the pipeline in production:**
   ```bash
   crontab -e
   # Add:
   0 2 * * * cd /path/to/JustSteer && /path/to/mlops/bin/python data_ingestion.py
   5 2 * * * cd /path/to/JustSteer && /path/to/mlops/bin/python drift_monitor.py
   ```

2. **View MLflow experiments:**
   ```bash
   mlflow ui --backend-store-uri mlruns
   # Open http://localhost:5000
   ```

3. **Deploy models to FastAPI:**
   Load `models/risk_classifier.json` and `models/velocity_predictor.pkl` in your FastAPI `/predict` endpoint.

4. **Monitor drift reports:**
   Check `drift_reports/*.html` weekly to see which features are drifting over time.

---

## ✅ Success Checklist

- [ ] `daily_features.db` exists and has rows
- [ ] `train.csv` exists (created by `train.py`)
- [ ] `models/risk_classifier.json` and `models/velocity_predictor.pkl` exist
- [ ] `mlruns/` directory contains experiment logs
- [ ] `drift_reports/*.html` opens in browser and shows KS statistics
- [ ] GitHub Actions tab shows "Auto Retrain on Drift" workflow (even if not run yet)
- [ ] GitHub repository secrets `TOMTOM_API_KEY` and `DISPATCH_TOKEN` are set

---

**Read the full docs:** [MLOPS_PIPELINE.md](MLOPS_PIPELINE.md)
