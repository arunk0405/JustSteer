# JustSteer MLOps End-to-End Demonstration Runbook

## Overview
This runbook provides the presentation sequence for demonstrating the JustSteer dynamic multi-stop route optimization system. The system combines contextual traffic information, OSM-derived structural attributes, and explicitly synthetic safety risk data to produce dynamically optimized edge costs.

> **IMPORTANT PRESENTATION WARNINGS:**
> 
> * Do not claim that the historical traffic dataset contains street-level traffic observations for every graph edge.
> * The system combines contextual traffic/weather information with OSM-derived road attributes for edge-aware inference.
> * The safety model is trained using synthetic safety data and is presented as a demonstration of the MLOps safety pipeline, not as a validated real-world accident predictor.

---

## 1. Architecture Diagram

```text
                    ┌─────────────────────┐
                    │   Historical Data   │
                    │  Traffic Context    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ML Training       │
                    │   + OSM Enrichment  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Traffic Model       │
                    │ velocity_predictor  │
                    └──────────┬──────────┘
                               │
                               │
        ┌──────────────────────┴─────────────────────┐
        │                                            │
        ▼                                            ▼
┌───────────────┐                           ┌────────────────┐
│ OSM Graph     │                           │ Runtime        │
│ road_type     │                           │ Context        │
│ maxspeed      │                           │ weather/time   │
└───────┬───────┘                           └───────┬────────┘
        │                                           │
        └────────────────┬──────────────────────────┘
                         ▼
                ┌──────────────────┐
                │ Feature Resolver │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ ML Predictions   │
                │ Traffic + Safety │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Dynamic Edge Cost│
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ A* Routing       │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Multi-stop       │
                │ Optimization     │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ FastAPI Response │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ React + Leaflet  │
                └──────────────────┘
```

---

## 2. MLOps Demonstration Chain

The presentation demonstrates the full MLOps data lineage and inference lifecycle:

1. **DATA** - Historical context & synthetic safety logs
2. **DATA VALIDATION** - `verify_model_provenance.py` and `verify_safety_provenance.py`
3. **FEATURE ENGINEERING** - Preprocessing pipelines in `src.mlops`
4. **OSM STRUCTURAL ENRICHMENT** - Road types & free-flow speeds
5. **MODEL TRAINING** - `train.py` & `train_safety.py`
6. **MODEL EVALUATION** - MAE/RMSE metrics attached in metadata
7. **MODEL ARTIFACT** - Pickled `.pkl` models securely saved via atomic-writes
8. **MODEL METADATA** - Rich JSON metadata containing training signatures
9. **SHA-256 INTEGRITY** - Immutable artifact checksums evaluated before inference
10. **RUNTIME MODEL LOADING** - `TrafficPredictionProvider` & `SafetyPredictionProvider`
11. **FEATURE CONTRACT VALIDATION** - Automatic strict matching on the 19 (traffic) and 10 (safety) expected features via `FeatureResolver`
12. **INFERENCE** - Predictions executed on single OSM edges
13. **DYNAMIC ROUTING** - `EdgeCostStore` feeding custom dynamically-calculated cost functions to `AStarRoutingEngine`.

---

## 3. Demonstration Sequence

### Step 1: Start Backend
In a terminal at the repository root, start the API:
```bash
.\mlops\Scripts\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### Step 2: Run Health Check & Integrity Validations
In a separate terminal, execute the system health check to demonstrate reproducibility and provenance integrity:
```bash
.\mlops\Scripts\python scripts/system_health_check.py
```
*Note: This script will evaluate checksums, load the graph, run real edge predictions, and verify the Multi-stop/A* pipeline.*

You can also explicitly show the provenance verifiers:
```bash
.\mlops\Scripts\python scripts/verify_model_provenance.py
.\mlops\Scripts\python scripts/verify_safety_provenance.py
```

### Step 3: Start Frontend
Open a third terminal, install packages, and boot the frontend:
```bash
cd frontend
npm install
npm run dev
```

### Step 4: Open Homepage
Navigate to `http://localhost:5173/` in your browser to display the initial landing page.

### Step 5: Navigate to Route Planner
Click "Start Planning" or go to `/plan`.

### Step 6: Set Origin
On the map, click to place the "Origin" marker in Chennai.

### Step 7: Add Deliveries
Add 2–3 delivery stops within reasonable distance to the origin.

### Step 8: Generate Route
Click **"Generate Route"** on the sidebar. Observe the system communicating with the backend API.

### Step 9: Show End-to-End Resolution
Explain how the request flowed through the stack:
ML prediction → dynamic edge cost generation → A* routing → Multi-stop sequence optimization → FastAPI response → React Leaflet visualization.

### Step 10: Open Explainability Panel
In the frontend Sidebar, expand the Explainability/Stats panel.

### Step 11: Show Provenance
Highlight the dynamically served data:
- `Evaluated Permutations` and `Expanded Nodes`
- Travel time and Safety metrics (powered by XGBoost)
- Mention how weights dynamically control the edge cost evaluation.

### Step 12: Recovery & Error Handling (Optional)
To demonstrate resilience:
1. Try placing a marker far away in an unroutable area. See the friendly validation/error state.
2. Stop the FastAPI backend and submit a request to observe graceful error handling in the frontend.
3. Restart the backend and re-submit the request to see complete recovery with proper request isolation.
