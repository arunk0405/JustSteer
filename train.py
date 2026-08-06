"""
Model Training Script — XGBoost Risk Classifier + Sklearn Edge Velocity Predictor

This script is called by the GitHub Actions retraining workflow (auto_retrain.yml)
when drift is detected. It:
1. Loads the latest data from daily_features.db (+ any existing train.csv)
2. Engineers features (time-of-day, lag features, etc.)
3. Trains two models:
   a. XGBoost RiskClassifier  — predicts risk_label (low/medium/high)
   b. Sklearn Ridge EdgeVelocityPredictor — predicts traffic_speed_kmh
4. Evaluates both models and logs metrics + artifacts to MLflow
5. Saves models to models/ and exports the new train.csv for the next drift cycle

Why two models?
- The A* graph needs TWO pieces of information per edge:
  * Edge weight (travel cost) ← driven by predicted traffic_speed_kmh
  * Risk penalty (safety multiplier) ← driven by predicted risk_label
- Keeping them separate means you can retrain one without destabilising the other.

Usage:
    python train.py [--db daily_features.db] [--output-dir models]
"""

import argparse
import logging
import os
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ===========================
# Configuration
# ===========================

DB_PATH = Path("daily_features.db")
TRAIN_CSV = Path("train.csv")
MODELS_DIR = Path("models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT = "route-optimization-retrain"

# Risk label mapping for XGBoost (needs integer classes)
RISK_LABEL_MAP = {"low": 0, "medium": 1, "high": 2}
RISK_LABEL_INV = {v: k for k, v in RISK_LABEL_MAP.items()}

# Minimum rows required to train reliably
MIN_TRAIN_ROWS = 50

# Test split fraction
TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ===========================
# Data Loading
# ===========================

def load_all_data() -> pd.DataFrame:
    """
    Load training data from train.csv and/or daily_features.db.

    Now accepts enriched CSV schema (34 columns from enriched_data.csv / synthetic
    data) as well as the legacy DB schema. The enriched schema includes all Phase 2
    weather + traffic context features needed for the 19-feature model.

    Enriched CSV columns expected (from output/enriched_data.csv or synthetic):
        timestamp, zone_code, road_type, risk_label,
        rain_1h_mm, temp_celsius, humidity_pct, wind_speed_mps, visibility_m,
        rain_risk_score, rain_forecast_hours, is_raining, is_foggy,
        is_high_wind, is_ne_monsoon, zone_type,
        current_speed_kmph, freeflow_speed_kmph, congestion_index

    Legacy DB columns (daily_features.db):
        timestamp, location_key, rainfall, temperature_c, wind_speed_kmh,
        traffic_speed_kmh

    If a CSV has enriched columns, they are used directly. If it has legacy
    columns, they are mapped. Rows from both sources are combined and deduplicated
    on (timestamp, location_key).

    Returns:
        DataFrame with harmonised columns ready for engineer_features()
    """
    frames = []

    # ── Load train.csv (enriched or legacy) ─────────────────────────────────
    if TRAIN_CSV.exists():
        df_csv = pd.read_csv(TRAIN_CSV)
        logger.info(f"Loaded {len(df_csv)} rows from {TRAIN_CSV}")

        # Detect schema by checking for enriched columns
        if "rain_1h_mm" in df_csv.columns and "congestion_index" in df_csv.columns:
            logger.info("Detected enriched CSV schema (Phase 2 columns)")
            # Enriched schema already has everything; just rename zone_code -> location_key
            if "zone_code" in df_csv.columns and "location_key" not in df_csv.columns:
                df_csv["location_key"] = df_csv["zone_code"]
        else:
            logger.info("Detected legacy CSV schema — will add missing enriched columns downstream")

        frames.append(df_csv)

    # ── Load daily_features.db (legacy schema only) ──────────────────────────
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df_db = pd.read_sql_query(
            "SELECT timestamp, location_key, rainfall, temperature_c, "
            "wind_speed_kmh, traffic_speed_kmh FROM features ORDER BY timestamp",
            conn,
        )
        conn.close()
        logger.info(f"Loaded {len(df_db)} rows from {DB_PATH} (legacy schema)")
        frames.append(df_db)

    if not frames:
        raise FileNotFoundError(
            f"No data found. Neither {TRAIN_CSV} nor {DB_PATH} exists."
        )

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp", "location_key"])
    logger.info(f"Combined: {before} rows -> {len(combined)} after deduplication")
    return combined



# ===========================
# Feature Engineering
# ===========================

# Zone type ordinal encoding — residential areas tolerate slower speeds;
# commercial/industrial zones have different baseline congestion patterns.
ZONE_TYPE_MAP = {
    "residential": 0,
    "commercial":  1,
    "industrial":  2,
    "coastal":     3,
    "highway":     4,
    "urban_road":  5,
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw columns into the full 19-feature model-ready set.

    Handles two input schemas transparently:

    Enriched schema (synthetic / enriched_data.csv):
        timestamp, location_key, rain_1h_mm, temp_celsius, humidity_pct,
        wind_speed_mps, visibility_m, rain_risk_score, rain_forecast_hours,
        is_raining, is_foggy, is_high_wind, is_ne_monsoon, zone_type,
        current_speed_kmph, freeflow_speed_kmph, congestion_index,
        traffic_speed_kmh (= current_speed_kmph for velocity target)

    Legacy schema (daily_features.db):
        timestamp, location_key, rainfall, temperature_c, wind_speed_kmh,
        traffic_speed_kmh

    Output — 19 features:
    ┌─────────────────────────┬─────────────────────────────────────────────┐
    │ Feature                 │ Signal                                      │
    ├─────────────────────────┼─────────────────────────────────────────────┤
    │ rain_1h_mm              │ Instantaneous precipitation intensity       │
    │ temperature_c           │ Heat stress, road surface behaviour         │
    │ humidity_pct            │ Road wetness persistence, tyre grip         │
    │ wind_speed_kmh          │ Lateral force on vehicles / two-wheelers   │
    │ visibility_m            │ Strongest single fog / haze signal          │
    │ rain_risk_score         │ Pre-computed 0–1 composite risk (+ monsoon) │
    │ rain_forecast_hours     │ Forward-looking rain — route planning       │
    │ is_raining              │ Clean binary split point for trees          │
    │ is_foggy                │ Visibility < 1000 m binary flag             │
    │ is_high_wind            │ Wind > 10 m/s binary flag                  │
    │ is_ne_monsoon           │ Oct–Dec structural risk uplift              │
    │ congestion_index        │ current/freeflow ratio — road-normalised    │
    │ freeflow_speed_kmph     │ Road capacity baseline                      │
    │ hour_of_day             │ AM/PM peak-hour signal (0–23)               │
    │ day_of_week             │ Weekday vs weekend pattern (0–6)            │
    │ is_weekend              │ Binary weekend flag                         │
    │ month                   │ Annual seasonal cycle (1–12)                │
    │ zone_type_enc           │ Spatial category (residential/commercial…)  │
    │ road_type_enc           │ Road class (highway/arterial/residential…)  │
    └─────────────────────────┴─────────────────────────────────────────────┘
    """
    df = df.copy()

    # ── Timestamp ────────────────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Temporal features ─────────────────────────────────────────────────────
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = df["timestamp"].dt.month

    # ── Schema bridge: map legacy columns -> enriched names ───────────────────
    # Legacy: rainfall -> rain_1h_mm
    if "rain_1h_mm" not in df.columns:
        df["rain_1h_mm"] = df.get("rainfall", pd.Series(0.0, index=df.index))
    elif "rainfall" in df.columns:
        df["rain_1h_mm"] = df["rain_1h_mm"].fillna(df["rainfall"]).fillna(0.0)
    else:
        df["rain_1h_mm"] = df["rain_1h_mm"].fillna(0.0)

    # Legacy: temperature_c stays as-is; temp_celsius -> temperature_c
    if "temperature_c" not in df.columns:
        df["temperature_c"] = df["temp_celsius"] if "temp_celsius" in df.columns else 28.0
    else:
        if "temp_celsius" in df.columns:
            df["temperature_c"] = df["temperature_c"].fillna(df["temp_celsius"])
        df["temperature_c"] = df["temperature_c"].fillna(28.0)

    # wind_speed_kmh — DB has it natively; enriched CSV only has wind_speed_mps
    if "wind_speed_kmh" not in df.columns:
        if "wind_speed_mps" in df.columns:
            df["wind_speed_kmh"] = df["wind_speed_mps"] * 3.6
        else:
            df["wind_speed_kmh"] = 0.0
    else:
        # Column exists but enriched rows will have NaN (DB col not in CSV)
        if "wind_speed_mps" in df.columns:
            df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(df["wind_speed_mps"] * 3.6)
        df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(0.0)

    # traffic_speed_kmh — regression target; current_speed_kmph in enriched source
    if "traffic_speed_kmh" not in df.columns:
        if "current_speed_kmph" in df.columns:
            df["traffic_speed_kmh"] = df["current_speed_kmph"]
        else:
            df["traffic_speed_kmh"] = np.nan
    else:
        if "current_speed_kmph" in df.columns:
            df["traffic_speed_kmh"] = df["traffic_speed_kmh"].fillna(df["current_speed_kmph"])

    # ── Enriched columns — fill defaults for legacy rows ─────────────────────
    def _fill(col, default):
        if col not in df.columns:
            df[col] = default

    _fill("humidity_pct",         65.0)
    _fill("visibility_m",         10000)
    _fill("rain_risk_score",      0.0)
    _fill("rain_forecast_hours",  0)
    _fill("is_raining",           0)
    _fill("is_foggy",             0)
    _fill("is_high_wind",         0)
    _fill("is_ne_monsoon",        df["month"].isin([10, 11, 12]).astype(int)
                                  if "month" in df.columns else 0)
    _fill("congestion_index",     0.75)   # moderate as neutral default
    _fill("freeflow_speed_kmph",  40.0)   # Chennai urban road average

    # ── Derived flags for legacy rows that may be missing them ───────────────
    # Only recompute if the column was just filled with default 0
    if (df["is_raining"] == 0).all() and (df["rain_1h_mm"] > 0).any():
        df["is_raining"] = (df["rain_1h_mm"] > 0.1).astype(int)

    if (df["is_foggy"] == 0).all() and (df["visibility_m"] < 10000).any():
        df["is_foggy"] = (df["visibility_m"] < 1000).astype(int)

    if (df["is_high_wind"] == 0).all() and (df["wind_speed_kmh"] > 0).any():
        df["is_high_wind"] = (df["wind_speed_kmh"] > 36.0).astype(int)  # 36 km/h = 10 m/s

    # ── Zone type encoding ────────────────────────────────────────────────────
    if "zone_type" in df.columns:
        df["zone_type_enc"] = (
            df["zone_type"].str.lower().map(ZONE_TYPE_MAP).fillna(0).astype(int)
        )
    else:
        df["zone_type_enc"] = 0   # default: residential

    # ── Road type encoding ────────────────────────────────────────────────────
    ROAD_TYPE_MAP = {"highway": 0, "arterial": 1, "residential": 2, "service": 3}
    if "road_type" in df.columns:
        df["road_type_enc"] = (
            df["road_type"].str.lower().map(ROAD_TYPE_MAP).fillna(2).astype(int)
        )
    else:
        df["road_type_enc"] = 2   # default: residential

    # ── Drop rows missing critical feature columns ────────────────────────────
    # traffic_speed_kmh is the regression target — only drop it if the column
    # exists and has NaN, but do NOT require it here (velocity model handles
    # missing targets separately in main()).
    required = ["rain_1h_mm", "temperature_c", "wind_speed_kmh"]
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} rows with NaN in required columns")

    logger.info(
        f"Feature engineering complete: {len(df)} usable rows | "
        f"Schema: {'enriched' if 'congestion_index' in df.columns else 'legacy'}"
    )
    return df


def synthesise_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate or synthesise risk_label using the full 19-feature context.

    If 'risk_label' already exists (enriched/synthetic data), it is used
    directly — no overwrite.

    For legacy rows without labels, the fallback rule now uses congestion_index
    AND rain_risk_score (not just raw rainfall + speed thresholds), which gives
    a more accurate label assignment consistent with the enriched schema:

        High:   (rain_risk_score >= 0.5  OR  congestion_index < 0.40)
                AND at least one of: is_foggy, is_high_wind, heavy rain
        Medium: rain_risk_score >= 0.2  OR  congestion_index < 0.60
                OR  traffic_speed_kmh < 30
        Low:    everything else

    Returns:
        DataFrame with 'risk_label' (str) and 'risk_label_enc' (int) columns.
    """
    if "risk_label" in df.columns:
        # Clean up any NaN labels before encoding
        df = df.dropna(subset=["risk_label"])
        logger.info(
            f"Using existing risk_label column | "
            f"Distribution: {df['risk_label'].value_counts().to_dict()}"
        )
    else:
        logger.warning(
            "No risk_label column — synthesising from enriched feature thresholds"
        )

        def _assign_risk(row) -> str:
            ci   = row.get("congestion_index",   0.75)
            rrs  = row.get("rain_risk_score",     0.0)
            fog  = row.get("is_foggy",            0)
            wind = row.get("is_high_wind",        0)
            spd  = row.get("traffic_speed_kmh",   40.0)
            rain = row.get("rain_1h_mm",          0.0)

            if (rrs >= 0.5 or ci < 0.40) and (fog or wind or rain > 10.0):
                return "high"
            elif rrs >= 0.2 or ci < 0.60 or spd < 30.0:
                return "medium"
            return "low"

        df["risk_label"] = df.apply(_assign_risk, axis=1)

    # Encode string labels to integers for XGBoost
    df["risk_label_enc"] = df["risk_label"].map(RISK_LABEL_MAP)
    unknown = df["risk_label_enc"].isna()
    if unknown.any():
        logger.warning(
            f"Unknown risk labels dropped: {df.loc[unknown, 'risk_label'].unique()}"
        )
        df = df[~unknown]

    df["risk_label_enc"] = df["risk_label_enc"].astype(int)
    return df



# ===========================
# Model Training
# ===========================

FEATURE_COLS = [
    # ── Weather (continuous) ──────────────────────────────────────────────────
    "rain_1h_mm",           # instantaneous precipitation intensity
    "temperature_c",        # ambient temperature (road surface behaviour)
    "humidity_pct",         # road wetness persistence, tyre grip
    "wind_speed_kmh",       # lateral force on vehicles / two-wheelers
    "visibility_m",         # strongest fog / haze hazard signal
    "rain_risk_score",      # composite 0–1 score (rain + NE monsoon multiplier)
    "rain_forecast_hours",  # forward-looking rain — route planning horizon
    # ── Weather (binary flags) ────────────────────────────────────────────────
    "is_raining",           # clean binary split point for trees
    "is_foggy",             # visibility < 1000 m
    "is_high_wind",         # wind > 10 m/s (36 km/h)
    "is_ne_monsoon",        # Oct–Dec structural risk uplift
    # ── Traffic ───────────────────────────────────────────────────────────────
    "congestion_index",     # current/freeflow ratio — road-normalised speed
    "freeflow_speed_kmph",  # road capacity baseline (distinguishes road class)
    # ── Temporal ──────────────────────────────────────────────────────────────
    "hour_of_day",          # AM/PM peak signal (0–23)
    "day_of_week",          # weekday pattern (0=Mon … 6=Sun)
    "is_weekend",           # binary weekend flag
    "month",                # annual seasonal cycle (1–12)
    # ── Spatial ───────────────────────────────────────────────────────────────
    "zone_type_enc",        # residential=0 / commercial=1 / industrial=2 / coastal=3
    "road_type_enc",        # highway=0 / arterial=1 / residential=2 / service=3
]


def train_risk_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost multi-class Road Risk Classifier on the 19-feature set.

    XGBoost is chosen because:
    - Risk boundaries are non-linear (congestion_index AND rain_risk_score
      interact; fog alone vs fog + rain produce very different risk profiles)
    - Handles class imbalance via per-sample inverse-frequency weights
    - Built-in feature importance for interpretability / drift debugging

    Metrics logged:
        accuracy, per-class F1 (low/medium/high), macro F1,
        weighted F1, ROC-AUC (macro OvR), per-class precision & recall

    Returns:
        (model, metrics_dict) for MLflow logging
    """
    logger.info("Training XGBoost Risk Classifier (19 features)...")

    # Inverse-frequency sample weights to handle class imbalance
    class_counts = np.bincount(y_train)
    total = len(y_train)
    sample_weights = np.array(
        [total / (len(class_counts) * class_counts[y]) for y in y_train]
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import label_binarize

    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=["low", "medium", "high"],
        output_dict=True,
    )

    # ROC-AUC (macro OvR) — threshold-independent ranking quality
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    metrics = {
        # Overall
        "risk_accuracy":            round(accuracy, 4),
        "risk_macro_f1":            round(report["macro avg"]["f1-score"], 4),
        "risk_weighted_f1":         round(report["weighted avg"]["f1-score"], 4),
        "risk_roc_auc_macro":       round(roc_auc, 4),
        # Per-class F1
        "risk_f1_low":              round(report["low"]["f1-score"], 4),
        "risk_f1_medium":           round(report["medium"]["f1-score"], 4),
        "risk_f1_high":             round(report["high"]["f1-score"], 4),
        # Per-class precision
        "risk_precision_low":       round(report["low"]["precision"], 4),
        "risk_precision_medium":    round(report["medium"]["precision"], 4),
        "risk_precision_high":      round(report["high"]["precision"], 4),
        # Per-class recall
        "risk_recall_low":          round(report["low"]["recall"], 4),
        "risk_recall_medium":       round(report["medium"]["recall"], 4),
        "risk_recall_high":         round(report["high"]["recall"], 4),
    }

    logger.info(
        f"Risk Classifier | Accuracy={accuracy:.4f}  Macro-F1={metrics['risk_macro_f1']:.4f}"
        f"  ROC-AUC={metrics['risk_roc_auc_macro']:.4f}"
    )
    logger.info(
        f"  Per-class F1 -> low={metrics['risk_f1_low']:.4f}  "
        f"medium={metrics['risk_f1_medium']:.4f}  high={metrics['risk_f1_high']:.4f}"
    )

    # Log top feature importances
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"  Top-5 features: {top5}")

    return model, metrics


def train_velocity_predictor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a Ridge Regression Edge Velocity Predictor on the 19-feature set.

    Predicts traffic_speed_kmh — the edge weight in the A* routing graph.
    Ridge is retained over a tree model because speed is largely additive
    across features (rain degrades speed by roughly a fixed fraction;
    peak hour adds another fixed penalty), and L2 regularisation prevents
    overfitting when real data is sparse.

    The pipeline wraps StandardScaler so that the wider feature set (which
    now includes 0/1 flags alongside continuous values like visibility_m)
    does not distort the Ridge penalty.

    Metrics logged:
        MAE, RMSE, R², MAPE, Max-Error
        — together they cover average accuracy (MAE/MAPE), variance
          sensitivity (RMSE), explained variance (R²), and worst-case
          edge weight error (Max-Error)

    Returns:
        (pipeline, metrics_dict) for MLflow logging
    """
    logger.info("Training Ridge Edge Velocity Predictor (19 features)...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0, fit_intercept=True)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae       = mean_absolute_error(y_test, y_pred)
    mse       = mean_squared_error(y_test, y_pred)
    rmse      = np.sqrt(mse)
    r2        = r2_score(y_test, y_pred)
    max_err   = float(np.max(np.abs(y_test.values - y_pred)))

    # MAPE — mean absolute percentage error (skip zero-speed rows to avoid div/0)
    nonzero   = y_test.values != 0
    mape      = float(np.mean(np.abs((y_test.values[nonzero] - y_pred[nonzero])
                                      / y_test.values[nonzero])) * 100)

    metrics = {
        "velocity_mae_kmh":   round(mae,     4),
        "velocity_rmse_kmh":  round(rmse,    4),
        "velocity_r2":        round(r2,      4),
        "velocity_mape_pct":  round(mape,    4),
        "velocity_max_err_kmh": round(max_err, 4),
    }

    logger.info(
        f"Velocity Predictor | MAE={mae:.2f} km/h  RMSE={rmse:.2f}  "
        f"R2={r2:.4f}  MAPE={mape:.2f}%  MaxErr={max_err:.2f} km/h"
    )
    return pipeline, metrics



# ===========================
# MLflow Logging & Artifact Export
# ===========================

def save_models(risk_model: XGBClassifier, velocity_model: Pipeline) -> Dict[str, str]:
    """
    Persist trained models to models/ directory using joblib.

    Saved as:
    - models/risk_classifier.pkl    — XGBoost model via joblib (avoids XGBoost
                                      2.x _estimator_type bug with save_model())
    - models/velocity_predictor.pkl — sklearn Pipeline via joblib

    MLflow's xgboost.log_model() handles its own serialisation separately.

    Returns dict of saved paths for MLflow artifact logging.
    """
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    risk_path     = MODELS_DIR / "risk_classifier.pkl"
    velocity_path = MODELS_DIR / "velocity_predictor.pkl"

    joblib.dump(risk_model,     risk_path)
    joblib.dump(velocity_model, velocity_path)

    logger.info(f"Saved risk classifier    -> {risk_path}")
    logger.info(f"Saved velocity predictor -> {velocity_path}")

    return {
        "risk_classifier":    str(risk_path),
        "velocity_predictor": str(velocity_path),
    }


def log_to_mlflow(
    risk_model: XGBClassifier,
    velocity_model: Pipeline,
    risk_metrics: Dict[str, Any],
    velocity_metrics: Dict[str, Any],
    train_df: pd.DataFrame,
    artifact_paths: Dict[str, str],
    run_tags: Dict[str, str],
) -> str:
    """
    Log everything to MLflow:
    - Parameters : model hyperparameters, feature list, dataset sizes
    - Metrics    : all 19 risk + velocity metrics
    - Artifacts  : model files, train.csv, feature_importance.csv

    MLflow tracking URI defaults to ./mlruns (local).
    Override with MLFLOW_TRACKING_URI env var for remote server.

    Returns:
        MLflow run_id
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(tags=run_tags) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # ── Parameters ────────────────────────────────────────────────────────
        mlflow.log_params({
            "risk_n_estimators":      300,
            "risk_max_depth":         5,
            "risk_learning_rate":     0.05,
            "risk_subsample":         0.8,
            "risk_colsample_bytree":  0.8,
            "risk_min_child_weight":  3,
            "risk_gamma":             0.1,
            "velocity_ridge_alpha":   1.0,
            "test_size":              TEST_SIZE,
            "random_state":           RANDOM_STATE,
            "train_rows":             len(train_df),
            "n_features":             len(FEATURE_COLS),
            "feature_cols":           ",".join(FEATURE_COLS),
        })

        # ── Metrics ───────────────────────────────────────────────────────────
        mlflow.log_metrics({**risk_metrics, **velocity_metrics})

        # ── Feature importance CSV ─────────────────────────────────────────────
        fi = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": risk_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        fi_path = MODELS_DIR / "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path), artifact_path="models")

        # ── Model artifacts ───────────────────────────────────────────────────
        mlflow.log_artifact(artifact_paths["risk_classifier"],  artifact_path="models")
        mlflow.log_artifact(artifact_paths["velocity_predictor"], artifact_path="models")

        if TRAIN_CSV.exists():
            mlflow.log_artifact(str(TRAIN_CSV), artifact_path="data")

        # MLflow native model flavours for serving
        # Note: mlflow.sklearn.log_model works for XGBClassifier since it
        # implements the sklearn estimator interface. mlflow.xgboost.log_model
        # is avoided here due to a known XGBoost 2.x _estimator_type issue.
        mlflow.sklearn.log_model(risk_model,     "risk_classifier_mlflow")
        mlflow.sklearn.log_model(velocity_model, "velocity_predictor_mlflow")

        logger.info(f"MLflow run complete: {run_id}")

    return run_id


# ===========================
# Main Orchestration
# ===========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train route optimization models")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to daily_features.db")
    parser.add_argument("--output-dir", default=str(MODELS_DIR), help="Directory to save models")
    parser.add_argument("--triggered-by", default="manual", help="Trigger source for MLflow tags")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    global DB_PATH, MODELS_DIR
    DB_PATH    = Path(args.db)
    MODELS_DIR = Path(args.output_dir)

    try:
        logger.info("=== Training Pipeline Starting (19-feature schema) ===")

        # 1. Load data (enriched CSV or legacy DB)
        raw_df = load_all_data()

        # 2. Engineer all 19 features
        df = engineer_features(raw_df)

        # 3. Validate / synthesise risk labels
        df = synthesise_risk_labels(df)

        if len(df) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Only {len(df)} usable rows — need at least {MIN_TRAIN_ROWS}. "
                f"Run data_ingestion.py or generate synthetic data."
            )

        logger.info(
            f"Training on {len(df)} rows | "
            f"{len(FEATURE_COLS)} features | "
            f"Risk dist: {df['risk_label'].value_counts().to_dict()}"
        )

        # 4. Build feature matrix and targets
        # Ensure all FEATURE_COLS are present (fill any remaining gaps)
        for col in FEATURE_COLS:
            if col not in df.columns:
                logger.warning(f"Feature '{col}' missing — filling with 0")
                df[col] = 0

        X       = df[FEATURE_COLS].fillna(0)
        y_risk  = df["risk_label_enc"]
        y_speed = df["traffic_speed_kmh"].fillna(df["traffic_speed_kmh"].median())

        X_train, X_test, y_risk_train, y_risk_test, y_speed_train, y_speed_test = (
            train_test_split(
                X, y_risk, y_speed,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_risk,
            )
        )

        logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        # 5. Train both models
        risk_model,     risk_metrics     = train_risk_classifier(
            X_train, y_risk_train, X_test, y_risk_test
        )
        velocity_model, velocity_metrics = train_velocity_predictor(
            X_train, y_speed_train, X_test, y_speed_test
        )

        # 6. Save artifacts to disk
        artifact_paths = save_models(risk_model, velocity_model)

        # 7. Export reference train.csv for next drift cycle
        # Keep all 19 feature cols + target cols + identifiers
        export_cols = (
            FEATURE_COLS
            + ["risk_label", "traffic_speed_kmh", "timestamp", "location_key"]
        )
        export_cols = [c for c in export_cols if c in df.columns]
        df[export_cols].to_csv(TRAIN_CSV, index=False)
        logger.info(f"Exported reference dataset -> {TRAIN_CSV} ({len(df)} rows)")

        # 8. Log to MLflow
        run_tags = {
            "triggered_by":    args.triggered_by,
            "train_timestamp": datetime.utcnow().isoformat(),
            "git_sha":         os.getenv("GITHUB_SHA", "local"),
            "n_features":      str(len(FEATURE_COLS)),
            "schema":          "enriched_v2",
        }

        run_id = log_to_mlflow(
            risk_model, velocity_model,
            risk_metrics, velocity_metrics,
            df, artifact_paths, run_tags,
        )

        # 9. Print summary
        print("\n=== Training Summary ===")
        print(f"MLflow Run ID      : {run_id}")
        print(f"Features used      : {len(FEATURE_COLS)}")
        print(f"Training rows      : {len(X_train)}  |  Test rows: {len(X_test)}")
        print()
        print("Risk Classifier (XGBoost)")
        print(f"  Accuracy         : {risk_metrics['risk_accuracy']}")
        print(f"  Macro F1         : {risk_metrics['risk_macro_f1']}")
        print(f"  Weighted F1      : {risk_metrics['risk_weighted_f1']}")
        print(f"  ROC-AUC (macro)  : {risk_metrics['risk_roc_auc_macro']}")
        print(f"  F1  low/med/high : "
              f"{risk_metrics['risk_f1_low']} / "
              f"{risk_metrics['risk_f1_medium']} / "
              f"{risk_metrics['risk_f1_high']}")
        print()
        print("Velocity Predictor (Ridge)")
        print(f"  MAE              : {velocity_metrics['velocity_mae_kmh']} km/h")
        print(f"  RMSE             : {velocity_metrics['velocity_rmse_kmh']} km/h")
        print(f"  R2               : {velocity_metrics['velocity_r2']}")
        print(f"  MAPE             : {velocity_metrics['velocity_mape_pct']}%")
        print(f"  Max Error        : {velocity_metrics['velocity_max_err_kmh']} km/h")
        print()
        print(f"Models saved to    : {MODELS_DIR}/")
        print(f"Reference CSV      : {TRAIN_CSV}")
        print("=== Training Complete ===\n")

        return 0

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Training aborted: {e}")
        return 1

    except Exception as e:
        logger.exception(f"Training failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
