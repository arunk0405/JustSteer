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
    Combine the existing train.csv (if present) with the latest rows from
    daily_features.db, deduplicate, and return a single DataFrame.

    Loading from both sources ensures the model sees:
    - The original curated training data (stable baseline)
    - The freshest real-world observations (captured by data_ingestion.py)

    Deduplication is on (timestamp, location_key) — same key as the DB's
    UNIQUE constraint, so no row gets counted twice.
    """
    frames = []

    # Load existing train.csv if present
    if TRAIN_CSV.exists():
        df_csv = pd.read_csv(TRAIN_CSV)
        logger.info(f"Loaded {len(df_csv)} rows from {TRAIN_CSV}")
        frames.append(df_csv)

    # Load all rows from SQLite
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df_db = pd.read_sql_query(
            "SELECT timestamp, location_key, rainfall, temperature_c, "
            "wind_speed_kmh, traffic_speed_kmh FROM features ORDER BY timestamp",
            conn,
        )
        conn.close()
        logger.info(f"Loaded {len(df_db)} rows from {DB_PATH}")
        frames.append(df_db)

    if not frames:
        raise FileNotFoundError(
            f"No data found. Neither {TRAIN_CSV} nor {DB_PATH} exists."
        )

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp", "location_key"])
    logger.info(f"Combined: {before} rows → {len(combined)} after deduplication")
    return combined



# ===========================
# Feature Engineering
# ===========================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw DB columns into model-ready features.

    Raw columns coming in:
        timestamp, location_key, rainfall, temperature_c,
        wind_speed_kmh, traffic_speed_kmh

    Engineered features added:
        hour_of_day   — captures rush-hour vs off-peak signal (0–23)
        day_of_week   — Mon=0 … Sun=6 (weekday vs weekend behavior)
        is_weekend    — binary flag derived from day_of_week
        rainfall_lag1 — rainfall from the previous hour (smooths noisy readings)

    Why lag features?
    - Traffic speed at hour H often depends on rainfall at H-1 (roads flood
      after sustained rain, not instantly during a single-reading spike).
    - A 1-step lag is the simplest form; extend to lag2, lag3 for richer models.
    """
    df = df.copy()

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time-based features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag feature: previous hour's rainfall (per location)
    df["rainfall_lag1"] = (
        df.groupby("location_key")["rainfall"]
        .shift(1)
        .fillna(df["rainfall"].median())  # fill first row per group with global median
    )

    # Drop rows that still have NaN in key feature columns after engineering
    required = ["rainfall", "temperature_c", "wind_speed_kmh", "traffic_speed_kmh"]
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} rows with NaN in required columns")

    logger.info(f"Feature engineering complete: {len(df)} usable rows")
    return df


def synthesise_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive risk_label if not already present in the dataset.

    Real training data should have ground-truth labels. This function provides
    a rule-based fallback for the synthetic/API-sourced rows that don't have
    them yet — it encodes domain knowledge about Chennai road risk:

        High risk:   heavy rain (>5 mm/h) AND slow traffic (<20 km/h)
        Medium risk: moderate rain (>2 mm/h) OR slow traffic (<30 km/h)
        Low risk:    everything else

    These thresholds should be validated against actual incident reports.
    When labeled data accumulates in the DB, remove this function and use
    the real labels directly.

    Returns:
        DataFrame with 'risk_label' column (string) and 'risk_label_enc' (int)
    """
    if "risk_label" in df.columns:
        logger.info("Using existing risk_label column from dataset")
    else:
        logger.warning(
            "No risk_label column found — synthesising labels from "
            "rainfall + traffic_speed thresholds (see synthesise_risk_labels docstring)"
        )

        def _assign_risk(row):
            if row["rainfall"] > 5.0 and row["traffic_speed_kmh"] < 20.0:
                return "high"
            elif row["rainfall"] > 2.0 or row["traffic_speed_kmh"] < 30.0:
                return "medium"
            return "low"

        df["risk_label"] = df.apply(_assign_risk, axis=1)

    # Encode string labels to integers for XGBoost
    df["risk_label_enc"] = df["risk_label"].map(RISK_LABEL_MAP)
    unknown = df["risk_label_enc"].isna()
    if unknown.any():
        logger.warning(
            f"Unknown risk labels found and dropped: "
            f"{df.loc[unknown, 'risk_label'].unique()}"
        )
        df = df[~unknown]

    df["risk_label_enc"] = df["risk_label_enc"].astype(int)
    return df



# ===========================
# Model Training
# ===========================

FEATURE_COLS = [
    "rainfall",
    "temperature_c",
    "wind_speed_kmh",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "rainfall_lag1",
]


def train_risk_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost multi-class Road Risk Classifier.

    XGBoost is chosen over a simpler model because:
    - The risk decision boundary is non-linear (rainfall AND speed interact)
    - XGBoost handles class imbalance well via scale_pos_weight / sample_weight
    - It is production-proven for tabular classification tasks

    Hyperparameters are conservative defaults suitable for our small dataset.
    For production, use Optuna or cross-validated grid search.

    Returns:
        (model, metrics_dict) for MLflow logging
    """
    logger.info("Training XGBoost Risk Classifier...")

    # Class weights to handle imbalanced risk distribution
    # (low risk rows are typically more frequent than high risk)
    class_counts = np.bincount(y_train)
    total = len(y_train)
    # Weight each sample inversely proportional to class frequency
    sample_weights = np.array([total / (len(class_counts) * class_counts[y]) for y in y_train])

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
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

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["low", "medium", "high"], output_dict=True)

    metrics = {
        "risk_classifier_accuracy": round(accuracy, 4),
        "risk_classifier_f1_low": round(report["low"]["f1-score"], 4),
        "risk_classifier_f1_medium": round(report["medium"]["f1-score"], 4),
        "risk_classifier_f1_high": round(report["high"]["f1-score"], 4),
        "risk_classifier_macro_f1": round(report["macro avg"]["f1-score"], 4),
    }

    logger.info(f"Risk Classifier → Accuracy: {accuracy:.4f}, Macro F1: {metrics['risk_classifier_macro_f1']:.4f}")
    return model, metrics


def train_velocity_predictor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a Sklearn Ridge Regression Edge Velocity Predictor.

    This predicts traffic_speed_kmh, which becomes the edge weight in the
    A* graph. Ridge regression is chosen because:
    - Speed prediction is approximately linear with rainfall/temperature
    - Ridge's L2 regularisation prevents overfitting on small daily batches
    - It is lightweight — fast to train and infer in the FastAPI hot path

    The model is wrapped in a sklearn Pipeline with StandardScaler so that
    feature magnitudes don't dominate the Ridge penalty term.

    Returns:
        (pipeline, metrics_dict) for MLflow logging
    """
    logger.info("Training Ridge Edge Velocity Predictor...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, fit_intercept=True)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "velocity_predictor_mae": round(mae, 4),
        "velocity_predictor_rmse": round(rmse, 4),
        "velocity_predictor_r2": round(r2, 4),
    }

    logger.info(f"Velocity Predictor → MAE: {mae:.4f} km/h, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return pipeline, metrics



# ===========================
# MLflow Logging & Artifact Export
# ===========================

def save_models(risk_model: XGBClassifier, velocity_model: Pipeline) -> Dict[str, str]:
    """
    Persist trained models to models/ directory.

    Saved as:
    - models/risk_classifier.json   — XGBoost native format (portable, version-safe)
    - models/velocity_predictor.pkl — sklearn Pipeline (joblib pickle)

    Returns dict of saved paths for MLflow artifact logging.
    """
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    risk_path = MODELS_DIR / "risk_classifier.json"
    velocity_path = MODELS_DIR / "velocity_predictor.pkl"

    risk_model.save_model(str(risk_path))
    joblib.dump(velocity_model, velocity_path)

    logger.info(f"Saved risk classifier → {risk_path}")
    logger.info(f"Saved velocity predictor → {velocity_path}")

    return {
        "risk_classifier": str(risk_path),
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
    - Parameters: model hyperparameters and dataset sizes
    - Metrics: accuracy, F1, MAE, RMSE, R²
    - Artifacts: model files and the new train.csv

    MLflow tracking URI is set to ./mlruns (local) by default.
    Override with MLFLOW_TRACKING_URI env var to point at a remote server.

    Returns:
        MLflow run_id for GitHub Actions output annotation
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(tags=run_tags) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # --- Parameters ---
        mlflow.log_params({
            "risk_classifier_n_estimators": 200,
            "risk_classifier_max_depth": 4,
            "risk_classifier_learning_rate": 0.05,
            "velocity_predictor_alpha": 1.0,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(train_df),
            "feature_cols": ",".join(FEATURE_COLS),
        })

        # --- Metrics ---
        mlflow.log_metrics({**risk_metrics, **velocity_metrics})

        # --- Artifacts ---
        mlflow.log_artifact(artifact_paths["risk_classifier"], artifact_path="models")
        mlflow.log_artifact(artifact_paths["velocity_predictor"], artifact_path="models")

        # Log the new train.csv so it's versioned alongside the run
        if TRAIN_CSV.exists():
            mlflow.log_artifact(str(TRAIN_CSV), artifact_path="data")

        # Log models using MLflow's native model flavors for serving compatibility
        mlflow.xgboost.log_model(risk_model, "risk_classifier_mlflow")
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

    # Allow CLI overrides for paths
    global DB_PATH, MODELS_DIR
    DB_PATH = Path(args.db)
    MODELS_DIR = Path(args.output_dir)

    try:
        logger.info("=== Training Pipeline Starting ===")

        # 1. Load and combine all available data
        raw_df = load_all_data()

        # 2. Engineer features
        df = engineer_features(raw_df)

        # 3. Synthesise or validate risk labels
        df = synthesise_risk_labels(df)

        if len(df) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Only {len(df)} usable rows — need at least {MIN_TRAIN_ROWS}. "
                f"Run data_ingestion.py to collect more data."
            )

        logger.info(f"Training on {len(df)} rows | Features: {FEATURE_COLS}")

        # 4. Split for evaluation
        X = df[FEATURE_COLS]
        y_risk = df["risk_label_enc"]
        y_speed = df["traffic_speed_kmh"]

        X_train, X_test, y_risk_train, y_risk_test, y_speed_train, y_speed_test = (
            train_test_split(
                X, y_risk, y_speed,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_risk,
            )
        )

        logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        # 5. Train models
        risk_model, risk_metrics = train_risk_classifier(
            X_train, y_risk_train, X_test, y_risk_test
        )
        velocity_model, velocity_metrics = train_velocity_predictor(
            X_train, y_speed_train, X_test, y_speed_test
        )

        # 6. Save model artifacts to disk
        artifact_paths = save_models(risk_model, velocity_model)

        # 7. Export new train.csv for next drift monitoring cycle
        export_cols = FEATURE_COLS + ["risk_label", "traffic_speed_kmh", "timestamp", "location_key"]
        export_cols = [c for c in export_cols if c in df.columns]
        df[export_cols].to_csv(TRAIN_CSV, index=False)
        logger.info(f"Exported new reference dataset → {TRAIN_CSV} ({len(df)} rows)")

        # 8. Log everything to MLflow
        run_tags = {
            "triggered_by": args.triggered_by,
            "train_timestamp": datetime.utcnow().isoformat(),
            "git_sha": os.getenv("GITHUB_SHA", "local"),
        }

        run_id = log_to_mlflow(
            risk_model, velocity_model,
            risk_metrics, velocity_metrics,
            df, artifact_paths, run_tags,
        )

        # Print summary for GitHub Actions step output
        print("\n=== Training Summary ===")
        print(f"MLflow Run ID : {run_id}")
        print(f"Risk Classifier   → Accuracy: {risk_metrics['risk_classifier_accuracy']}  |  Macro F1: {risk_metrics['risk_classifier_macro_f1']}")
        print(f"Velocity Predictor → MAE: {velocity_metrics['velocity_predictor_mae']} km/h  |  R²: {velocity_metrics['velocity_predictor_r2']}")
        print(f"Models saved to   : {MODELS_DIR}/")
        print(f"Reference CSV     : {TRAIN_CSV}")
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
