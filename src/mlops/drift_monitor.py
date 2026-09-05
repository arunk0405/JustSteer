"""
Evidently AI Drift Monitor with GitHub Actions Webhook Trigger

This script is the core of the self-healing loop. It runs after data_ingestion.py
and answers the question: "Has the world changed enough that our model is no longer
reliable?" If yes, it fires a GitHub Actions workflow to retrain.

Drift Detection Strategy:
- Reference dataset: train.csv (the data the models were trained on)
- Current dataset: last 7 days from daily_features.db (the recent real world)
- Test statistic: Kolmogorov-Smirnov (KS) test for continuous features
  * KS tests whether two distributions are drawn from the same population
  * Null hypothesis (H0): reference and current come from the same distribution
  * If p-value < 0.05, we reject H0 — the distributions have drifted significantly
- Features monitored: 'rainfall' and 'traffic_speed_kmh' (most volatile inputs
  to the route optimization system — most likely to invalidate model weights)

Why KS over PSI (Population Stability Index)?
- KS gives a formal p-value — easy to threshold (0.05 is standard)
- PSI requires binning discretization choices that can obscure real drift
- KS works well for our small daily batches (no minimum bin-count requirement)

Why monitor only these 2 columns?
- rainfall and traffic_speed_kmh are the direct inputs to the edge weight formula
- They're the most sensitive to Chennai's monsoon seasonality
- Monitoring all columns increases false-positive rate (multiple testing problem)

Usage:
    python drift_monitor.py [--train-csv train.csv] [--lookback-days 7]

Scheduling (after data_ingestion.py):
    5 2 * * * /path/to/venv/bin/python /path/to/drift_monitor.py
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import requests
from evidently.metrics import ColumnDriftMetric, DataDriftTable
from evidently.report import Report

# ===========================
# Configuration
# ===========================

# Database path — must match data_ingestion.py
DB_PATH = Path("data/db/daily_features.db")

# Features to monitor for drift
# These are the columns whose drift should trigger a retrain
MONITORED_COLUMNS = ["rainfall", "traffic_speed_kmh"]

# Statistical threshold: reject H0 (no drift) if p-value < this
# 0.05 = 95% confidence that drift is real, not random noise
DRIFT_P_VALUE_THRESHOLD = 0.05

# How many days of recent data to compare against reference
LOOKBACK_DAYS = 7

# GitHub configuration — pulled from environment variables for security
# In GitHub Actions, set these as repository secrets
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")           # Must have repo:write scope
GITHUB_REPO_OWNER = os.getenv("GITHUB_REPO_OWNER") # e.g., "your-github-username"
GITHUB_REPO_NAME = os.getenv("GITHUB_REPO_NAME")   # e.g., "JustSteer"

# Report output directory
REPORT_DIR = Path("drift_reports")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/drift_monitor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ===========================
# Data Loading
# ===========================

def load_reference_data(train_csv_path: str) -> pd.DataFrame:
    """
    Load the reference dataset that the models were trained on (train.csv).
    
    This is the "before" snapshot — what the world looked like during training.
    Evidently will compare current data against this to detect if the world
    has shifted significantly enough to degrade model predictions.
    
    Args:
        train_csv_path: Path to train.csv file
    
    Returns:
        DataFrame with at minimum 'rainfall' and 'traffic_speed_kmh' columns
    
    Raises:
        FileNotFoundError: If train.csv doesn't exist
        KeyError: If required columns are missing
    """
    path = Path(train_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {path}\n"
            "Ensure train.py has been run at least once to produce train.csv."
        )
    
    df = pd.read_csv(path)
    
    # Verify required columns are present
    missing_cols = [col for col in MONITORED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Reference dataset is missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    logger.info(f"Loaded reference data: {len(df)} rows from {path}")
    return df


def load_current_data(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Load recent data from daily_features.db.
    
    We look back `lookback_days` days to get a statistically meaningful sample.
    Too short (1 day) = too few rows, high variance in KS statistic.
    Too long (30 days) = might mask the current drift with historical data.
    7 days is a good balance for Chennai's conditions.
    
    Args:
        lookback_days: Number of days to include in current dataset
    
    Returns:
        DataFrame with recent feature data
    
    Raises:
        FileNotFoundError: If daily_features.db doesn't exist
        ValueError: If not enough rows for a reliable KS test
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run data_ingestion.py first."
        )
    
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:00")
    
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            timestamp,
            location_key,
            rainfall,
            temperature_c,
            wind_speed_kmh,
            traffic_speed_kmh,
            feature_source
        FROM features
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn, params=(cutoff_str,))
    conn.close()
    
    logger.info(f"Loaded current data: {len(df)} rows from last {lookback_days} days")
    
    # Need at minimum ~24 rows for KS test to be meaningful
    # (KS is asymptotically valid for large samples; for small samples
    # it's conservative — p-values are over-estimated, reducing false positives)
    if len(df) < 24:
        logger.warning(
            f"Current dataset has only {len(df)} rows — KS test may be unreliable. "
            f"Consider reducing LOOKBACK_DAYS or checking data_ingestion.py logs."
        )
    
    return df


# ===========================
# Evidently Drift Detection
# ===========================

def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run Evidently AI drift detection comparing reference vs current data.
    
    Statistical approach: Kolmogorov-Smirnov test
    - KS test compares the empirical CDFs of two distributions
    - A large KS statistic (large distance between CDFs) indicates the samples
      come from different populations
    - The p-value tells us the probability of seeing this large a KS statistic
      by chance if H0 (same distribution) were true
    - We use p-value < 0.05 (standard significance level) to declare drift
    
    Why we monitor BOTH columns:
    - If rainfall has drifted (e.g., monsoon arrived) → Risk Classifier probabilities change
    - If traffic_speed has drifted (e.g., new road opened) → Edge Velocity Predictor errors grow
    - Drift in EITHER column is sufficient to retrain (conservative: false retrains
      are cheap, false negatives [missing real drift] are expensive)
    
    Args:
        reference_df: Training data (what the model knows)
        current_df: Recent live data (what the world looks like now)
    
    Returns:
        Tuple of:
        - drift_detected: True if at least one monitored column has drifted
        - drift_details: Dict with per-column metrics for the GitHub Actions payload
    """
    logger.info("Running Evidently AI drift analysis...")
    
    # Build Evidently report with column-level drift metrics for each monitored column
    # ColumnDriftMetric produces: drift_score (KS statistic), p_value, drift_detected flag
    # DataDriftTable gives a summary table across all columns (useful for the HTML report)
    metrics = [
        ColumnDriftMetric(column_name=col, stattest="ks", stattest_threshold=DRIFT_P_VALUE_THRESHOLD)
        for col in MONITORED_COLUMNS
    ]
    metrics.append(DataDriftTable())
    
    report = Report(metrics=metrics)
    
    # Run comparison: Evidently aligns reference and current by column name
    report.run(reference_data=reference_df, current_data=current_df)
    
    # Extract results as structured dict
    report_dict = report.as_dict()
    
    # Parse per-column drift results from the Evidently output structure
    # report_dict["metrics"] is a list matching the order of metrics[] above
    drift_details: Dict[str, Any] = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "lookback_days": LOOKBACK_DAYS,
        "p_value_threshold": DRIFT_P_VALUE_THRESHOLD,
        "columns": {}
    }
    
    any_drift = False
    
    for i, col in enumerate(MONITORED_COLUMNS):
        try:
            # Evidently returns metrics in the same order we passed them
            metric_result = report_dict["metrics"][i]["result"]
            
            drift_score = metric_result.get("drift_score")  # KS statistic (0–1)
            p_value = metric_result.get("p_value")          # Probability under H0
            col_drift = metric_result.get("drift_detected", False)
            
            drift_details["columns"][col] = {
                "ks_statistic": round(drift_score, 6) if drift_score is not None else None,
                "p_value": round(p_value, 6) if p_value is not None else None,
                "drift_detected": col_drift
            }
            
            if col_drift:
                any_drift = True
                logger.warning(
                    f"DRIFT DETECTED in '{col}': "
                    f"KS={drift_score:.4f}, p-value={p_value:.6f} < {DRIFT_P_VALUE_THRESHOLD}"
                )
            else:
                logger.info(
                    f"No drift in '{col}': "
                    f"KS={drift_score:.4f}, p-value={p_value:.6f} >= {DRIFT_P_VALUE_THRESHOLD}"
                )
        
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse drift result for column '{col}': {e}")
            drift_details["columns"][col] = {
                "ks_statistic": None,
                "p_value": None,
                "drift_detected": False,
                "error": str(e)
            }
    
    drift_details["any_drift_detected"] = any_drift
    
    # Save HTML report for human review
    _save_html_report(report)
    
    return any_drift, drift_details


def _save_html_report(report: Report) -> None:
    """Save the Evidently HTML report to disk for manual inspection."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"drift_report_{timestamp}.html"
    
    report.save_html(str(report_path))
    logger.info(f"Drift report saved: {report_path}")


# ===========================
# GitHub Actions Webhook Trigger
# ===========================

def trigger_retraining_webhook(drift_details: Dict[str, Any]) -> bool:
    """
    Trigger GitHub Actions workflow via repository_dispatch event.
    
    repository_dispatch is a GitHub API endpoint that allows external services
    to kick off a workflow run programmatically. This decouples the drift monitor
    (runs anywhere: locally, on a server, in a cron job) from the CI/CD
    infrastructure that actually runs the retrain.
    
    The `client_payload` travels into the workflow as:
        ${{ github.event.client_payload.drift_details }}
    
    allowing the workflow to log WHY it was triggered and annotate the MLflow run.
    
    Prerequisite:
    - Set GITHUB_TOKEN, GITHUB_REPO_OWNER, GITHUB_REPO_NAME as environment vars
    - The token must have repo:write scope (Fine-grained PAT or classic PAT)
    
    Args:
        drift_details: Dict with per-column KS statistics and p-values
    
    Returns:
        True on success (HTTP 204), False on failure
    """
    if not all([GITHUB_TOKEN, GITHUB_REPO_OWNER, GITHUB_REPO_NAME]):
        logger.error(
            "GitHub environment variables not set. "
            "Required: GITHUB_TOKEN, GITHUB_REPO_OWNER, GITHUB_REPO_NAME. "
            "Saving drift payload locally instead."
        )
        _save_drift_payload_locally(drift_details)
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/dispatches"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Build the repository_dispatch payload
    # `event_type` must match the one listed in the workflow YAML:
    #   on:
    #     repository_dispatch:
    #       types: [drift_detected]
    payload = {
        "event_type": "drift_detected",
        "client_payload": {
            "drift_details": drift_details,
            "triggered_at": datetime.utcnow().isoformat(),
            "trigger_source": "drift_monitor.py"
        }
    }
    
    logger.info(f"Triggering retraining webhook: POST {url}")
    logger.info(f"Drift summary: {json.dumps(drift_details['columns'], indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 204:
            # 204 No Content = GitHub accepted the dispatch (standard success response)
            logger.info("✓ Retraining workflow triggered successfully (HTTP 204)")
            return True
        
        elif response.status_code == 401:
            logger.error(
                f"Authentication failed (HTTP 401). "
                f"Check GITHUB_TOKEN has repo:write scope."
            )
        
        elif response.status_code == 404:
            logger.error(
                f"Repository not found (HTTP 404). "
                f"Check GITHUB_REPO_OWNER='{GITHUB_REPO_OWNER}' and "
                f"GITHUB_REPO_NAME='{GITHUB_REPO_NAME}'."
            )
        
        elif response.status_code == 422:
            logger.error(
                f"Workflow dispatch rejected (HTTP 422). "
                f"Ensure auto_retrain.yml has 'repository_dispatch' trigger "
                f"with event_type 'drift_detected'.\n"
                f"Response: {response.text}"
            )
        
        else:
            logger.error(f"Unexpected response: HTTP {response.status_code}\n{response.text}")
        
        return False
    
    except requests.RequestException as e:
        logger.error(f"Failed to trigger webhook: {e}")
        return False


def _save_drift_payload_locally(drift_details: Dict[str, Any]) -> None:
    """
    Fallback: save the drift payload to disk when GitHub credentials aren't available.
    Useful for local testing and CI pipelines that check the file.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = REPORT_DIR / f"drift_payload_{timestamp}.json"
    
    with open(out_path, "w") as f:
        json.dump(drift_details, f, indent=2)
    
    logger.info(f"Drift payload saved locally: {out_path}")


# ===========================
# Main Orchestration
# ===========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidently AI drift monitor for route optimization models")
    parser.add_argument(
        "--train-csv",
        default="data/raw/train.csv",
        help="Path to reference training dataset (default: data/raw/train.csv)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS,
        help=f"Days of recent data to compare (default: {LOOKBACK_DAYS})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run drift detection but do NOT trigger the webhook (for testing)"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main execution flow:
    1. Load reference dataset (train.csv)
    2. Load current dataset (daily_features.db, last 7 days)
    3. Run Evidently KS drift test on rainfall and traffic_speed_kmh
    4. If drift detected → trigger GitHub Actions repository_dispatch webhook
    
    Returns:
        Exit code: 0 on no drift or successful trigger, 1 on failure
    """
    args = parse_args()
    
    try:
        logger.info("=== Drift Monitor Starting ===")
        logger.info(f"Reference: {args.train_csv}")
        logger.info(f"Current: last {args.lookback_days} days from {DB_PATH}")
        logger.info(f"Monitored columns: {MONITORED_COLUMNS}")
        logger.info(f"Drift p-value threshold: {DRIFT_P_VALUE_THRESHOLD}")
        
        # Load datasets
        reference_df = load_reference_data(args.train_csv)
        current_df = load_current_data(lookback_days=args.lookback_days)
        
        # Run drift detection
        drift_detected, drift_details = detect_drift(reference_df, current_df)
        
        if not drift_detected:
            logger.info("✓ No significant drift detected — models remain valid.")
            logger.info("=== Drift Monitor Complete (no action needed) ===")
            return 0
        
        # Drift was detected — log the specifics
        logger.warning(f"⚠ Data drift detected!")
        for col, details in drift_details["columns"].items():
            if details.get("drift_detected"):
                logger.warning(
                    f"  {col}: KS={details.get('ks_statistic', 'N/A')}, "
                    f"p-value={details.get('p_value', 'N/A')}"
                )
        
        if args.dry_run:
            logger.info("[DRY RUN] Webhook trigger suppressed. Drift payload:")
            logger.info(json.dumps(drift_details, indent=2))
            logger.info("=== Drift Monitor Complete (dry-run, no webhook fired) ===")
            return 0
        
        # Trigger GitHub Actions workflow
        success = trigger_retraining_webhook(drift_details)
        
        if success:
            logger.info("=== Drift Monitor Complete — Retraining workflow triggered ===")
            return 0
        else:
            logger.error("=== Drift Monitor Complete — Webhook trigger failed ===")
            return 1
    
    except FileNotFoundError as e:
        logger.error(f"Missing required file: {e}")
        return 1
    
    except KeyError as e:
        logger.error(f"Schema mismatch: {e}")
        return 1
    
    except Exception as e:
        logger.exception(f"Drift monitor failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
