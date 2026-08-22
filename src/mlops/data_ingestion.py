"""
Automated Data Ingestion Script for Route Optimization MLOps Pipeline

This script runs daily to fetch fresh features from external APIs and append them
to a local SQLite database for drift monitoring and model retraining.

Data Sources:
1. Open-Meteo API — Chennai hourly weather (rainfall, temperature, wind)
2. TomTom Traffic Flow API — Bounding-box average traffic speeds

Design:
- Deduplication via composite unique constraint on (timestamp, location_key)
- Robust error handling with retry logic for API failures
- Schema validation before insertion
- Logging for production monitoring

Usage:
    python data_ingestion.py
    
Scheduling (Linux/Mac cron):
    0 2 * * * /path/to/venv/bin/python /path/to/data_ingestion.py

Scheduling (Windows Task Scheduler):
    Trigger: Daily at 2:00 AM
    Action: Start program → python.exe
    Arguments: data_ingestion.py
"""

import argparse
import logging
import sqlite3
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import List, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()  # Load TOMTOM_API_KEY and other vars from .env file

# ===========================
# Configuration
# ===========================

# Chennai coordinates for Open-Meteo
CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707

# TomTom Traffic Flow API - Chennai bounding box (adjust as needed)
# Covers roughly 10km x 10km area around Chennai city center
CHENNAI_BBOX = {
    "min_lat": 12.95,
    "max_lat": 13.15,
    "min_lon": 80.15,
    "max_lon": 80.35
}

# API Configuration
TOMTOM_API_KEY =os.getenv("TOMTOM_API_KEY")
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Database
DB_PATH = Path("data/db/daily_features.db")

# IST offset — used to align all timestamps to Asia/Kolkata
IST_OFFSET = timedelta(hours=5, minutes=30)

# Retry configuration for API calls
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # 2, 4, 8 seconds

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/data_ingestion.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===========================
# Database Setup
# ===========================

def init_database() -> None:
    """
    Initialize SQLite database with schema.
    
    Schema:
    - timestamp: ISO8601 string (hourly granularity)
    - location_key: string identifier (e.g., "chennai_center", "chennai_north")
    - rainfall: mm per hour
    - temperature_c: degrees Celsius
    - wind_speed_kmh: km/h
    - traffic_speed_kmh: average speed from TomTom
    - feature_source: 'api' for live data, 'synthetic' for backfill
    - created_at: insertion timestamp for audit trail
    
    Composite unique constraint prevents duplicate (timestamp, location_key) rows.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            location_key TEXT NOT NULL,
            rainfall REAL,
            temperature_c REAL,
            wind_speed_kmh REAL,
            traffic_speed_kmh REAL,
            feature_source TEXT DEFAULT 'api',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, location_key)
        )
    """)
    
    # Index for faster drift monitoring queries (temporal range scans)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON features(timestamp)
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


# ===========================
# Dynamic Lookback Window
# ===========================

def get_dynamic_hours_lookback(default: int = 24) -> int:
    """
    Compute how many hours of data to fetch based on what is already in the DB.

    Queries MAX(timestamp) from the features table.  The timestamps stored
    there are IST strings (e.g. "2026-08-10T14:00").  We compare that against
    the current IST time and ask for data from that point forward, plus a
    one-hour overlap buffer to avoid gaps from API rounding.

    Falls back to ``default`` hours when:
    - The database does not exist yet (first run).
    - The features table is empty.
    - The latest timestamp cannot be parsed.

    Args:
        default: Fallback lookback in hours (default 24).

    Returns:
        Number of hours to pass to the Open-Meteo and TomTom fetchers.
    """
    if not DB_PATH.exists():
        logger.info(f"DB not found — using default lookback of {default}h")
        return default

    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT MAX(timestamp) FROM features").fetchone()
        conn.close()

        if not row or row[0] is None:
            logger.info(f"DB is empty — using default lookback of {default}h")
            return default

        latest_ts_str = row[0]  # e.g. "2026-08-10T14:00"
        latest_ts = datetime.fromisoformat(latest_ts_str)

        # Current IST time (naive, same reference as stored timestamps)
        now_ist = datetime.now(timezone.utc).astimezone(
            timezone(IST_OFFSET)
        ).replace(tzinfo=None)

        hours_since_latest = (now_ist - latest_ts).total_seconds() / 3600

        if hours_since_latest < 0.5:
            logger.info("DB is already up-to-date for this half-hour — nothing new to fetch")
            return 1  # Fetch just 1 hour so the script exits cleanly

        # Add 1-hour overlap buffer to avoid gaps from API rounding
        hours_needed = max(1, int(hours_since_latest) + 1)
        logger.info(
            f"Latest DB timestamp: {latest_ts_str} IST — "
            f"fetching last {hours_needed}h (gap: {hours_since_latest:.1f}h)"
        )
        return hours_needed

    except Exception as exc:
        logger.warning(f"Could not compute dynamic lookback ({exc}) — using default {default}h")
        return default


# ===========================
# HTTP Session with Retry Logic
# ===========================

def create_retry_session() -> requests.Session:
    """
    Create a requests session with exponential backoff retry logic.
    
    Retries on:
    - 5xx server errors (transient failures)
    - 429 rate limit (with backoff)
    - Connection errors
    
    Does NOT retry on:
    - 4xx client errors (except 429) — these indicate bad requests
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# ===========================
# Open-Meteo Weather Data
# ===========================

def fetch_weather_data(session: requests.Session, hours_lookback: int = 24) -> List[Dict[str, Any]]:
    """
    Fetch hourly weather data from Open-Meteo API for Chennai.
    
    Open-Meteo is free, no API key required, and provides:
    - Hourly precipitation (mm)
    - Temperature (°C)
    - Wind speed (km/h)
    
    Args:
        session: HTTP session with retry logic
        hours_lookback: Number of past hours to fetch (default 24 for daily batch)
    
    Returns:
        List of dicts: [{"timestamp": "2026-07-24T00:00", "rainfall": 2.5, ...}, ...]
    
    Raises:
        requests.RequestException: If API call fails after retries
    """
    # Calculate time range (Open-Meteo expects ISO8601 date strings)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(hours=hours_lookback)
    
    params = {
        "latitude": CHENNAI_LAT,
        "longitude": CHENNAI_LON,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "Asia/Kolkata"
    }
    
    logger.info(f"Fetching Open-Meteo data for Chennai ({CHENNAI_LAT}, {CHENNAI_LON})")
    
    try:
        response = session.get(OPEN_METEO_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Parse hourly data
        hourly = data.get("hourly", {})
        timestamps = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        winds = hourly.get("wind_speed_10m", [])
        
        # Filter out future timestamps since we only want historical data and current hour
        now_ist_str = datetime.now(timezone.utc).astimezone(timezone(IST_OFFSET)).strftime("%Y-%m-%dT%H:00")

        records = []
        for i, ts in enumerate(timestamps):
            if ts > now_ist_str:
                continue
            records.append({
                "timestamp": ts,  # ISO8601 format: "2026-07-24T00:00"
                "location_key": "chennai_center",
                "rainfall": precip[i] if i < len(precip) else None,
                "temperature_c": temps[i] if i < len(temps) else None,
                "wind_speed_kmh": winds[i] if i < len(winds) else None,
            })
        
        logger.info(f"Fetched {len(records)} weather records from Open-Meteo")
        return records
    
    except requests.RequestException as e:
        logger.error(f"Open-Meteo API call failed: {e}")
        raise


# ===========================
# TomTom Traffic Data
# ===========================

def fetch_traffic_data(session: requests.Session, hours_lookback: int = 24) -> List[Dict[str, Any]]:
    """
    Fetch traffic flow data from TomTom Traffic Flow API for Chennai bounding box.
    
    TomTom Traffic Flow API provides real-time average speeds.
    NOTE: TomTom requires an API key (free tier: 2,500 requests/day).
    Sign up at: https://developer.tomtom.com/
    
    This implementation:
    1. Queries multiple sample points within the Chennai bounding box
    2. Averages the speeds to get a city-wide "traffic_speed_kmh" metric
    3. Generates one record per hour for the past 24 hours
    
    LIMITATION: TomTom's free API returns *current* flow, not historical hourly data.
    For a production system with historical requirements, you'd need to:
    - Run this script hourly (not daily) to capture snapshots, OR
    - Use TomTom's Historical Traffic API (paid tier), OR
    - Store snapshots yourself over time
    
    For this MVP, we'll fetch the current speed and replicate it for the past 24 hours
    as a proxy (acknowledging this is synthetic historical data).
    
    Args:
        session: HTTP session with retry logic
        hours_lookback: Number of past hours to generate records for
    
    Returns:
        List of dicts: [{"timestamp": "2026-07-24T00:00", "traffic_speed_kmh": 45.2}, ...]
    
    Raises:
        requests.RequestException: If API call fails after retries
    """
    if not TOMTOM_API_KEY or TOMTOM_API_KEY == "YOUR_TOMTOM_API_KEY":
        logger.warning("TomTom API key not configured. Using synthetic traffic data (speed=40 km/h).")
        return _generate_synthetic_traffic(hours_lookback)
    
    # Sample 4 points within the Chennai bounding box for spatial averaging
    sample_points = [
        (13.00, 80.20),  # South-West
        (13.00, 80.30),  # South-East
        (13.10, 80.20),  # North-West
        (13.10, 80.30),  # North-East
    ]
    
    speeds = []
    
    for lat, lon in sample_points:
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {
            "key": TOMTOM_API_KEY,
            "point": f"{lat},{lon}",
            "unit": "KMPH"
        }
        
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract current speed
            flow_data = data.get("flowSegmentData", {})
            current_speed = flow_data.get("currentSpeed")
            
            if current_speed is not None:
                speeds.append(current_speed)
                logger.debug(f"TomTom point ({lat}, {lon}): {current_speed} km/h")
            
            # Rate limiting: TomTom free tier has limits
            sleep(0.5)
        
        except requests.RequestException as e:
            logger.warning(f"TomTom API call failed for point ({lat}, {lon}): {e}")
            continue
    
    if not speeds:
        logger.error("All TomTom API calls failed. Falling back to synthetic data.")
        return _generate_synthetic_traffic(hours_lookback)
    
    avg_speed = sum(speeds) / len(speeds)
    logger.info(f"TomTom average traffic speed: {avg_speed:.2f} km/h (from {len(speeds)} points)")
    
    # Generate hourly records (limitation acknowledged: replicating current speed)
    # Use IST time to match the timestamps returned by Open-Meteo
    # (which is queried with timezone=Asia/Kolkata).
    records = []
    now_ist = datetime.now(timezone.utc) + IST_OFFSET   # naive IST

    for hour_offset in range(hours_lookback):
        ts = now_ist - timedelta(hours=hours_lookback - 1 - hour_offset)
        records.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:00"),
            "location_key": "chennai_center",
            "traffic_speed_kmh": round(avg_speed, 2)
        })
    
    logger.info(f"Generated {len(records)} traffic records")
    return records


def _generate_synthetic_traffic(hours_lookback: int) -> List[Dict[str, Any]]:
    """
    Fallback: Generate synthetic traffic data when TomTom API is unavailable.
    Uses a baseline speed of 40 km/h (typical urban average).
    Timestamps are in IST to match Open-Meteo's timezone=Asia/Kolkata output.
    """
    records = []
    now_ist = datetime.now(timezone.utc) + IST_OFFSET   # naive IST

    for hour_offset in range(hours_lookback):
        ts = now_ist - timedelta(hours=hours_lookback - 1 - hour_offset)
        records.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:00"),
            "location_key": "chennai_center",
            "traffic_speed_kmh": 40.0  # Baseline synthetic value
        })
    
    logger.info(f"Generated {len(records)} synthetic traffic records")
    return records


# ===========================
# Data Merging & Validation
# ===========================

def merge_and_validate(
    weather_records: List[Dict[str, Any]],
    traffic_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge weather and traffic data on (timestamp, location_key).
    
    Validation rules:
    - timestamp must be valid ISO8601 string
    - rainfall, temperature, wind_speed must be numeric or None
    - traffic_speed must be numeric or None
    
    Args:
        weather_records: List from Open-Meteo
        traffic_records: List from TomTom
    
    Returns:
        Merged list of validated records ready for DB insertion
    """
    # Build a lookup index for traffic data
    traffic_lookup = {
        (rec["timestamp"], rec["location_key"]): rec["traffic_speed_kmh"]
        for rec in traffic_records
    }
    
    merged = []
    
    for weather_rec in weather_records:
        key = (weather_rec["timestamp"], weather_rec["location_key"])
        traffic_speed = traffic_lookup.get(key)
        
        # Merge
        full_record = {
            "timestamp": weather_rec["timestamp"],
            "location_key": weather_rec["location_key"],
            "rainfall": weather_rec.get("rainfall"),
            "temperature_c": weather_rec.get("temperature_c"),
            "wind_speed_kmh": weather_rec.get("wind_speed_kmh"),
            "traffic_speed_kmh": traffic_speed,
            "feature_source": "api"
        }
        
        # Validate
        if _validate_record(full_record):
            merged.append(full_record)
        else:
            logger.warning(f"Skipping invalid record: {full_record}")
    
    logger.info(f"Merged and validated {len(merged)} records")
    return merged


def _validate_record(record: Dict[str, Any]) -> bool:
    """
    Validate a single feature record.
    
    Rules:
    - timestamp must be parseable as ISO8601
    - numeric fields must be float/int or None
    - at least one feature field must be non-None (otherwise it's useless)
    """
    try:
        # Validate timestamp
        datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        
        # Validate numeric fields
        numeric_fields = ["rainfall", "temperature_c", "wind_speed_kmh", "traffic_speed_kmh"]
        for field in numeric_fields:
            val = record.get(field)
            if val is not None and not isinstance(val, (int, float)):
                logger.warning(f"Field '{field}' is not numeric: {val}")
                return False
        
        # At least one feature must be present
        if all(record.get(f) is None for f in numeric_fields):
            logger.warning("Record has all-None features")
            return False
        
        return True
    
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Validation failed: {e}")
        return False


# ===========================
# Database Insertion
# ===========================

def insert_records(records: List[Dict[str, Any]]) -> int:
    """
    Insert records into SQLite database.
    
    Uses UPSERT to handle duplicates gracefully (composite unique constraint
    on timestamp + location_key) and fill in missing fields if updated data is retrieved.
    
    Args:
        records: List of validated feature dicts
    
    Returns:
        Number of successfully inserted rows (excludes duplicates)
    """
    if not records:
        logger.info("No records to insert")
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    inserted_count = 0
    
    for rec in records:
        try:
            cursor.execute("""
                INSERT INTO features (
                    timestamp, location_key, rainfall, temperature_c, 
                    wind_speed_kmh, traffic_speed_kmh, feature_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp, location_key) DO UPDATE SET
                    rainfall = COALESCE(excluded.rainfall, features.rainfall),
                    temperature_c = COALESCE(excluded.temperature_c, features.temperature_c),
                    wind_speed_kmh = COALESCE(excluded.wind_speed_kmh, features.wind_speed_kmh),
                    traffic_speed_kmh = COALESCE(excluded.traffic_speed_kmh, features.traffic_speed_kmh)
            """, (
                rec["timestamp"],
                rec["location_key"],
                rec.get("rainfall"),
                rec.get("temperature_c"),
                rec.get("wind_speed_kmh"),
                rec.get("traffic_speed_kmh"),
                rec.get("feature_source", "api")
            ))
            
            if cursor.rowcount > 0:
                inserted_count += 1
        
        except sqlite3.Error as e:
            logger.error(f"Failed to insert record {rec['timestamp']}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    logger.info(f"Inserted {inserted_count} new records (skipped {len(records) - inserted_count} duplicates)")
    return inserted_count


# ===========================
# Main Orchestration
# ===========================

def main() -> int:
    """
    Main execution flow:
    1. Initialize database
    2. Compute hours_lookback (dynamic from DB max timestamp, or from CLI arg)
    3. Fetch weather data (Open-Meteo)
    4. Fetch traffic data (TomTom)
    5. Merge and validate
    6. Insert into SQLite

    Returns:
        Exit code: 0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(description="Daily data ingestion for route-optimization pipeline.")
    parser.add_argument(
        "--hours-lookback",
        type=int,
        default=None,
        help=(
            "Number of past hours to fetch from Open-Meteo and TomTom. "
            "If omitted, computed automatically from the latest timestamp already "
            "in the database (falls back to 24h on an empty DB)."
        ),
    )
    args = parser.parse_args()

    try:
        logger.info("=== Starting Daily Data Ingestion ===")
        
        # Initialize DB (idempotent)
        init_database()

        # Determine lookback window
        if args.hours_lookback is not None:
            hours_lookback = args.hours_lookback
            logger.info(f"Using CLI-supplied hours_lookback={hours_lookback}")
        else:
            hours_lookback = get_dynamic_hours_lookback(default=24)
        
        # Create HTTP session with retry logic
        session = create_retry_session()
        
        # Fetch data from APIs
        weather_records = fetch_weather_data(session, hours_lookback)
        traffic_records = fetch_traffic_data(session, hours_lookback)
        
        # Merge and validate
        merged_records = merge_and_validate(weather_records, traffic_records)
        
        # Insert into database
        inserted_count = insert_records(merged_records)
        
        logger.info(f"=== Data Ingestion Complete: {inserted_count} new rows ===")
        return 0
    
    except Exception as e:
        logger.exception(f"Data ingestion failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
