"""
src/traffic_collector/collector.py
------------------------------------
Main traffic data collection script.

Called by Windows Task Scheduler every 20 minutes.
Each run:
  1. Loads all corridors from configs/zones/chennai.yaml
  2. Samples up to MAX_WAYPOINTS_PER_CORRIDOR waypoints per corridor
  3. Calls TomTom Flow API for each waypoint
  4. Writes results to data/traffic_history.db (SQLite)
  5. Logs run summary (success / failure counts, duration)

API usage per run:
  10 corridors × 3 waypoints = 30 calls
  Every 20 min → 72 runs/day × 30 = 2,160 calls/day
  TomTom free tier = 2,500 calls/day  ✓

Usage:
  python -m src.traffic_collector.collector          # live run
  python -m src.traffic_collector.collector --dry-run  # no API calls, no DB write
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from src.context.zone_registry import ZoneRegistry
from src.traffic_collector.db import (
    DEFAULT_DB_PATH,
    get_connection,
    index_to_level,
    init_schema,
    insert_collection_log,
    insert_snapshot,
)

# ── Configuration ─────────────────────────────────────────────────────────────
ZONES_YAML              = Path("configs/zones/chennai.yaml")
LOG_FILE                = Path("logs/traffic_collector.log")
MAX_WAYPOINTS_PER_CORR  = 3        # max waypoints sampled per corridor per run
REQUEST_TIMEOUT_SEC     = 5
IST_OFFSET              = timedelta(hours=5, minutes=30)

TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
)

# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = True) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [collector] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)


# ── Time helpers ──────────────────────────────────────────────────────────────

def _ist_fields(dt_utc: datetime) -> tuple[str, int, int, int, int]:
    """
    Returns (ist_iso, hour_ist, day_of_week, is_weekend, month)
    from a UTC datetime.
    """
    dt_ist   = dt_utc + IST_OFFSET
    ist_iso  = dt_ist.strftime("%Y-%m-%dT%H:%M:%S+05:30")
    hour     = dt_ist.hour
    dow      = dt_ist.weekday()          # 0=Mon … 6=Sun
    is_wknd  = 1 if dow >= 5 else 0
    month    = dt_ist.month
    return ist_iso, hour, dow, is_wknd, month


# ── Waypoint sampling ─────────────────────────────────────────────────────────

def _sample_waypoints(
    waypoints: list[tuple[float, float]], max_n: int
) -> list[tuple[int, tuple[float, float]]]:
    """
    Returns up to max_n (index, waypoint) pairs, evenly spaced along
    the corridor. Always includes the first and last waypoint.
    """
    n = len(waypoints)
    if n <= max_n:
        return list(enumerate(waypoints))
    step    = (n - 1) / (max_n - 1)
    indices = sorted({round(i * step) for i in range(max_n)})
    return [(i, waypoints[i]) for i in indices]


# ── TomTom API call ───────────────────────────────────────────────────────────

def _fetch_tomtom(
    lat: float, lon: float, api_key: str
) -> Optional[dict]:
    """
    Calls TomTom Flow API for a single (lat, lon) point.
    Returns a dict with current_speed, freeflow_speed, confidence,
    or None on any error.
    """
    try:
        resp = requests.get(
            TOMTOM_FLOW_URL,
            params={"point": f"{lat},{lon}", "key": api_key, "unit": "KMPH"},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        seg = resp.json().get("flowSegmentData", {})
        current  = float(seg.get("currentSpeed",  0))
        freeflow = float(seg.get("freeFlowSpeed",  1))
        conf     = float(seg.get("confidence",     0))
        index    = round(current / freeflow, 3) if freeflow > 0 else 0.0
        index    = max(0.0, min(1.0, index))
        return {
            "current_speed":  current,
            "freeflow_speed": freeflow,
            "congestion_index": index,
            "congestion_level": index_to_level(index),
            "confidence": conf,
        }
    except requests.exceptions.Timeout:
        logger.warning("TomTom timeout at (%.4f, %.4f)", lat, lon)
    except requests.exceptions.HTTPError as exc:
        logger.warning("TomTom HTTP error at (%.4f, %.4f) — %s", lat, lon, exc)
    except requests.exceptions.RequestException as exc:
        logger.warning("TomTom request failed — %s", exc)
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        logger.warning("TomTom parse error — %s", exc)
    return None


# ── Dry-run mock ──────────────────────────────────────────────────────────────

def _mock_fetch(lat: float, lon: float) -> dict:
    """Returns plausible synthetic data for dry-run mode."""
    now_ist_hour = (datetime.now(timezone.utc) + IST_OFFSET).hour
    # Simple peak-hour simulation
    if 8 <= now_ist_hour < 10 or 17 <= now_ist_hour < 20:
        current, freeflow = 22.0, 45.0
    elif 22 <= now_ist_hour or now_ist_hour < 6:
        current, freeflow = 55.0, 60.0
    else:
        current, freeflow = 38.0, 50.0
    index = round(current / freeflow, 3)
    return {
        "current_speed":    current,
        "freeflow_speed":   freeflow,
        "congestion_index": index,
        "congestion_level": index_to_level(index),
        "confidence":       1.0,
    }


# ── Main collection logic ─────────────────────────────────────────────────────

def run_collection(
    db_path: Path = DEFAULT_DB_PATH,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Runs one full collection cycle across all corridors.

    Parameters
    ----------
    db_path  : Path to the SQLite database file.
    dry_run  : If True, uses mock data instead of real API calls.
               No rows are written to the database.
    verbose  : Whether to log each waypoint result.

    Returns
    -------
    dict with keys: total, success, failure, duration_sec
    """
    _setup_logging(verbose)

    # Load .env so API key is available when running via Task Scheduler
    load_dotenv(override=False)
    api_key = os.getenv("TOMTOM_API_KEY", "")

    if not api_key and not dry_run:
        logger.error(
            "TOMTOM_API_KEY is not set and dry_run=False. "
            "Set the key in .env or run with --dry-run."
        )
        sys.exit(1)

    if dry_run:
        logger.info("=== DRY-RUN mode — no API calls, no DB writes ===")
    else:
        logger.info("=== Live collection run starting ===")

    # Load zone registry
    if not ZONES_YAML.exists():
        logger.error("Zone config not found: %s", ZONES_YAML)
        sys.exit(1)

    registry = ZoneRegistry(ZONES_YAML)
    logger.info("Loaded %s", registry.summary())

    # Time context
    now_utc  = datetime.now(timezone.utc)
    utc_iso  = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    ist_iso, hour_ist, dow, is_wknd, month = _ist_fields(now_utc)

    logger.info(
        "Run time: %s IST | hour=%d | %s | month=%d",
        ist_iso, hour_ist,
        "weekend" if is_wknd else "weekday",
        month,
    )

    # Open DB (skip in dry-run)
    conn = None
    if not dry_run:
        conn = get_connection(db_path)
        init_schema(conn)

    # ── Collect ───────────────────────────────────────────────────────────────
    t_start       = time.monotonic()
    total_wps     = 0
    success_count = 0
    failure_count = 0

    for corridor in registry.corridors:
        sampled = _sample_waypoints(corridor.waypoints, MAX_WAYPOINTS_PER_CORR)
        logger.info(
            "Corridor: %-40s (%d waypoints sampled)",
            corridor.name, len(sampled),
        )

        for wp_idx, (lat, lon) in sampled:
            total_wps += 1

            if dry_run:
                result = _mock_fetch(lat, lon)
                tag = "[DRY-RUN]"
            else:
                result = _fetch_tomtom(lat, lon, api_key)
                tag = "[LIVE]"

            if result is None:
                failure_count += 1
                logger.warning(
                    "  %s FAILED  wp[%d] (%.4f, %.4f)",
                    tag, wp_idx, lat, lon,
                )
                continue

            success_count += 1
            logger.debug(
                "  %s wp[%d] (%.4f,%.4f) "
                "speed=%.1f/%.1f kmph  index=%.3f  %s",
                tag, wp_idx, lat, lon,
                result["current_speed"], result["freeflow_speed"],
                result["congestion_index"], result["congestion_level"],
            )

            if conn is not None:
                insert_snapshot(
                    conn=conn,
                    corridor_code=corridor.code,
                    corridor_name=corridor.name,
                    waypoint_index=wp_idx,
                    waypoint_lat=lat,
                    waypoint_lon=lon,
                    sampled_at_utc=utc_iso,
                    sampled_at_ist=ist_iso,
                    hour_ist=hour_ist,
                    day_of_week=dow,
                    is_weekend=is_wknd,
                    month=month,
                    current_speed_kmph=result["current_speed"],
                    freeflow_speed_kmph=result["freeflow_speed"],
                    congestion_index=result["congestion_index"],
                    congestion_level=result["congestion_level"],
                    confidence=result["confidence"],
                )

    # Commit all inserts in one batch
    if conn is not None:
        conn.commit()

    duration = round(time.monotonic() - t_start, 2)

    # Log the run summary to DB
    if conn is not None:
        insert_collection_log(
            conn=conn,
            run_at_utc=utc_iso,
            total_waypoints=total_wps,
            success_count=success_count,
            failure_count=failure_count,
            duration_sec=duration,
            api_key_present=1 if api_key else 0,
        )
        conn.close()

    logger.info(
        "=== Run complete — %d/%d waypoints OK in %.2fs ===",
        success_count, total_wps, duration,
    )

    return {
        "total":        total_wps,
        "success":      success_count,
        "failure":      failure_count,
        "duration_sec": duration,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TomTom traffic data for Chennai corridors."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use mock data. No API calls made, nothing written to DB.",
    )
    parser.add_argument(
        "--db", default=str(DEFAULT_DB_PATH),
        help=f"SQLite DB path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-waypoint log lines.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    result = run_collection(
        db_path=Path(args.db),
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    print(
        f"\nDone. {result['success']}/{result['total']} waypoints collected "
        f"in {result['duration_sec']}s."
    )
