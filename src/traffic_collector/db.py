"""
src/traffic_collector/db.py
----------------------------
SQLite database helpers for the traffic history collector.

Schema
------
Table: traffic_snapshots
  Each row is one TomTom flow reading for a single waypoint on a corridor
  at a specific moment in time.

  id                  — autoincrement primary key
  corridor_code       — e.g. "CHN_GST"
  corridor_name       — e.g. "GST Road"
  waypoint_index      — position in the corridor's waypoint list (0-based)
  waypoint_lat        — latitude of the sampled waypoint
  waypoint_lon        — longitude of the sampled waypoint
  sampled_at_utc      — ISO 8601 UTC timestamp of the reading
  sampled_at_ist      — ISO 8601 IST timestamp (UTC+5:30) for convenience
  hour_ist            — local hour 0-23 (for peak-hour analysis)
  day_of_week         — 0=Monday … 6=Sunday
  is_weekend          — 1 if Saturday or Sunday, else 0
  month               — 1-12 (useful for monsoon season analysis)
  current_speed_kmph  — TomTom currentSpeed
  freeflow_speed_kmph — TomTom freeFlowSpeed
  congestion_index    — current / freeflow, clamped [0, 1]
  congestion_level    — free_flow | moderate | heavy | severe
  confidence          — TomTom confidence score 0-1

Table: collection_log
  One row per collection run — tracks success/failure counts for monitoring.

  id              — autoincrement primary key
  run_at_utc      — ISO 8601 UTC timestamp
  total_waypoints — how many waypoints were attempted
  success_count   — how many returned valid data
  failure_count   — how many failed (timeout / HTTP error)
  duration_sec    — wall-clock time for the full run
  api_key_present — 1 if key was set, 0 if dry-run
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# ── Default DB path ───────────────────────────────────────────────────────────
DEFAULT_DB_PATH = Path("data/traffic_history.db")

# ── Congestion level thresholds ───────────────────────────────────────────────
_THRESH_FREE  = 0.85
_THRESH_MOD   = 0.60
_THRESH_HEAVY = 0.40


# ── Connection helper ─────────────────────────────────────────────────────────

def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Opens (and creates if needed) the SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row      # rows accessible as dicts
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent readers
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_schema(conn: sqlite3.Connection) -> None:
    """Creates tables and indexes if they don't already exist.
    Also migrates existing traffic_snapshots tables to add the
    UNIQUE(corridor_code, waypoint_index, sampled_at_utc) constraint
    if it is missing (one-time migration for pre-existing DBs).
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS traffic_snapshots (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            corridor_code       TEXT    NOT NULL,
            corridor_name       TEXT    NOT NULL,
            waypoint_index      INTEGER NOT NULL,
            waypoint_lat        REAL    NOT NULL,
            waypoint_lon        REAL    NOT NULL,
            sampled_at_utc      TEXT    NOT NULL,
            sampled_at_ist      TEXT    NOT NULL,
            hour_ist            INTEGER NOT NULL,
            day_of_week         INTEGER NOT NULL,
            is_weekend          INTEGER NOT NULL,
            month               INTEGER NOT NULL,
            current_speed_kmph  REAL,
            freeflow_speed_kmph REAL,
            congestion_index    REAL,
            congestion_level    TEXT,
            confidence          REAL,
            UNIQUE (corridor_code, waypoint_index, sampled_at_utc)
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_corridor_time
            ON traffic_snapshots (corridor_code, sampled_at_utc);

        CREATE INDEX IF NOT EXISTS idx_snapshots_hour
            ON traffic_snapshots (hour_ist, day_of_week);

        CREATE TABLE IF NOT EXISTS collection_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at_utc      TEXT    NOT NULL,
            total_waypoints INTEGER NOT NULL,
            success_count   INTEGER NOT NULL,
            failure_count   INTEGER NOT NULL,
            duration_sec    REAL    NOT NULL,
            api_key_present INTEGER NOT NULL
        );
    """)
    conn.commit()

    # One-time migration: add UNIQUE constraint to any pre-existing DB
    # that was created before this constraint was introduced.
    _migrate_unique_constraint(conn)


def _migrate_unique_constraint(conn: sqlite3.Connection) -> None:
    """
    Rebuilds traffic_snapshots with the UNIQUE constraint if it is absent.
    Deduplicates rows in the process (keeps the row with the lowest id).
    Safe to call on a DB that is already up-to-date (no-op in that case).
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='traffic_snapshots'"
    ).fetchone()
    if row and "UNIQUE" in (row[0] or "").upper():
        return  # Already constrained — nothing to do

    conn.executescript("""
        CREATE TABLE traffic_snapshots_new (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            corridor_code       TEXT    NOT NULL,
            corridor_name       TEXT    NOT NULL,
            waypoint_index      INTEGER NOT NULL,
            waypoint_lat        REAL    NOT NULL,
            waypoint_lon        REAL    NOT NULL,
            sampled_at_utc      TEXT    NOT NULL,
            sampled_at_ist      TEXT    NOT NULL,
            hour_ist            INTEGER NOT NULL,
            day_of_week         INTEGER NOT NULL,
            is_weekend          INTEGER NOT NULL,
            month               INTEGER NOT NULL,
            current_speed_kmph  REAL,
            freeflow_speed_kmph REAL,
            congestion_index    REAL,
            congestion_level    TEXT,
            confidence          REAL,
            UNIQUE (corridor_code, waypoint_index, sampled_at_utc)
        );

        INSERT OR IGNORE INTO traffic_snapshots_new (
            id, corridor_code, corridor_name, waypoint_index,
            waypoint_lat, waypoint_lon,
            sampled_at_utc, sampled_at_ist,
            hour_ist, day_of_week, is_weekend, month,
            current_speed_kmph, freeflow_speed_kmph,
            congestion_index, congestion_level, confidence
        )
        SELECT
            id, corridor_code, corridor_name, waypoint_index,
            waypoint_lat, waypoint_lon,
            sampled_at_utc, sampled_at_ist,
            hour_ist, day_of_week, is_weekend, month,
            current_speed_kmph, freeflow_speed_kmph,
            congestion_index, congestion_level, confidence
        FROM traffic_snapshots
        ORDER BY id;

        DROP TABLE traffic_snapshots;
        ALTER TABLE traffic_snapshots_new RENAME TO traffic_snapshots;

        CREATE INDEX IF NOT EXISTS idx_snapshots_corridor_time
            ON traffic_snapshots (corridor_code, sampled_at_utc);

        CREATE INDEX IF NOT EXISTS idx_snapshots_hour
            ON traffic_snapshots (hour_ist, day_of_week);
    """)
    conn.commit()


# ── Insert helpers ─────────────────────────────────────────────────────────────

def insert_snapshot(
    conn: sqlite3.Connection,
    corridor_code: str,
    corridor_name: str,
    waypoint_index: int,
    waypoint_lat: float,
    waypoint_lon: float,
    sampled_at_utc: str,
    sampled_at_ist: str,
    hour_ist: int,
    day_of_week: int,
    is_weekend: int,
    month: int,
    current_speed_kmph: float,
    freeflow_speed_kmph: float,
    congestion_index: float,
    congestion_level: str,
    confidence: float,
) -> None:
    conn.execute("""
        INSERT OR IGNORE INTO traffic_snapshots (
            corridor_code, corridor_name, waypoint_index,
            waypoint_lat, waypoint_lon,
            sampled_at_utc, sampled_at_ist,
            hour_ist, day_of_week, is_weekend, month,
            current_speed_kmph, freeflow_speed_kmph,
            congestion_index, congestion_level, confidence
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        corridor_code, corridor_name, waypoint_index,
        waypoint_lat, waypoint_lon,
        sampled_at_utc, sampled_at_ist,
        hour_ist, day_of_week, is_weekend, month,
        current_speed_kmph, freeflow_speed_kmph,
        congestion_index, congestion_level, confidence,
    ))


def insert_collection_log(
    conn: sqlite3.Connection,
    run_at_utc: str,
    total_waypoints: int,
    success_count: int,
    failure_count: int,
    duration_sec: float,
    api_key_present: int,
) -> None:
    conn.execute("""
        INSERT INTO collection_log (
            run_at_utc, total_waypoints,
            success_count, failure_count,
            duration_sec, api_key_present
        ) VALUES (?,?,?,?,?,?)
    """, (run_at_utc, total_waypoints, success_count,
          failure_count, duration_sec, api_key_present))
    conn.commit()


# ── Utility: compute congestion level from index ──────────────────────────────

def index_to_level(index: float) -> str:
    if index >= _THRESH_FREE:
        return "free_flow"
    elif index >= _THRESH_MOD:
        return "moderate"
    elif index >= _THRESH_HEAVY:
        return "heavy"
    else:
        return "severe"


# ── Query helpers (used by status.py) ────────────────────────────────────────

def row_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM traffic_snapshots"
    ).fetchone()[0]


def oldest_sample(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT MIN(sampled_at_utc) FROM traffic_snapshots"
    ).fetchone()
    return row[0] if row else None


def newest_sample(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(sampled_at_utc) FROM traffic_snapshots"
    ).fetchone()
    return row[0] if row else None


def rows_per_corridor(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT corridor_code, corridor_name,
               COUNT(*)              AS total_rows,
               MIN(sampled_at_utc)   AS first_sample,
               MAX(sampled_at_utc)   AS last_sample,
               ROUND(AVG(congestion_index), 3) AS avg_congestion_index
        FROM   traffic_snapshots
        GROUP  BY corridor_code
        ORDER  BY corridor_code
    """).fetchall()


def recent_runs(conn: sqlite3.Connection, n: int = 10) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT run_at_utc, total_waypoints,
               success_count, failure_count,
               ROUND(duration_sec, 2) AS duration_sec
        FROM   collection_log
        ORDER  BY id DESC
        LIMIT  ?
    """, (n,)).fetchall()
