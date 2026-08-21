"""
scripts/merge_artifact_db.py
─────────────────────────────
Merges a downloaded GitHub Actions artifact DB into the local SQLite DBs
without losing any locally-collected rows.

Usage
─────
  # After downloading an artifact ZIP from GitHub Actions and extracting it:
  python scripts/merge_artifact_db.py --artifact-dir ./db-sync

  # Or point at specific remote files:
  python scripts/merge_artifact_db.py \\
      --traffic-remote  ./db-sync/data/traffic_history.db \\
      --features-remote ./db-sync/data/db/daily_features.db

  # Dry-run — shows what would be merged without writing anything:
  python scripts/merge_artifact_db.py --artifact-dir ./db-sync --dry-run

What it does
────────────
  traffic_history.db
    • Runs the UNIQUE-constraint migration on the local DB first (safe no-op
      if already migrated).
    • INSERT OR IGNORE into traffic_snapshots  → dedupes on
      (corridor_code, waypoint_index, sampled_at_utc)
    • INSERT OR IGNORE into collection_log     → dedupes on run_at_utc

  daily_features.db
    • INSERT OR IGNORE into features           → dedupes on
      (timestamp, location_key)

Both merges keep all existing local rows; only genuinely new rows from the
remote file are added.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# ── Repo-root relative paths (defaults) ──────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

LOCAL_TRAFFIC_DB  = REPO_ROOT / "data" / "traffic_history.db"
LOCAL_FEATURES_DB = REPO_ROOT / "data" / "db" / "daily_features.db"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608


def _has_unique_constraint(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='traffic_snapshots'"
    ).fetchone()
    return bool(row and "UNIQUE" in (row[0] or "").upper())


def _migrate_traffic_constraint(conn: sqlite3.Connection) -> None:
    """
    One-time rebuild of traffic_snapshots to add the UNIQUE constraint.
    No-op if the constraint is already present.
    """
    if _has_unique_constraint(conn):
        return

    print("  [migrate] Adding UNIQUE constraint to local traffic_snapshots …")
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
    print("  [migrate] Done.")


# ── Merge functions ───────────────────────────────────────────────────────────

def merge_traffic(local_path: Path, remote_path: Path, dry_run: bool) -> None:
    print(f"\n-- traffic_history.db ------------------------------------------")
    print(f"  local  : {local_path}")
    print(f"  remote : {remote_path}")

    if not remote_path.exists():
        print("  SKIP: remote file not found.")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    local_conn = sqlite3.connect(str(local_path))
    local_conn.execute("PRAGMA journal_mode=WAL")

    # Ensure local schema exists and is migrated
    _migrate_traffic_constraint(local_conn)

    before_snap = _row_count(local_conn, "traffic_snapshots")
    before_log  = _row_count(local_conn, "collection_log")

    if dry_run:
        remote_conn = sqlite3.connect(str(remote_path))
        remote_snap = _row_count(remote_conn, "traffic_snapshots")
        remote_log  = _row_count(remote_conn, "collection_log")
        remote_conn.close()
        print(f"  [dry-run] local  snapshots={before_snap:,}  log_runs={before_log:,}")
        print(f"  [dry-run] remote snapshots={remote_snap:,}  log_runs={remote_log:,}")
        print(f"  [dry-run] would attempt to merge (duplicates silently ignored).")
        local_conn.close()
        return

    local_conn.execute(f"ATTACH DATABASE ? AS remote", (str(remote_path),))

    local_conn.execute("""
        INSERT OR IGNORE INTO traffic_snapshots (
            corridor_code, corridor_name, waypoint_index,
            waypoint_lat, waypoint_lon,
            sampled_at_utc, sampled_at_ist,
            hour_ist, day_of_week, is_weekend, month,
            current_speed_kmph, freeflow_speed_kmph,
            congestion_index, congestion_level, confidence
        )
        SELECT
            corridor_code, corridor_name, waypoint_index,
            waypoint_lat, waypoint_lon,
            sampled_at_utc, sampled_at_ist,
            hour_ist, day_of_week, is_weekend, month,
            current_speed_kmph, freeflow_speed_kmph,
            congestion_index, congestion_level, confidence
        FROM remote.traffic_snapshots
    """)

    local_conn.execute("""
        INSERT OR IGNORE INTO collection_log (
            run_at_utc, total_waypoints,
            success_count, failure_count,
            duration_sec, api_key_present
        )
        SELECT
            run_at_utc, total_waypoints,
            success_count, failure_count,
            duration_sec, api_key_present
        FROM remote.collection_log
    """)

    local_conn.commit()
    local_conn.execute("DETACH DATABASE remote")

    after_snap = _row_count(local_conn, "traffic_snapshots")
    after_log  = _row_count(local_conn, "collection_log")
    local_conn.close()

    print(f"  traffic_snapshots : {before_snap:,} → {after_snap:,}  (+{after_snap - before_snap:,} new)")
    print(f"  collection_log    : {before_log:,}  → {after_log:,}   (+{after_log - before_log:,} new)")


def merge_features(local_path: Path, remote_path: Path, dry_run: bool) -> None:
    print(f"\n-- daily_features.db --------------------------------------------")
    print(f"  local  : {local_path}")
    print(f"  remote : {remote_path}")

    if not remote_path.exists():
        print("  SKIP: remote file not found.")
        return

    local_conn = sqlite3.connect(str(local_path))
    local_conn.execute("PRAGMA journal_mode=WAL")

    # Ensure local schema exists (idempotent)
    local_conn.execute("""
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
    local_conn.commit()

    before = _row_count(local_conn, "features")

    if dry_run:
        remote_conn = sqlite3.connect(str(remote_path))
        remote_rows = _row_count(remote_conn, "features")
        remote_conn.close()
        print(f"  [dry-run] local={before:,}  remote={remote_rows:,}")
        print(f"  [dry-run] would attempt to merge (duplicates silently ignored).")
        local_conn.close()
        return

    local_conn.execute(f"ATTACH DATABASE ? AS remote", (str(remote_path),))

    local_conn.execute("""
        INSERT OR IGNORE INTO features (
            timestamp, location_key, rainfall,
            temperature_c, wind_speed_kmh, traffic_speed_kmh,
            feature_source, created_at
        )
        SELECT
            timestamp, location_key, rainfall,
            temperature_c, wind_speed_kmh, traffic_speed_kmh,
            feature_source, created_at
        FROM remote.features
    """)

    local_conn.commit()
    local_conn.execute("DETACH DATABASE remote")

    after = _row_count(local_conn, "features")
    local_conn.close()

    print(f"  features : {before:,} → {after:,}  (+{after - before:,} new)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a downloaded GitHub Actions artifact DB into the local SQLite DBs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After extracting db-snapshots-<run>.zip into ./db-sync:
  python scripts/merge_artifact_db.py --artifact-dir ./db-sync

  # Preview without writing:
  python scripts/merge_artifact_db.py --artifact-dir ./db-sync --dry-run

  # Point at files explicitly:
  python scripts/merge_artifact_db.py \\
      --traffic-remote  ./db-sync/data/traffic_history.db \\
      --features-remote ./db-sync/data/db/daily_features.db
""",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=None,
        help="Directory where the artifact ZIP was extracted. "
             "Expects the same layout as the repo: "
             "data/db/daily_features.db inside, data/traffic_history.db inside.",
    )
    parser.add_argument(
        "--traffic-remote", type=Path, default=None,
        help="Explicit path to the remote traffic_history.db.",
    )
    parser.add_argument(
        "--features-remote", type=Path, default=None,
        help="Explicit path to the remote daily_features.db.",
    )
    parser.add_argument(
        "--traffic-local", type=Path, default=LOCAL_TRAFFIC_DB,
        help=f"Local traffic DB to merge into (default: {LOCAL_TRAFFIC_DB})",
    )
    parser.add_argument(
        "--features-local", type=Path, default=LOCAL_FEATURES_DB,
        help=f"Local features DB to merge into (default: {LOCAL_FEATURES_DB})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be merged without writing anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Resolve remote paths
    if args.artifact_dir:
        traffic_remote  = args.artifact_dir / "data" / "traffic_history.db"
        features_remote = args.artifact_dir / "data" / "db" / "daily_features.db"
    else:
        traffic_remote  = args.traffic_remote
        features_remote = args.features_remote

    if traffic_remote is None and features_remote is None:
        print("ERROR: provide --artifact-dir or at least one of --traffic-remote / --features-remote")
        return 1

    if args.dry_run:
        print("=== DRY RUN — no changes will be written ===")

    if traffic_remote:
        merge_traffic(args.traffic_local, traffic_remote, args.dry_run)

    if features_remote:
        merge_features(args.features_local, features_remote, args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
