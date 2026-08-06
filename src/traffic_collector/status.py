"""
src/traffic_collector/status.py
--------------------------------
Prints a human-readable summary of what has been collected so far.

Usage:
    python -m src.traffic_collector.status
    python -m src.traffic_collector.status --db data/traffic_history.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.traffic_collector.db import (
    DEFAULT_DB_PATH,
    get_connection,
    newest_sample,
    oldest_sample,
    recent_runs,
    row_count,
    rows_per_corridor,
)


def print_status(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Prints collection status to stdout."""

    if not db_path.exists():
        print(f"No database found at {db_path}.")
        print("Run the collector first:  python -m src.traffic_collector.collector --dry-run")
        return

    conn = get_connection(db_path)

    total      = row_count(conn)
    oldest     = oldest_sample(conn) or "—"
    newest     = newest_sample(conn) or "—"
    corridors  = rows_per_corridor(conn)
    runs       = recent_runs(conn, n=5)

    print("=" * 65)
    print("  JustSteer — Traffic Collection Status")
    print("=" * 65)
    print(f"  Database    : {db_path}")
    print(f"  Total rows  : {total:,}")
    print(f"  Oldest      : {oldest}")
    print(f"  Newest      : {newest}")

    # ── Per-corridor summary ──────────────────────────────────────────────────
    if corridors:
        print()
        print(f"  {'Corridor':<42} {'Rows':>6}  {'Avg Index':>9}  Last sample")
        print(f"  {'-'*42} {'-'*6}  {'-'*9}  {'-'*20}")
        for row in corridors:
            last = (row["last_sample"] or "")[:16]
            print(
                f"  {row['corridor_name']:<42} "
                f"{row['total_rows']:>6}  "
                f"{row['avg_congestion_index']:>9.3f}  "
                f"{last}"
            )
    else:
        print("\n  No corridor data yet.")

    # ── Recent runs ───────────────────────────────────────────────────────────
    if runs:
        print()
        print(f"  {'Run time (UTC)':<22} {'Total WP':>8}  {'OK':>4}  {'Fail':>4}  {'Sec':>6}")
        print(f"  {'-'*22} {'-'*8}  {'-'*4}  {'-'*4}  {'-'*6}")
        for r in runs:
            print(
                f"  {(r['run_at_utc'] or '')[:19]:<22} "
                f"{r['total_waypoints']:>8}  "
                f"{r['success_count']:>4}  "
                f"{r['failure_count']:>4}  "
                f"{r['duration_sec']:>6.2f}"
            )
    else:
        print("\n  No run log entries yet.")

    print("=" * 65)

    # ── Training data estimate ────────────────────────────────────────────────
    if total > 0:
        runs_done = len(runs)
        # Rough estimate: runs per day at 20-min interval = 72
        # Each run writes ~30 rows (10 corridors × 3 waypoints)
        days_of_data = total / (72 * 30)
        print(f"\n  Estimated data coverage : ~{days_of_data:.1f} days")
        if days_of_data < 7:
            print("  Status : Keep collecting — aim for at least 7 days")
        elif days_of_data < 30:
            print("  Status : Good for initial model training")
        else:
            print("  Status : Solid training dataset — ready for full model training")

    conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show traffic collection status."
    )
    parser.add_argument(
        "--db", default=str(DEFAULT_DB_PATH),
        help=f"SQLite DB path (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    print_status(Path(args.db))
