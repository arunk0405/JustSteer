"""
src/traffic_collector/__init__.py
----------------------------------
Traffic history collector package.

Collects real-time TomTom traffic flow data for every corridor
defined in configs/zones/chennai.yaml and stores it in a local
SQLite database (data/traffic_history.db).

Designed to run on a schedule (every 20 minutes via Windows Task
Scheduler) to build up historical traffic data for model training.

Usage:
    # Run a single collection cycle
    python -m src.traffic_collector.collector

    # Check collection status / summary
    python -m src.traffic_collector.status

    # Dry-run (no API calls, no DB writes — just prints what would happen)
    python -m src.traffic_collector.collector --dry-run
"""


def run_collection(*args, **kwargs):
    from src.traffic_collector.collector import run_collection as _run
    return _run(*args, **kwargs)


def print_status(*args, **kwargs):
    from src.traffic_collector.status import print_status as _ps
    return _ps(*args, **kwargs)


__all__ = ["run_collection", "print_status"]
