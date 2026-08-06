"""
src/context/enricher.py
-----------------------
Phase 2 context enrichment pipeline.

Reads output/processed_data.csv (produced by Phase 1 ingestion),
appends weather and traffic context columns for every row, and writes
output/enriched_data.csv — ready for model training.

Since processed_data.csv contains road-level records without explicit
lat/lon coordinates, this module maps road_type → representative zone
using a configurable road_type → zone_code lookup. When lat/lon columns
are present in the CSV, those are used directly instead.

New columns appended
--------------------
  zone_name, zone_code, zone_type, zone_tags,
  corridor_name, corridor_code,
  weather_condition, temp_celsius, humidity_pct,
  wind_speed_mps, visibility_m, rain_1h_mm,
  rain_risk_score, is_raining, is_foggy, is_high_wind,
  is_ne_monsoon, rain_forecast_hours,
  current_speed_kmph, freeflow_speed_kmph,
  congestion_index, congestion_level,
  travel_time_ratio, traffic_source

Usage (CLI)
-----------
  python -m src.context.enricher
  python -m src.context.enricher --input output/processed_data.csv \\
                                  --output output/enriched_data.csv \\
                                  --config configs/context_config.yaml

Usage (Python)
--------------
  from src.context.enricher import run_enrichment, EnricherConfig
  result = run_enrichment()
  print(result.output_path, result.rows_enriched)
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.context.zone_registry import ZoneRegistry
from src.context.weather_fetcher import WeatherFetcher
from src.context.traffic_fetcher import TrafficFetcher
from src.context.context_aggregator import ContextAggregator

logger = logging.getLogger(__name__)

# ── Road-type → zone mapping ──────────────────────────────────────────────────
# Maps the road_type values in processed_data.csv to representative Chennai zones.
# These are used when the CSV has no lat/lon columns.
# Rationale: highway records → GST Road area (Chromepet), arterial → T. Nagar,
# residential → Adyar, service → Sholinganallur (IT corridor).

_ROAD_TYPE_TO_ZONE: dict[str, tuple[float, float]] = {
    "highway":     (12.9516, 80.1462),   # Chromepet / GST Road corridor
    "arterial":    (13.0418, 80.2341),   # T. Nagar
    "residential": (13.0067, 80.2572),   # Adyar
    "service":     (12.9010, 80.2279),   # Sholinganallur
    "collector":   (13.0850, 80.2101),   # Anna Nagar
    "trunk":       (13.1143, 80.1548),   # Ambattur / IRR
}
_FALLBACK_COORD = (13.0418, 80.2341)     # T. Nagar — city-centre fallback

# time_of_day string → approximate local hour (used by heuristic)
_TIME_OF_DAY_TO_HOUR: dict[str, int] = {
    "morning":   8,
    "afternoon": 14,
    "evening":   18,
    "night":     23,
}


# ── Config & result dataclasses ───────────────────────────────────────────────

@dataclass
class EnricherConfig:
    """All tunable parameters for the enrichment pipeline."""
    input_csv:      str  = "output/processed_data.csv"
    output_csv:     str  = "output/enriched_data.csv"
    zones_yaml:     str  = "configs/zones/chennai.yaml"
    context_config: str  = "configs/context_config.yaml"
    lat_column:     Optional[str] = None   # direct lat column if present
    lon_column:     Optional[str] = None   # direct lon column if present
    time_column:    str  = "time_of_day"
    weekend_column: str  = "is_weekend"
    verbose:        bool = True
    dry_run:        bool = True            # always dry_run unless API keys present


@dataclass
class EnrichmentResult:
    """Summary of what the enricher did."""
    output_path:    str
    rows_enriched:  int
    columns_added:  list[str] = field(default_factory=list)
    warnings:       list[str] = field(default_factory=list)


# ── Main enrichment function ──────────────────────────────────────────────────

def run_enrichment(config: Optional[EnricherConfig] = None) -> EnrichmentResult:
    """
    Reads processed_data.csv, appends context columns, writes enriched_data.csv.

    Parameters
    ----------
    config : EnricherConfig, optional
        If None, loads from configs/context_config.yaml + environment variables.

    Returns
    -------
    EnrichmentResult
        Summary of the operation.
    """
    if config is None:
        config = _build_config_from_yaml()

    _setup_logging(config.verbose)

    # ── 1. Load input CSV ─────────────────────────────────────────────────────
    input_path = Path(config.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)
    original_cols = list(df.columns)

    # ── 2. Wire up the context pipeline ──────────────────────────────────────
    ctx_cfg = _load_context_config(config.context_config)
    weather_cfg = ctx_cfg.get("weather", {})
    traffic_cfg = ctx_cfg.get("traffic", {})

    # Auto-detect dry_run: use real APIs only if keys are set
    dry_run = not (os.getenv("OWM_API_KEY") and os.getenv("TOMTOM_API_KEY"))
    if dry_run:
        logger.info(
            "No API keys detected — using dry_run mode "
            "(weather=mock, traffic=heuristic)"
        )

    registry = ZoneRegistry(config.zones_yaml)
    weather  = WeatherFetcher(config=weather_cfg, dry_run=dry_run)
    traffic  = TrafficFetcher(config=traffic_cfg, dry_run=dry_run)
    aggregator = ContextAggregator(
        registry, weather, traffic,
        corridor_max_distance_km=ctx_cfg.get("zone_registry", {}).get(
            "corridor_max_distance_km", 5.0
        ),
    )

    # ── 3. Build coordinate list for each row ─────────────────────────────────
    coords = _resolve_coordinates(df, config)

    # ── 4. Enrich each row ────────────────────────────────────────────────────
    logger.info("Enriching %d rows with context data...", len(df))
    context_rows = []

    for idx, (lat, lon) in enumerate(coords):
        # Get time context for heuristic (traffic fetcher uses it)
        hour       = _get_hour(df, idx, config.time_column)
        is_weekend = _get_weekend(df, idx, config.weekend_column)

        ctx = aggregator.get_context(lat, lon)

        # Patch traffic with time context if it came from heuristic
        if ctx.corridor is not None:
            traffic_snap = traffic.fetch(
                corridor_code=ctx.corridor.code,
                waypoints=ctx.corridor.waypoints,
                hour=hour,
                is_weekend=is_weekend,
            )
        else:
            traffic_snap = None

        feat = ctx.to_feature_dict()
        # Override traffic fields with time-aware version
        if traffic_snap is not None:
            feat.update(traffic_snap.to_feature_dict())

        context_rows.append(feat)

        if config.verbose and (idx + 1) % 50 == 0:
            logger.info("  ... processed %d / %d rows", idx + 1, len(df))

    # ── 5. Merge and write ────────────────────────────────────────────────────
    context_df = pd.DataFrame(context_rows)

    # Drop redundant coordinate columns already present in df
    drop_cols = {"input_lat", "input_lon"} & set(context_df.columns)
    context_df = context_df.drop(columns=list(drop_cols))

    enriched = pd.concat([df.reset_index(drop=True), context_df.reset_index(drop=True)], axis=1)

    # Identify which columns were actually added
    added_cols = [c for c in enriched.columns if c not in original_cols]

    output_path = Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)

    if config.verbose:
        logger.info("Wrote %d rows × %d columns to %s",
                    len(enriched), len(enriched.columns), output_path)
        logger.info("New columns added (%d): %s", len(added_cols), added_cols)

    return EnrichmentResult(
        output_path=str(output_path),
        rows_enriched=len(enriched),
        columns_added=added_cols,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_config_from_yaml(
    config_path: str = "configs/context_config.yaml",
) -> EnricherConfig:
    """Reads EnricherConfig values from context_config.yaml enricher section."""
    raw = _load_context_config(config_path)
    enr = raw.get("enricher", {})
    return EnricherConfig(
        input_csv=      enr.get("input_csv",     "output/processed_data.csv"),
        output_csv=     enr.get("output_csv",    "output/enriched_data.csv"),
        zones_yaml=     raw.get("zone_registry", {}).get(
                            "chennai_zones", "configs/zones/chennai.yaml"),
        context_config= config_path,
        lat_column=     enr.get("lat_column"),
        lon_column=     enr.get("lon_column"),
        time_column=    enr.get("time_column",   "time_of_day"),
        weekend_column= enr.get("weekend_column","is_weekend"),
        verbose=        bool(enr.get("verbose",  True)),
    )


def _load_context_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        logger.warning("context_config not found at %s — using defaults", p)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_coordinates(
    df: pd.DataFrame, config: EnricherConfig
) -> list[tuple[float, float]]:
    """
    Returns a (lat, lon) pair for every row.
    Priority: explicit lat/lon columns → road_type mapping → city fallback.
    """
    if (config.lat_column and config.lon_column
            and config.lat_column in df.columns
            and config.lon_column in df.columns):
        logger.info("Using explicit lat/lon columns: %s, %s",
                    config.lat_column, config.lon_column)
        return list(zip(
            df[config.lat_column].astype(float),
            df[config.lon_column].astype(float),
        ))

    if "road_type" in df.columns:
        logger.info("Mapping road_type → coordinates using built-in lookup table")
        def road_to_coord(rt: str) -> tuple[float, float]:
            return _ROAD_TYPE_TO_ZONE.get(str(rt).lower().strip(), _FALLBACK_COORD)
        return [road_to_coord(rt) for rt in df["road_type"]]

    logger.warning(
        "No lat/lon columns and no road_type column found — "
        "using city-centre fallback for all rows"
    )
    return [_FALLBACK_COORD] * len(df)


def _get_hour(df: pd.DataFrame, idx: int, time_col: str) -> Optional[int]:
    if time_col not in df.columns:
        return None
    val = str(df.at[idx, time_col]).lower().strip()
    return _TIME_OF_DAY_TO_HOUR.get(val)


def _get_weekend(df: pd.DataFrame, idx: int, weekend_col: str) -> Optional[bool]:
    if weekend_col not in df.columns:
        return None
    val = df.at[idx, weekend_col]
    if isinstance(val, bool):
        return val
    return str(val).lower().strip() in {"true", "1", "yes"}


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [enricher] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> EnricherConfig:
    parser = argparse.ArgumentParser(
        description="Enrich processed_data.csv with zone/weather/traffic context."
    )
    parser.add_argument("--input",   default="output/processed_data.csv",
                        help="Input CSV (default: output/processed_data.csv)")
    parser.add_argument("--output",  default="output/enriched_data.csv",
                        help="Output CSV (default: output/enriched_data.csv)")
    parser.add_argument("--config",  default="configs/context_config.yaml",
                        help="Context config YAML")
    parser.add_argument("--zones",   default="configs/zones/chennai.yaml",
                        help="Zone registry YAML")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress progress logging")
    args = parser.parse_args()

    base = _build_config_from_yaml(args.config)
    base.input_csv    = args.input
    base.output_csv   = args.output
    base.zones_yaml   = args.zones
    base.verbose      = not args.quiet
    return base


if __name__ == "__main__":
    cfg    = _parse_args()
    result = run_enrichment(cfg)
    print(f"\nEnrichment complete.")
    print(f"  Output : {result.output_path}")
    print(f"  Rows   : {result.rows_enriched}")
    print(f"  +Cols  : {result.columns_added}")
