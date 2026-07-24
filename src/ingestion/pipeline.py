"""
Sub-step 1.5 — Orchestrator.

The single entry point everything else (a CLI, a FastAPI upload endpoint,
a notebook) should call. It chains the four independent stages and returns
both the cleaned DataFrame and its full schema metadata in memory, in
addition to writing them to disk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.cleaner import clean_dataset
from ingestion.config import IngestionConfig
from ingestion.exporter import export_results
from ingestion.loaders import load_dataset
from ingestion.profiler import ColumnProfile, profile_dataset
from ingestion.validator import validate_dataset

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    dataframe: pd.DataFrame
    profiles: list[ColumnProfile]
    warnings: list[str]
    output_paths: dict


def run_ingestion(
    input_path: str | Path, config: IngestionConfig | None = None
) -> IngestionResult:
    config = config or IngestionConfig()

    logger.info("=== Ingestion pipeline starting: %s ===", input_path)

    raw_df = load_dataset(input_path)

    profiles = profile_dataset(raw_df, config)
    logger.info(
        "Profiled %d columns: %s",
        len(profiles),
        {p.name: p.inferred_type for p in profiles},
    )

    clean_df = clean_dataset(raw_df, profiles, config)

    # Re-profile after cleaning: missing counts and dtypes may have changed
    # (e.g. a datetime column parsed successfully), and downstream consumers
    # should see metadata that matches the data they're actually getting.
    final_profiles = profile_dataset(clean_df, config)

    warnings = validate_dataset(clean_df, final_profiles, config)

    output_paths = export_results(
        clean_df, final_profiles, warnings, config.output_dir, str(input_path)
    )

    logger.info("=== Ingestion pipeline complete. Outputs: %s ===", output_paths)

    return IngestionResult(
        dataframe=clean_df,
        profiles=final_profiles,
        warnings=warnings,
        output_paths=output_paths,
    )
