"""
Sub-step 1.4 — Validator.

Two jobs:
1. Quality gates on THIS dataset alone (enough rows, no column missing an
   unrecoverable fraction of its values).
2. Schema-consistency check against a PREVIOUSLY saved schema, if one
   exists — catches "the new scraped CSV silently renamed a column" or
   "gained/lost a column" before it reaches training.

Raises on hard failures; returns a list of warning strings for soft issues
so the caller decides whether to proceed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ingestion.config import IngestionConfig
from ingestion.profiler import ColumnProfile

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


def validate_dataset(
    df: pd.DataFrame, profiles: list[ColumnProfile], config: IngestionConfig
) -> list[str]:
    warnings: list[str] = []

    if len(df) < config.min_rows:
        raise ValidationError(
            f"Dataset has only {len(df)} rows; minimum required is {config.min_rows}."
        )

    for profile in profiles:
        if profile.missing_pct >= config.max_missing_threshold_hard_fail:
            raise ValidationError(
                f"Column '{profile.name}' is {profile.missing_pct:.0%} missing "
                f"(hard-fail threshold {config.max_missing_threshold_hard_fail:.0%}). "
                "Imputing this much data would manufacture values, not clean them."
            )
        if profile.missing_pct >= config.missing_warn_threshold:
            warnings.append(
                f"Column '{profile.name}' is {profile.missing_pct:.0%} missing — "
                "review whether imputation is appropriate here."
            )

    schema_warnings = _check_against_previous_schema(profiles, config)
    warnings.extend(schema_warnings)

    for w in warnings:
        logger.warning(w)

    return warnings


def _check_against_previous_schema(
    profiles: list[ColumnProfile], config: IngestionConfig
) -> list[str]:
    schema_path = Path(config.schema_history_path)
    if not schema_path.exists():
        return []  # first-ever run — nothing to compare against

    try:
        with open(schema_path, "r") as f:
            previous = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read previous schema at %s: %s", schema_path, exc)
        return []

    previous_columns = {c["name"] for c in previous.get("columns", [])}
    current_columns = {p.name for p in profiles}

    warnings = []
    missing_now = previous_columns - current_columns
    new_now = current_columns - previous_columns

    if missing_now:
        warnings.append(f"Columns present before but missing now: {sorted(missing_now)}")
    if new_now:
        warnings.append(f"New columns not seen in the previous schema: {sorted(new_now)}")

    previous_types = {c["name"]: c["inferred_type"] for c in previous.get("columns", [])}
    for p in profiles:
        prev_type = previous_types.get(p.name)
        if prev_type and prev_type != p.inferred_type:
            warnings.append(
                f"Column '{p.name}' changed inferred type: {prev_type} -> {p.inferred_type}"
            )

    return warnings
