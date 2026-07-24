"""
Sub-step 1.3 — Cleaner.

Applies type-appropriate cleaning based on the profile produced in 1.2.
Deliberately kept separate from profiling: you always get to see what the
data looked like *before* anything here touches it, by inspecting the
ColumnProfile list first.

Nothing here silently drops columns based on "this looks messy" — only
based on explicit config-driven rules (suggested_role == "drop", or the
hard-fail missing threshold), so behavior is predictable and defensible in
a report.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ingestion.config import IngestionConfig
from ingestion.profiler import ColumnProfile

logger = logging.getLogger(__name__)


def clean_dataset(
    df: pd.DataFrame, profiles: list[ColumnProfile], config: IngestionConfig
) -> pd.DataFrame:
    df = df.copy()

    if config.drop_duplicate_rows:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            logger.info("Dropped %d exact duplicate rows", removed)

    profile_by_name = {p.name: p for p in profiles}

    drop_cols = [p.name for p in profiles if p.suggested_role == "drop"]
    if drop_cols:
        logger.info("Dropping columns flagged as 'drop': %s", drop_cols)
        df = df.drop(columns=drop_cols)

    for col in df.columns:
        profile = profile_by_name.get(col)
        if profile is None or profile.suggested_role == "identifier":
            continue  # never impute/alter identifier columns

        if profile.inferred_type == "numeric":
            df[col] = _clean_numeric_column(df[col], config)
        elif profile.inferred_type in ("categorical", "boolean"):
            df[col] = _clean_categorical_column(df[col], config)
        elif profile.inferred_type == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()

    return df


def _clean_numeric_column(series: pd.Series, config: IngestionConfig) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")

    if config.numeric_impute_strategy == "median":
        fill_value = series.median()
    elif config.numeric_impute_strategy == "mean":
        fill_value = series.mean()
    elif config.numeric_impute_strategy == "drop_rows":
        return series  # rows with NaN are dropped centrally, see clean_dataset caller if needed
    else:
        raise ValueError(f"Unknown numeric_impute_strategy: {config.numeric_impute_strategy}")

    series = series.fillna(fill_value)

    if config.outlier_capping:
        series = _cap_outliers_iqr(series, config.outlier_iqr_multiplier)

    return series


def _cap_outliers_iqr(series: pd.Series, multiplier: float) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return series.clip(lower=lower, upper=upper)


def _clean_categorical_column(series: pd.Series, config: IngestionConfig) -> pd.Series:
    if config.categorical_impute_strategy == "mode":
        mode = series.mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else config.categorical_impute_constant
    elif config.categorical_impute_strategy == "constant":
        fill_value = config.categorical_impute_constant
    elif config.categorical_impute_strategy == "drop_rows":
        return series
    else:
        raise ValueError(
            f"Unknown categorical_impute_strategy: {config.categorical_impute_strategy}"
        )

    return series.fillna(fill_value)
