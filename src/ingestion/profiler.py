"""
Sub-step 1.2 — Attribute profiler.

This is the module that produces "proper information about the
attributes" for the model: for every column, what kind of data is it,
how much is missing, how many distinct values does it have, and what role
should it play (feature / target / identifier / drop).

The output (a list of ColumnProfile) is the contract between ingestion and
everything downstream — training, the FastAPI serving layer's expected
input schema, and later drift monitoring all read from this shape.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from ingestion.config import IngestionConfig

InferredType = str  # one of: "numeric", "categorical", "datetime", "boolean", "identifier", "text"


@dataclass
class ColumnProfile:
    name: str
    pandas_dtype: str
    inferred_type: InferredType
    missing_count: int
    missing_pct: float
    unique_count: int
    cardinality_ratio: float
    sample_values: list
    suggested_role: str  # "feature" | "target" | "identifier" | "drop"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _try_parse_datetime(series: pd.Series) -> bool:
    import warnings

    if series.dropna().empty:
        return False
    sample = series.dropna().head(50)
    try:
        with warnings.catch_warnings():
            # We're deliberately probing "does this parse as a date at all" —
            # the format-inference warning is expected noise here, not a bug.
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="raise", utc=False)
        return not parsed.isna().all()
    except (ValueError, TypeError):
        return False


def _infer_column_type(
    series: pd.Series, config: IngestionConfig, column_name: str
) -> InferredType:
    non_null = series.dropna()
    n = len(series)
    unique_count = series.nunique(dropna=True)

    if n == 0 or unique_count == 0:
        return "text"

    # Boolean check first — pandas bool dtype, or a 2-value {0,1}/{True,False}/{yes,no} column
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    normalized_uniques = set(str(v).strip().lower() for v in non_null.unique())
    if normalized_uniques.issubset({"true", "false", "0", "1", "yes", "no"}) and unique_count <= 2:
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        cardinality_ratio = unique_count / n
        # A numeric column being all-unique is NOT enough to call it an
        # identifier — a precise continuous measurement (e.g. rainfall_mm)
        # is very often all-unique in a small sample too. Only treat a
        # numeric column as an identifier if its name also looks like one
        # (id/code/key/no/number) — cardinality alone is too weak a signal.
        if (
            cardinality_ratio >= config.id_cardinality_ratio
            and unique_count == n
            and _name_looks_like_identifier(column_name)
        ):
            return "identifier"
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series) or _try_parse_datetime(series):
        return "datetime"

    # For string/object columns, distinguish free text (sentences) from
    # structured identifiers/categories BEFORE applying the cardinality
    # check — otherwise a column of unique sentences (e.g. "notes") gets
    # misclassified as an identifier just because every value differs.
    str_values = non_null.astype(str)
    contains_space_ratio = (
        str_values.str.contains(" ").mean() if len(str_values) else 0.0
    )
    looks_like_free_text = contains_space_ratio > 0.3

    cardinality_ratio = unique_count / n
    if not looks_like_free_text and cardinality_ratio >= config.id_cardinality_ratio:
        return "identifier"

    if unique_count <= config.categorical_max_unique:
        return "categorical"

    return "text"


def _name_looks_like_identifier(column_name: str) -> bool:
    tokens = _tokenize(column_name)
    id_hints = {"id", "code", "key", "no", "num", "number", "uuid", "identifier"}
    return any(token in id_hints for token in tokens)


def _tokenize(name: str) -> list[str]:
    import re

    return [t for t in re.split(r"[^a-zA-Z0-9]+", name.lower()) if t]


def _suggest_role(
    column_name: str,
    inferred_type: InferredType,
    missing_pct: float,
    config: IngestionConfig,
) -> str:
    if config.target_column and column_name == config.target_column:
        return "target"

    if inferred_type == "identifier":
        return "identifier"

    if not config.target_column:
        tokens = _tokenize(column_name)
        name_lower = column_name.lower()
        for hint in config.target_name_hints:
            hint_lower = hint.lower()
            # Single/short hints (like "y") must match a whole token, not
            # just appear as a substring — otherwise "y" matches inside
            # "road_type" (contains "ty"...no, contains the letter "y").
            # Longer hints are allowed as substrings (e.g. "risk" inside
            # "risk_label" even without an underscore boundary).
            if len(hint_lower) <= 2:
                if hint_lower in tokens:
                    return "target"
            elif hint_lower in name_lower:
                return "target"

    if inferred_type == "text":
        return "drop"

    if missing_pct >= config.max_missing_threshold_hard_fail:
        return "drop"

    return "feature"


def profile_dataset(df: pd.DataFrame, config: IngestionConfig) -> list[ColumnProfile]:
    """Produce a ColumnProfile for every column in the DataFrame."""
    profiles: list[ColumnProfile] = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = missing_count / n_rows if n_rows else 0.0
        unique_count = int(series.nunique(dropna=True))
        cardinality_ratio = unique_count / n_rows if n_rows else 0.0

        inferred_type = _infer_column_type(series, config, col)
        suggested_role = _suggest_role(col, inferred_type, missing_pct, config)

        sample_values = (
            series.dropna().unique()[:5].tolist() if unique_count > 0 else []
        )
        sample_values = [_json_safe(v) for v in sample_values]

        profiles.append(
            ColumnProfile(
                name=col,
                pandas_dtype=str(series.dtype),
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_pct=round(missing_pct, 4),
                unique_count=unique_count,
                cardinality_ratio=round(cardinality_ratio, 4),
                sample_values=sample_values,
                suggested_role=suggested_role,
            )
        )

    # If no explicit target_column was configured, name-hint matching could
    # in principle flag more than one column as "target" (e.g. a dataset
    # with both "risk_label" and "risk_score"). Keep only the first match
    # so exactly one target is ever produced automatically — genuine
    # multi-target datasets should set target_column explicitly instead.
    if not config.target_column:
        seen_target = False
        for p in profiles:
            if p.suggested_role == "target":
                if seen_target:
                    p.suggested_role = "feature"
                seen_target = True

    return profiles


def _json_safe(value):
    """Convert pandas/numpy scalar types into plain JSON-serializable Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def get_target_column(profiles: list[ColumnProfile]) -> Optional[str]:
    for p in profiles:
        if p.suggested_role == "target":
            return p.name
    return None
