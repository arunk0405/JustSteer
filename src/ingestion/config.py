"""
Central configuration for the ingestion engine. Everything that's a
"policy decision" (thresholds, imputation strategy, etc.) lives here —
never hardcoded inside the loader/profiler/cleaner logic — so the same
code can be re-tuned for a very different dataset without touching Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class IngestionConfig:
    # --- Profiling thresholds ---
    id_cardinality_ratio: float = 0.9
    # A column where (unique values / row count) exceeds this is treated as
    # an identifier, not a feature — e.g. an order_id or road_id.

    categorical_max_unique: int = 30
    # A non-numeric column with more distinct values than this is treated
    # as free text rather than a categorical feature.

    missing_warn_threshold: float = 0.30
    # Columns missing more than this fraction of values are flagged in the
    # validation report (not silently dropped — that's a human decision).

    # --- Cleaning strategy ---
    numeric_impute_strategy: str = "median"   # "median" | "mean" | "drop_rows"
    categorical_impute_strategy: str = "mode"  # "mode" | "constant" | "drop_rows"
    categorical_impute_constant: str = "missing"
    drop_duplicate_rows: bool = True
    outlier_capping: bool = False
    outlier_iqr_multiplier: float = 1.5

    # --- Validation gates ---
    min_rows: int = 10
    max_missing_threshold_hard_fail: float = 0.95
    # A column missing more than this fraction of values fails validation
    # outright rather than just warning — imputing 96%+ of a column
    # manufactures data, it doesn't clean it.

    # --- Target column ---
    target_column: Optional[str] = None
    target_name_hints: list = field(
        default_factory=lambda: ["target", "label", "risk", "class", "y"]
    )

    # --- I/O ---
    output_dir: str = "output"
    schema_history_path: str = "output/schema_metadata.json"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IngestionConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
