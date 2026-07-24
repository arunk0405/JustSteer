"""
Writes the two artifacts this whole engine exists to produce:
- processed_data.csv    — the cleaned DataFrame, ready for a model
- schema_metadata.json  — the attribute information the model (and later
                           the drift monitor) needs about every column
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingestion.profiler import ColumnProfile


def export_results(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    warnings: list[str],
    output_dir: str | Path,
    source_path: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "processed_data.csv"
    schema_path = output_dir / "schema_metadata.json"

    df.to_csv(data_path, index=False)

    schema = {
        "source_file": str(source_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [p.to_dict() for p in profiles],
        "target_column": next(
            (p.name for p in profiles if p.suggested_role == "target"), None
        ),
        "feature_columns": [p.name for p in profiles if p.suggested_role == "feature"],
        "identifier_columns": [p.name for p in profiles if p.suggested_role == "identifier"],
        "dropped_columns": [p.name for p in profiles if p.suggested_role == "drop"],
        "warnings": warnings,
    }

    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    return {"data_path": str(data_path), "schema_path": str(schema_path)}
