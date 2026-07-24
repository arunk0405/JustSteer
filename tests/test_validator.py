import json

import pandas as pd
import pytest

from ingestion.config import IngestionConfig
from ingestion.profiler import profile_dataset
from ingestion.validator import ValidationError, validate_dataset


def test_too_few_rows_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    config = IngestionConfig(min_rows=10)
    profiles = profile_dataset(df, config)

    with pytest.raises(ValidationError):
        validate_dataset(df, profiles, config)


def test_column_over_hard_fail_threshold_raises():
    df = pd.DataFrame({"a": [1, None, None, None, None], "b": range(5)})
    config = IngestionConfig(min_rows=1, max_missing_threshold_hard_fail=0.5)
    profiles = profile_dataset(df, config)

    with pytest.raises(ValidationError):
        validate_dataset(df, profiles, config)


def test_soft_warning_for_moderate_missingness():
    df = pd.DataFrame({"a": [1, None, None, 4, 5, 6, 7, 8, 9, 10]})
    config = IngestionConfig(min_rows=1, missing_warn_threshold=0.1, max_missing_threshold_hard_fail=0.9)
    profiles = profile_dataset(df, config)

    warnings = validate_dataset(df, profiles, config)

    assert any("a" in w for w in warnings)


def test_schema_drift_detected_against_previous_run(tmp_path):
    schema_path = tmp_path / "schema_metadata.json"
    previous_schema = {
        "columns": [
            {"name": "a", "inferred_type": "numeric"},
            {"name": "b", "inferred_type": "categorical"},
        ]
    }
    schema_path.write_text(json.dumps(previous_schema))

    df = pd.DataFrame({"a": range(20), "c": range(20)})  # "b" gone, "c" new
    config = IngestionConfig(min_rows=1, schema_history_path=str(schema_path))
    profiles = profile_dataset(df, config)

    warnings = validate_dataset(df, profiles, config)

    assert any("missing now" in w for w in warnings)
    assert any("New columns" in w for w in warnings)
