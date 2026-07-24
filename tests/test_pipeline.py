import json

import pandas as pd

from ingestion.config import IngestionConfig
from ingestion.pipeline import run_ingestion

SAMPLE_DATA = "sample_data/sample_delivery_data.csv"


def test_end_to_end_pipeline_on_sample_data(tmp_path):
    output_dir = tmp_path / "output"
    config = IngestionConfig(output_dir=str(output_dir))

    result = run_ingestion(SAMPLE_DATA, config)

    # No missing values should remain in the delivered dataset
    assert result.dataframe.isna().sum().sum() == 0

    # The synthetic duplicate row injected by generate_sample_data.py should be gone
    assert result.dataframe.duplicated().sum() == 0

    # Target column should be auto-detected via the "risk" name hint
    target_profile = next(p for p in result.profiles if p.suggested_role == "target")
    assert target_profile.name == "risk_label"

    # Identifier column should be recognized and excluded from features
    identifier_names = [p.name for p in result.profiles if p.suggested_role == "identifier"]
    assert "road_id" in identifier_names

    # Output files should actually exist and be loadable
    data_path = result.output_paths["data_path"]
    schema_path = result.output_paths["schema_path"]

    reloaded = pd.read_csv(data_path)
    assert len(reloaded) == len(result.dataframe)

    with open(schema_path) as f:
        schema = json.load(f)
    assert schema["target_column"] == "risk_label"
    assert schema["row_count"] == len(result.dataframe)
    assert "road_id" not in schema["feature_columns"]
