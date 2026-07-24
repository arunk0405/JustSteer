import numpy as np
import pandas as pd

from ingestion.cleaner import clean_dataset
from ingestion.config import IngestionConfig
from ingestion.profiler import profile_dataset


def test_numeric_missing_values_imputed_with_median():
    df = pd.DataFrame({"speed": [10.0, 20.0, np.nan, 40.0]})
    config = IngestionConfig()
    profiles = profile_dataset(df, config)

    cleaned = clean_dataset(df, profiles, config)

    assert cleaned["speed"].isna().sum() == 0
    assert cleaned["speed"].iloc[2] == df["speed"].median()


def test_categorical_missing_values_imputed_with_mode():
    # A distinguishing "row_num" column avoids accidental exact-duplicate
    # rows being dropped before we get to check the imputed value.
    df = pd.DataFrame(
        {
            "row_num": [1, 2, 3, 4],
            "road_type": ["highway", "highway", "arterial", None],
        }
    )
    config = IngestionConfig()
    profiles = profile_dataset(df, config)

    cleaned = clean_dataset(df, profiles, config)

    assert cleaned["road_type"].isna().sum() == 0
    assert cleaned["road_type"].iloc[3] == "highway"


def test_duplicate_rows_dropped():
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    config = IngestionConfig(drop_duplicate_rows=True)
    profiles = profile_dataset(df, config)

    cleaned = clean_dataset(df, profiles, config)

    assert len(cleaned) == 2


def test_identifier_column_left_untouched():
    df = pd.DataFrame({"order_id": [f"O{i}" for i in range(15)], "val": range(15)})
    config = IngestionConfig(id_cardinality_ratio=0.9)
    profiles = profile_dataset(df, config)

    cleaned = clean_dataset(df, profiles, config)

    assert list(cleaned["order_id"]) == list(df["order_id"])


def test_drop_role_columns_removed():
    long_text = [f"free text entry number {i} with unique content" for i in range(15)]
    df = pd.DataFrame({"notes": long_text, "val": range(15)})
    config = IngestionConfig(categorical_max_unique=5)
    profiles = profile_dataset(df, config)

    cleaned = clean_dataset(df, profiles, config)

    assert "notes" not in cleaned.columns
