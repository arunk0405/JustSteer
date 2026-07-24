import pandas as pd

from ingestion.config import IngestionConfig
from ingestion.profiler import get_target_column, profile_dataset


def make_test_df():
    return pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(20)],       # identifier
            "temperature_c": [20.0 + i * 0.5 for i in range(20)],  # numeric
            "road_type": ["highway", "arterial"] * 10,       # categorical
            "is_rainy": [True, False] * 10,                   # boolean
            "risk_label": ["low", "high"] * 10,               # target (name hint)
        }
    )


def test_identifier_detected():
    df = make_test_df()
    profiles = profile_dataset(df, IngestionConfig())
    order_id_profile = next(p for p in profiles if p.name == "order_id")
    assert order_id_profile.inferred_type == "identifier"
    assert order_id_profile.suggested_role == "identifier"


def test_numeric_detected():
    df = make_test_df()
    profiles = profile_dataset(df, IngestionConfig())
    temp_profile = next(p for p in profiles if p.name == "temperature_c")
    assert temp_profile.inferred_type == "numeric"
    assert temp_profile.suggested_role == "feature"


def test_categorical_detected():
    df = make_test_df()
    profiles = profile_dataset(df, IngestionConfig())
    road_profile = next(p for p in profiles if p.name == "road_type")
    assert road_profile.inferred_type == "categorical"


def test_boolean_detected():
    df = make_test_df()
    profiles = profile_dataset(df, IngestionConfig())
    rainy_profile = next(p for p in profiles if p.name == "is_rainy")
    assert rainy_profile.inferred_type == "boolean"


def test_target_auto_detected_by_name_hint():
    df = make_test_df()
    profiles = profile_dataset(df, IngestionConfig())
    assert get_target_column(profiles) == "risk_label"


def test_target_override_via_config():
    df = make_test_df()
    config = IngestionConfig(target_column="road_type")
    profiles = profile_dataset(df, config)
    assert get_target_column(profiles) == "road_type"


def test_missing_pct_calculated_correctly():
    df = pd.DataFrame({"a": [1, 2, None, None, 5]})
    profiles = profile_dataset(df, IngestionConfig())
    a_profile = next(p for p in profiles if p.name == "a")
    assert a_profile.missing_count == 2
    assert a_profile.missing_pct == 0.4
