import pandas as pd
import pytest

from ingestion.loaders import (
    EmptyDatasetError,
    UnsupportedFileTypeError,
    load_dataset,
)


def test_load_csv(tmp_path):
    p = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)

    df = load_dataset(p)

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.csv")


def test_load_unsupported_extension_raises(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("hello")
    with pytest.raises(UnsupportedFileTypeError):
        load_dataset(p)


def test_load_empty_csv_raises(tmp_path):
    p = tmp_path / "empty.csv"
    pd.DataFrame({"a": []}).to_csv(p, index=False)
    with pytest.raises(EmptyDatasetError):
        load_dataset(p)


def test_load_json(tmp_path):
    p = tmp_path / "data.json"
    pd.DataFrame({"a": [1, 2, 3]}).to_json(p, orient="records")
    df = load_dataset(p)
    assert len(df) == 3


def test_load_parquet(tmp_path):
    p = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(p)
    df = load_dataset(p)
    assert len(df) == 3
