"""
Sub-step 1.1 — Generic loader.

Accepts CSV, JSON, Parquet, or Excel by file extension, and gets a clean
pandas DataFrame out — nothing else. Type inference, cleaning, and
validation are deliberately NOT this module's job; keeping "get bytes into
a DataFrame" isolated means you can later swap this for a database query
or an API pull without touching profiling/cleaning code at all.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".json", ".parquet", ".xlsx", ".xls"}

# Encodings to try in order for CSV files with unknown/inconsistent encoding —
# common for scraped or Kaggle datasets that weren't saved as clean UTF-8.
_CSV_ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "latin1", "cp1252")


class UnsupportedFileTypeError(ValueError):
    pass


class EmptyDatasetError(ValueError):
    pass


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load any supported dataset file into a DataFrame.

    Raises:
        FileNotFoundError: path doesn't exist.
        UnsupportedFileTypeError: extension isn't one this engine handles.
        EmptyDatasetError: the file parsed but contains zero rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"'{suffix}' is not supported. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Loading dataset from %s (type=%s)", path, suffix)

    if suffix == ".csv":
        df = _load_csv_with_encoding_fallback(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:  # .xlsx / .xls
        df = pd.read_excel(path)

    if df.shape[0] == 0:
        raise EmptyDatasetError(f"{path} loaded successfully but contains 0 rows.")

    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def _load_csv_with_encoding_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in _CSV_ENCODING_FALLBACKS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            logger.warning("Encoding '%s' failed for %s, trying next fallback", encoding, path)
    raise UnicodeDecodeError(
        "unknown", b"", 0, 1, f"Could not decode {path} with any of {_CSV_ENCODING_FALLBACKS}"
    ) from last_error
