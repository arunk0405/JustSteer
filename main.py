"""
CLI entry point.

Usage:
    python main.py --input sample_data/sample_delivery_data.csv
    python main.py --input path/to/any_dataset.csv --target-col risk_label
    python main.py --input data.csv --config configs/default_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running this script directly (e.g. `python main.py`) from the repo
# root without needing to `pip install -e .` or set PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ingestion.config import IngestionConfig
from ingestion.pipeline import run_ingestion


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic dataset ingestion engine")
    parser.add_argument("--input", required=True, help="Path to the input dataset")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    parser.add_argument("--target-col", default=None, help="Override target column name")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    config = IngestionConfig.from_yaml(args.config) if args.config else IngestionConfig()
    if args.target_col:
        config.target_column = args.target_col
    if args.output_dir:
        config.output_dir = args.output_dir
        config.schema_history_path = f"{args.output_dir}/schema_metadata.json"

    try:
        result = run_ingestion(args.input, config)
    except Exception as exc:
        logging.getLogger(__name__).error("Ingestion failed: %s", exc)
        return 1

    print("\n--- Ingestion summary ---")
    print(f"Rows: {len(result.dataframe)}  Columns: {len(result.dataframe.columns)}")
    print(f"Target column: {next((p.name for p in result.profiles if p.suggested_role == 'target'), 'NOT DETECTED')}")
    print(f"Feature columns: {[p.name for p in result.profiles if p.suggested_role == 'feature']}")
    print(f"Identifier columns: {[p.name for p in result.profiles if p.suggested_role == 'identifier']}")
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  - {w}")
    print(f"\nProcessed data written to: {result.output_paths['data_path']}")
    print(f"Schema metadata written to: {result.output_paths['schema_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
