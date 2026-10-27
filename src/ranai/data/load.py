"""Load raw KPI, configuration, and change-log files."""

from pathlib import Path

import pandas as pd
import yaml

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_table(path: str | Path) -> pd.DataFrame:
    """Read one CSV or Excel file without modifying the source."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    extension = source.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(source)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise ValueError(f"Unsupported input format {extension!r}: {source}")


def load_directory(path: str | Path) -> pd.DataFrame:
    """Read and concatenate supported files in a directory, in path order."""
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {directory}")
    files = sorted(
        file
        for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No CSV or Excel files found in {directory}")
    return pd.concat((load_table(file) for file in files), ignore_index=True)


def load_config(path: str | Path) -> dict:
    """Load a YAML mapping and report malformed or non-mapping config files."""
    source = Path(path)
    with source.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {source}")
    return config
