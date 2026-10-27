"""Leakage-aware temporal and cell-holdout split utilities."""

from collections.abc import Iterator

import pandas as pd


def temporal_split(
    frame: pd.DataFrame,
    timestamp_column: str,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically, reserving the latest fraction for validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if timestamp_column not in frame.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_column}")
    if frame.empty:
        raise ValueError("Cannot split an empty dataframe.")

    ordered = frame.sort_values(timestamp_column, kind="stable")
    timestamps = ordered[timestamp_column].drop_duplicates().sort_values()
    split_at = int(len(timestamps) * (1 - validation_fraction))
    if split_at == 0 or split_at == len(timestamps):
        raise ValueError("Not enough rows for the requested split.")
    validation_start = timestamps.iloc[split_at]
    train = ordered.loc[ordered[timestamp_column] < validation_start]
    validation = ordered.loc[ordered[timestamp_column] >= validation_start]
    return train.copy(), validation.copy()


def cell_holdout_split(
    frame: pd.DataFrame,
    entity_column: str,
    held_out_cells: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate rows for specified cells from all remaining cells."""
    if entity_column not in frame.columns:
        raise KeyError(f"Missing entity column: {entity_column}")
    held_out = frame[entity_column].isin(held_out_cells)
    if not held_out.any():
        raise ValueError("None of the requested held-out cells were found.")
    return frame.loc[~held_out].copy(), frame.loc[held_out].copy()


def cell_time_split(
    frame: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    held_out_cells: set[str],
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train earlier periods on non-held-out cells and test later held-out cells."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if entity_column not in frame or timestamp_column not in frame:
        raise KeyError("Entity and timestamp columns must both exist.")
    if not held_out_cells:
        raise ValueError("At least one held-out cell is required.")
    present = set(frame[entity_column].dropna())
    if not held_out_cells.intersection(present):
        raise ValueError("None of the requested held-out cells were found.")
    timestamps = frame[timestamp_column].drop_duplicates().sort_values()
    split_at = int(len(timestamps) * (1 - validation_fraction))
    if split_at == 0 or split_at == len(timestamps):
        raise ValueError("Not enough timestamps for the requested split.")
    validation_start = timestamps.iloc[split_at]
    training = frame.loc[
        (~frame[entity_column].isin(held_out_cells)) & (frame[timestamp_column] < validation_start)
    ].copy()
    validation = frame.loc[
        frame[entity_column].isin(held_out_cells) & (frame[timestamp_column] >= validation_start)
    ].copy()
    if training.empty or validation.empty:
        raise ValueError("The requested cell/time split produced an empty partition.")
    return training, validation


def iter_time_splits(
    frame: pd.DataFrame,
    timestamp_column: str,
    validation_fraction: float = 0.2,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield a single chronological train/validation split."""
    yield temporal_split(frame, timestamp_column, validation_fraction)
