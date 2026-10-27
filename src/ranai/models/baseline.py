"""Seasonal-naive forecasting baseline."""

import pandas as pd


def seasonal_naive(
    frame: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    target_column: str,
    period: str = "7D",
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict from one period earlier, optionally using a separate history frame."""
    required = {entity_column, timestamp_column, target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    source = frame if history is None else history
    if source.duplicated([entity_column, timestamp_column]).any():
        raise ValueError("Baseline requires unique cell/timestamp rows.")

    if not {entity_column, timestamp_column, target_column}.issubset(source.columns):
        raise KeyError("Baseline history must include entity, timestamp, and target columns.")
    lookup = source[[entity_column, timestamp_column, target_column]].rename(
        columns={timestamp_column: "__source_timestamp", target_column: "prediction"}
    )
    predictions = frame[[entity_column, timestamp_column]].copy()
    predictions["__source_timestamp"] = pd.to_datetime(
        predictions[timestamp_column], errors="raise"
    ) - pd.Timedelta(period)
    result = predictions.merge(
        lookup,
        on=[entity_column, "__source_timestamp"],
        how="left",
        validate="many_to_one",
    )
    return result[[entity_column, timestamp_column, "prediction"]]
