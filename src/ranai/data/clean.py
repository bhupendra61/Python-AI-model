"""Clean KPI data and produce per-cell data-quality summaries."""

from collections.abc import Sequence

import pandas as pd


def clean_kpi(
    frame: pd.DataFrame,
    timestamp_column: str,
    entity_column: str,
    counter_columns: Sequence[str] = (),
    outage_flag_column: str | None = None,
    zero_is_outage_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Validate keys, flag counter resets/outages, and null known outage zeros.

    Vendor-specific outage and counter semantics must be mapped in configuration;
    this function does not infer that every zero-valued KPI is an outage.
    """
    required = {timestamp_column, entity_column, *counter_columns, *zero_is_outage_columns}
    if outage_flag_column:
        required.add(outage_flag_column)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot clean an empty KPI dataframe.")

    cleaned = frame.copy()
    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="raise")
    if cleaned[[timestamp_column, entity_column]].isna().any().any():
        raise ValueError("Timestamp and cell identifier columns cannot contain nulls.")
    if cleaned.duplicated([entity_column, timestamp_column]).any():
        raise ValueError("KPI input must have at most one row per cell and timestamp.")
    cleaned = cleaned.sort_values([entity_column, timestamp_column], kind="stable")
    cleaned["counter_reset"] = False
    for column in counter_columns:
        reset = cleaned.groupby(entity_column, sort=False)[column].diff().lt(0)
        cleaned["counter_reset"] |= reset.fillna(False)

    if outage_flag_column:
        flags = cleaned[outage_flag_column]
        if pd.api.types.is_bool_dtype(flags):
            outage = flags.fillna(False)
        else:
            outage = (
                flags.astype("string")
                .str.strip()
                .str.lower()
                .isin({"1", "true", "yes", "y", "outage"})
            )
    else:
        outage = pd.Series(False, index=cleaned.index)
    for column in zero_is_outage_columns:
        outage |= cleaned[column].eq(0)
    cleaned["outage"] = outage
    for column in zero_is_outage_columns:
        cleaned.loc[outage, column] = pd.NA
    return cleaned.reset_index(drop=True)


def quality_report(
    frame: pd.DataFrame,
    timestamp_column: str,
    entity_column: str,
    interval: str = "1h",
    value_columns: Sequence[str] = (),
    outlier_bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Summarize interval gaps, missing values, resets, and outage hours by cell."""
    required = {timestamp_column, entity_column, *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    expected_delta = pd.Timedelta(interval)
    if expected_delta <= pd.Timedelta(0):
        raise ValueError("interval must be a positive pandas duration.")
    summaries = []
    for cell_id, group in frame.groupby(entity_column, sort=True):
        timestamps = pd.DatetimeIndex(pd.to_datetime(group[timestamp_column])).unique()
        if timestamps.empty:
            continue
        expected = pd.date_range(timestamps.min(), timestamps.max(), freq=expected_delta)
        missing_intervals = len(expected.difference(timestamps))
        summary = {
            entity_column: cell_id,
            "observed_rows": len(group),
            "expected_intervals": len(expected),
            "missing_intervals": missing_intervals,
            "missing_rate": missing_intervals / len(expected),
            "outage_hours": int(group["outage"].sum()) if "outage" in group else 0,
            "counter_resets": int(group["counter_reset"].sum()) if "counter_reset" in group else 0,
        }
        for column in value_columns:
            summary[f"{column}_missing_rate"] = float(group[column].isna().mean())
            bounds = (outlier_bounds or {}).get(column)
            if bounds:
                lower, upper = bounds
                outlier = pd.Series(False, index=group.index)
                if lower is not None:
                    outlier |= group[column] < lower
                if upper is not None:
                    outlier |= group[column] > upper
                summary[f"{column}_outlier_count"] = int(outlier.sum())
        summaries.append(summary)
    return pd.DataFrame(summaries)
