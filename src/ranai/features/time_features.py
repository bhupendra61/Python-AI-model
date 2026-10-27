"""Calendar, lag, and leakage-safe rolling-window feature generation."""

from collections.abc import Sequence

import pandas as pd


def add_time_features(
    frame: pd.DataFrame,
    timestamp_column: str,
    holiday_dates: Sequence[str | pd.Timestamp] = (),
    enabled: Sequence[str] = ("hour", "day_of_week", "day_of_year", "is_weekend", "is_holiday"),
) -> pd.DataFrame:
    """Add calendar fields and an optional date-based holiday indicator."""
    if timestamp_column not in frame:
        raise KeyError(f"Missing timestamp column: {timestamp_column}")
    result = frame.copy()
    timestamp = pd.to_datetime(result[timestamp_column], errors="raise")
    if "hour" in enabled:
        result["hour"] = timestamp.dt.hour
    if "day_of_week" in enabled:
        result["day_of_week"] = timestamp.dt.day_of_week
    if "day_of_year" in enabled:
        result["day_of_year"] = timestamp.dt.day_of_year
    if "is_weekend" in enabled:
        result["is_weekend"] = timestamp.dt.day_of_week >= 5
    if "is_holiday" in enabled:
        holiday_days = pd.to_datetime(pd.Series(holiday_dates)).dt.date
        result["is_holiday"] = timestamp.dt.date.isin(set(holiday_days.dropna()))
    return result


def add_lag_rolling_features(
    frame: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    value_columns: Sequence[str],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    rolling_statistics: Sequence[str] = ("mean", "std"),
    interval: str = "1h",
) -> pd.DataFrame:
    """Add exact-time lags and rolling windows using prior observations only."""
    required = {entity_column, timestamp_column, *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if any(lag < 1 for lag in lags) or any(window < 1 for window in rolling_windows):
        raise ValueError("Lags and rolling windows must be positive integers.")
    unsupported = set(rolling_statistics).difference({"mean", "std", "min", "max"})
    if unsupported:
        raise ValueError(f"Unsupported rolling statistics: {sorted(unsupported)}")
    delta = pd.Timedelta(interval)
    if delta <= pd.Timedelta(0):
        raise ValueError("interval must be a positive pandas duration.")

    result = (
        frame.sort_values([entity_column, timestamp_column], kind="stable")
        .reset_index(drop=True)
        .copy()
    )
    for indexes in result.groupby(entity_column, sort=False).groups.values():
        group = result.loc[indexes]
        timestamps = pd.DatetimeIndex(group[timestamp_column])
        for column in value_columns:
            numeric_values = pd.to_numeric(group[column], errors="raise").to_numpy(
                dtype="float64", na_value=float("nan")
            )
            values = pd.Series(numeric_values, index=timestamps)
            for lag in lags:
                target_times = timestamps - delta * lag
                result.loc[group.index, f"{column}_lag_{lag}"] = values.reindex(
                    target_times
                ).to_numpy()
            for window in rolling_windows:
                rolling = values.rolling(
                    window=delta * window,
                    min_periods=1,
                    closed="left",
                )
                for statistic in rolling_statistics:
                    result.loc[group.index, f"{column}_rolling_{statistic}_{window}"] = getattr(
                        rolling, statistic
                    )().to_numpy()
    return result
