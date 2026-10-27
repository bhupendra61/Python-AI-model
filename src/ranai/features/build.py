"""Assemble the final feature table from cleaned and enriched data."""

from collections.abc import Mapping, Sequence

import pandas as pd

from ranai.features.time_features import add_lag_rolling_features, add_time_features


def build_features(
    frame: pd.DataFrame,
    timestamp_column: str,
    entity_column: str,
    value_columns: Sequence[str],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    rolling_statistics: Sequence[str] = ("mean", "std"),
    holiday_dates: Sequence[str | pd.Timestamp] = (),
    ratios: dict[str, tuple[str, str]] | None = None,
    interval: str = "1h",
    time_feature_options: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Build calendar and history features for each cell."""
    calendar_features = time_feature_options or {
        "hour": True,
        "day_of_week": True,
        "day_of_year": True,
        "is_weekend": True,
        "is_holiday": True,
    }
    featured = add_time_features(
        frame,
        timestamp_column,
        holiday_dates,
        enabled=[name for name, enabled in calendar_features.items() if enabled],
    )
    for name, (numerator, denominator) in (ratios or {}).items():
        missing = {numerator, denominator}.difference(featured.columns)
        if missing:
            raise KeyError(f"Ratio {name!r} is missing columns: {sorted(missing)}")
        computed = featured[numerator].div(featured[denominator].where(featured[denominator] != 0))
        if name in featured:
            featured[name] = computed.combine_first(featured[name])
        else:
            featured[name] = computed
    return add_lag_rolling_features(
        featured,
        entity_column,
        timestamp_column,
        value_columns,
        lags,
        rolling_windows,
        rolling_statistics,
        interval,
    )
