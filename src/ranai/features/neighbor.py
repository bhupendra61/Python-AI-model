"""Neighbor-cell and same-site aggregate features."""

import pandas as pd


def add_neighbor_features(
    frame: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    value_columns: list[str],
    neighbors: dict[str, list[str]],
) -> pd.DataFrame:
    """Add timestamp-aligned means from each cell's explicitly listed neighbors."""
    required = {entity_column, timestamp_column, *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if frame.duplicated([entity_column, timestamp_column]).any():
        raise ValueError("Neighbor aggregation requires unique cell/timestamp rows.")

    result = frame.copy()
    lookup = frame.set_index([timestamp_column, entity_column])
    for value_column in value_columns:
        means = []
        for _, row in frame.iterrows():
            values = [
                lookup[value_column].get((row[timestamp_column], neighbor))
                for neighbor in neighbors.get(row[entity_column], [])
                if neighbor != row[entity_column]
            ]
            observed = [value for value in values if pd.notna(value)]
            means.append(sum(observed) / len(observed) if observed else pd.NA)
        result[f"{value_column}_neighbor_mean"] = means
    return result


def add_site_features(
    frame: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    site_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """Add per-timestamp, same-site means excluding the current cell."""
    required = {entity_column, timestamp_column, site_column, *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    for column in value_columns:
        group = result.groupby([site_column, timestamp_column], sort=False)[column]
        total = group.transform("sum")
        count = group.transform("count")
        current = result[column].fillna(0)
        other_count = count - result[column].notna().astype(int)
        result[f"{column}_same_site_mean"] = (total - current).div(
            other_count.where(other_count > 0)
        )
    return result
