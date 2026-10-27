"""Join KPI observations with configuration and parameter-change data."""

import pandas as pd


def join_static_config(
    kpi: pd.DataFrame,
    config: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """Attach one static configuration row per cell to KPI observations."""
    for name, frame in (("KPI", kpi), ("configuration", config)):
        if entity_column not in frame:
            raise KeyError(f"{name} data is missing {entity_column!r}.")
    if config[entity_column].duplicated().any():
        raise ValueError(f"Configuration must contain one row per {entity_column}.")
    return kpi.merge(config, on=entity_column, how="left", validate="many_to_one")


def add_change_history(
    kpi: pd.DataFrame,
    changes: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    change_timestamp_column: str = "change_timestamp",
) -> pd.DataFrame:
    """Attach the most recent change event at or before each KPI observation."""
    required_kpi = {entity_column, timestamp_column}
    required_changes = {entity_column, change_timestamp_column}
    if missing := required_kpi.difference(kpi.columns):
        raise KeyError(f"KPI data is missing columns: {sorted(missing)}")
    if missing := required_changes.difference(changes.columns):
        raise KeyError(f"Change log is missing columns: {sorted(missing)}")
    if kpi.empty or changes.empty:
        raise ValueError("KPI data and change log must both contain rows.")

    left = kpi.copy()
    right = changes.copy()
    left[timestamp_column] = pd.to_datetime(left[timestamp_column], errors="raise")
    right[change_timestamp_column] = pd.to_datetime(right[change_timestamp_column], errors="raise")
    left["__row_order"] = range(len(left))
    right = right.rename(columns={change_timestamp_column: "last_change_timestamp"})
    left = left.sort_values([timestamp_column, entity_column], kind="stable")
    right = right.sort_values(["last_change_timestamp", entity_column], kind="stable")
    result = pd.merge_asof(
        left,
        right,
        left_on=timestamp_column,
        right_on="last_change_timestamp",
        by=entity_column,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_change"),
    )
    result["hours_since_change"] = (
        result[timestamp_column] - result["last_change_timestamp"]
    ).dt.total_seconds() / 3600
    return result.sort_values("__row_order", kind="stable").drop(columns="__row_order")
