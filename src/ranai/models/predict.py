"""Model loading, batch prediction, and residual-based anomaly scores."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ranai.features.build import build_features
from ranai.features.neighbor import add_neighbor_features, add_site_features


def predict_model(model_bundle: dict, frame: pd.DataFrame) -> pd.Series:
    """Predict using the model and feature order recorded in its model bundle."""
    required = set(model_bundle["features"])
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Prediction data is missing features: {sorted(missing)}")
    return pd.Series(
        model_bundle["model"].predict(frame[model_bundle["features"]]),
        index=frame.index,
        name="prediction",
    )


def forecast_recursive(
    model_bundle: dict,
    history: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    target_column: str,
    horizon_steps: int,
    interval: str,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    rolling_statistics: Sequence[str] = ("mean", "std"),
    holiday_dates: Sequence[str | pd.Timestamp] = (),
    ratios: Mapping[str, tuple[str, str]] | None = None,
    neighbors: Mapping[str, list[str]] | None = None,
    site_column: str | None = None,
    time_feature_options: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Recursively forecast future periods using each cell's prior observations."""
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive.")
    if history.empty:
        raise ValueError("Cannot forecast from empty history.")
    ratio_config = dict(ratios or {})
    required = {entity_column, timestamp_column}
    if target_column not in ratio_config:
        required.add(target_column)
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"Forecast history is missing columns: {sorted(missing)}")
    if history.duplicated([entity_column, timestamp_column]).any():
        raise ValueError("Forecast history must have unique cell/timestamp rows.")
    delta = pd.Timedelta(interval)
    if delta <= pd.Timedelta(0):
        raise ValueError("interval must be a positive pandas duration.")

    working = history.copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], errors="raise")
    if target_column in ratio_config and target_column not in working:
        numerator, denominator = ratio_config[target_column]
        if numerator not in working or denominator not in working:
            raise KeyError(f"History is missing ratio inputs for {target_column!r}.")
        working[target_column] = working[numerator].div(
            working[denominator].where(working[denominator] != 0)
        )
    forecasts = []
    lagged_values = [f"{target_column}_lag_{lag}" for lag in lags]
    ratio_lags = [f"{name}_lag_{lag}" for name in (ratios or {}) for lag in lags]
    spatial_values = [name for name in lagged_values + ratio_lags if name.endswith("_lag_1")]
    for _ in range(horizon_steps):
        latest = (
            working.sort_values(timestamp_column, kind="stable")
            .groupby(entity_column, sort=False)
            .tail(1)
            .copy()
        )
        next_rows = latest.copy()
        next_rows[timestamp_column] = next_rows[timestamp_column] + delta
        next_rows[target_column] = float("nan")
        combined = pd.concat([working, next_rows], ignore_index=True)
        featured = build_features(
            combined,
            timestamp_column,
            entity_column,
            [target_column, *(ratios or {}).keys()],
            lags,
            rolling_windows,
            rolling_statistics,
            holiday_dates,
            ratio_config,
            interval,
            time_feature_options,
        )
        if neighbors:
            featured = add_neighbor_features(
                featured, entity_column, timestamp_column, spatial_values, dict(neighbors)
            )
        if site_column and site_column in featured.columns:
            featured = add_site_features(
                featured,
                entity_column,
                timestamp_column,
                site_column,
                spatial_values,
            )
        keys = pd.MultiIndex.from_frame(next_rows[[entity_column, timestamp_column]])
        future_features = featured.set_index([entity_column, timestamp_column]).loc[keys]
        predicted = predict_model(model_bundle, future_features)
        next_rows[target_column] = predicted.to_numpy()
        forecasts.append(
            next_rows[[entity_column, timestamp_column, target_column]].assign(
                prediction=predicted.to_numpy()
            )
        )
        working = pd.concat([working, next_rows], ignore_index=True)
    return pd.concat(forecasts, ignore_index=True)


def residual_anomalies(
    actual: pd.Series,
    predicted: pd.Series,
    entity_ids: pd.Series,
    threshold_quantile: float = 0.99,
    residual_window: int = 168,
) -> pd.DataFrame:
    """Score residuals against a per-cell quantile of prior absolute residuals."""
    if not (len(actual) == len(predicted) == len(entity_ids)):
        raise ValueError("Actual, predicted, and entity series must have equal lengths.")
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    if residual_window < 1:
        raise ValueError("residual_window must be positive.")
    result = pd.DataFrame(
        {"entity_id": entity_ids, "actual": actual, "prediction": predicted}
    ).reset_index(drop=True)
    result["residual"] = result["actual"] - result["prediction"]
    result["anomaly_score"] = result["residual"].abs()
    prior = result.groupby("entity_id", sort=False)["anomaly_score"].shift(1)
    thresholds = (
        prior.groupby(result["entity_id"], sort=False)
        .rolling(residual_window, min_periods=1)
        .quantile(threshold_quantile)
    )
    result["anomaly_threshold"] = thresholds.reset_index(level=0, drop=True)
    result["is_anomaly"] = result["anomaly_score"] > result["anomaly_threshold"]
    return result


def load_model(path: str | Path):
    """Load a persisted model bundle."""
    import joblib

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model file does not exist: {source}")
    return joblib.load(source)
