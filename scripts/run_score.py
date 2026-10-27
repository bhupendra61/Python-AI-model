"""Generate recursive cell forecasts and write a ranked prediction report."""

import argparse
import csv
from pathlib import Path

import pandas as pd

from ranai.data.load import load_config
from ranai.models.predict import (
    forecast_recursive,
    load_model,
    predict_model,
    residual_anomalies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--model-config", default="configs/model_lgbm.yaml")
    args = parser.parse_args()
    data_config = load_config(args.data_config)
    feature_config = load_config(args.features_config)
    model_config = load_config(args.model_config)
    technology = data_config.get("technology", "lte")
    frame = pd.read_parquet(
        Path(data_config["paths"]["processed"]) / f"{technology}_features.parquet"
    )
    bundle = load_model(model_config["model_output"])
    entity_column = feature_config["entity_column"]
    timestamp_column = feature_config["timestamp_column"]
    target = bundle["target"]
    anomaly = model_config.get("anomaly", {})
    alerts = pd.DataFrame(columns=[entity_column, "anomaly_score", "is_anomaly"])
    if target in frame.columns:
        historical_prediction = predict_model(bundle, frame)
        alerts = residual_anomalies(
            frame[target],
            historical_prediction,
            frame[entity_column],
            threshold_quantile=anomaly.get("threshold_quantile", 0.99),
            residual_window=anomaly.get("residual_window", 168),
        )
        alerts[entity_column] = frame[entity_column].to_numpy()
        alerts[timestamp_column] = frame[timestamp_column].to_numpy()
        alerts = (
            alerts.sort_values(timestamp_column, kind="stable")
            .groupby(entity_column, sort=False)
            .tail(1)[[entity_column, "anomaly_score", "is_anomaly"]]
        )

    history = pd.read_parquet(
        Path(data_config["paths"]["interim"]) / f"{technology}_joined.parquet"
    )
    grain = data_config.get("grain", "hourly")
    holiday_dates = []
    holiday_file = feature_config.get("time_features", {}).get("holiday_dates_file")
    if holiday_file:
        holiday_path = Path(holiday_file)
        if not holiday_path.is_file():
            raise FileNotFoundError(f"Configured holiday calendar not found: {holiday_path}")
        with holiday_path.open(newline="", encoding="utf-8-sig") as file:
            holiday_dates = [row["date"] for row in csv.DictReader(file) if row.get("date")]
    forecast = model_config.get("forecast", {})
    scored = forecast_recursive(
        model_bundle=bundle,
        history=history,
        entity_column=entity_column,
        timestamp_column=timestamp_column,
        target_column=target,
        horizon_steps=forecast.get("horizon_steps", 24),
        interval=data_config.get("quality", {}).get("interval", "1h"),
        lags=feature_config["lags"][grain],
        rolling_windows=feature_config["rolling_windows"][grain],
        rolling_statistics=feature_config.get("rolling_statistics", ["mean", "std"]),
        holiday_dates=holiday_dates,
        ratios=feature_config.get("ratios", {}),
        neighbors=feature_config.get("neighbors", {}),
        site_column=feature_config.get("site_column"),
        time_feature_options={
            name: enabled
            for name, enabled in feature_config.get("time_features", {}).items()
            if name != "holiday_dates_file"
        },
    )
    importance = getattr(bundle["model"], "feature_importances_", None)
    if importance is not None:
        ranked = sorted(
            zip(bundle["features"], importance, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        scored["top_features"] = ", ".join(name for name, _ in ranked[:5])
    scored = scored.merge(alerts, on=entity_column, how="left", validate="many_to_one")
    scored["is_anomaly"] = scored["is_anomaly"].fillna(False).astype(bool)

    scored = scored.sort_values(
        ["is_anomaly", "prediction"], ascending=[False, False], kind="stable"
    )
    scored = (
        scored.sort_values(feature_config["timestamp_column"], kind="stable")
        .groupby(feature_config["entity_column"], sort=False)
        .tail(1)
        .sort_values(["is_anomaly", "prediction"], ascending=[False, False], kind="stable")
    )
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    scored.to_csv(reports / f"{technology}_ranked_cells.csv", index=False)
    scored.to_parquet(reports / f"{technology}_ranked_cells.parquet", index=False)


if __name__ == "__main__":
    main()
