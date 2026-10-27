"""Command-line entry point for feature generation."""

import argparse
import csv
from pathlib import Path

import pandas as pd

from ranai.data.load import load_config
from ranai.features.build import build_features
from ranai.features.neighbor import add_neighbor_features, add_site_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    args = parser.parse_args()
    data_config = load_config(args.data_config)
    feature_config = load_config(args.features_config)
    technology = data_config.get("technology", "lte")
    source = Path(data_config["paths"]["interim"]) / f"{technology}_joined.parquet"
    frame = pd.read_parquet(source)

    grain = data_config.get("grain", "hourly")
    holiday_dates = []
    holiday_file = feature_config.get("time_features", {}).get("holiday_dates_file")
    if holiday_file:
        holiday_path = Path(holiday_file)
        if not holiday_path.is_file():
            raise FileNotFoundError(f"Configured holiday calendar not found: {holiday_path}")
        with holiday_path.open(newline="", encoding="utf-8-sig") as file:
            holiday_dates = [row["date"] for row in csv.DictReader(file) if row.get("date")]
    features = build_features(
        frame,
        timestamp_column=feature_config["timestamp_column"],
        entity_column=feature_config["entity_column"],
        value_columns=feature_config.get("target_columns", []),
        lags=feature_config["lags"][grain],
        rolling_windows=feature_config["rolling_windows"][grain],
        rolling_statistics=feature_config.get("rolling_statistics", ["mean", "std"]),
        holiday_dates=holiday_dates,
        ratios=feature_config.get("ratios", {}),
        interval=data_config.get("quality", {}).get("interval", "1h"),
        time_feature_options={
            name: enabled
            for name, enabled in feature_config.get("time_features", {}).items()
            if name != "holiday_dates_file"
        },
    )
    value_columns = [
        f"{column}_lag_1"
        for column in (
            feature_config.get("target_columns", []) + list(feature_config.get("ratios", {}))
        )
        if f"{column}_lag_1" in features.columns
    ]
    neighbors = feature_config.get("neighbors", {})
    if neighbors:
        features = add_neighbor_features(
            features,
            feature_config["entity_column"],
            feature_config["timestamp_column"],
            value_columns,
            neighbors,
        )
    site_column = feature_config.get("site_column")
    if site_column in features.columns:
        features = add_site_features(
            features,
            feature_config["entity_column"],
            feature_config["timestamp_column"],
            site_column,
            value_columns,
        )
    output_dir = Path(data_config["paths"]["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_dir / f"{technology}_features.parquet", index=False)


if __name__ == "__main__":
    main()
