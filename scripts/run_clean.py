"""Command-line entry point for cleaning raw KPI data."""

import argparse
from pathlib import Path

import pandas as pd

from ranai.data.clean import clean_kpi, quality_report
from ranai.data.load import load_config, load_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["columns"]
    technology = config.get("technology", "lte")
    raw_dir = Path(config["paths"]["raw"]) / "kpi" / technology
    frame = load_directory(raw_dir)
    frame = frame.rename(columns=data.get("column_mapping", {}))
    for source, canonical in data.get("kpis", {}).items():
        if source in frame.columns and source != canonical:
            frame = frame.rename(columns={source: canonical})
    frame[data["timestamp"]] = pd.to_datetime(frame[data["timestamp"]], errors="raise")
    date_range = config.get("date_range", {})
    if date_range.get("start") is not None:
        frame = frame.loc[frame[data["timestamp"]] >= pd.Timestamp(date_range["start"])]
    if date_range.get("end") is not None:
        end_exclusive = pd.Timestamp(date_range["end"]) + pd.Timedelta(days=1)
        frame = frame.loc[frame[data["timestamp"]] < end_exclusive]
    cleaned = clean_kpi(
        frame,
        timestamp_column=data["timestamp"],
        entity_column=data["cell_id"],
        counter_columns=data.get("counters", []),
        outage_flag_column=data.get("outage_flag"),
        zero_is_outage_columns=config.get("quality", {}).get("zero_is_outage_columns", []),
    )
    output_dir = Path(config["paths"]["interim"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_dir / f"{technology}_clean.parquet", index=False)
    report = quality_report(
        cleaned,
        data["timestamp"],
        data["cell_id"],
        interval=config.get("quality", {}).get("interval", "1h"),
        value_columns=list(data.get("kpis", {}).values()),
        outlier_bounds=config.get("quality", {}).get("outlier_bounds", {}),
    )
    report.to_csv(output_dir / f"{technology}_data_quality.csv", index=False)


if __name__ == "__main__":
    main()
