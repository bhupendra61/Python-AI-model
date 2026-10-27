"""Command-line entry point for forecast evaluation."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ranai.models.evaluate import alert_metrics, forecast_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="CSV with target and prediction columns")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--timestamp-column", default="timestamp")
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    metrics = forecast_metrics(frame[args.target_column], frame[args.prediction_column])
    if "baseline_prediction" in frame.columns:
        baseline = forecast_metrics(frame[args.target_column], frame["baseline_prediction"])
        metrics.update({f"baseline_{key}": value for key, value in baseline.items()})
        baseline_wape = baseline["wape"]
        metrics["wape_improvement_over_baseline"] = (
            1 - metrics["wape"] / baseline_wape
            if math.isfinite(baseline_wape) and baseline_wape > 0
            else float("nan")
        )
    if {"actual_alert", "predicted_alert"}.issubset(frame.columns):
        metrics.update(alert_metrics(frame["actual_alert"], frame["predicted_alert"]))
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    output = reports / f"{Path(args.predictions).stem}_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if args.timestamp_column in frame.columns:
        plot_data = frame.copy()
        plot_data[args.timestamp_column] = pd.to_datetime(
            plot_data[args.timestamp_column], errors="raise"
        )
        columns = [args.target_column, args.prediction_column]
        if "baseline_prediction" in plot_data:
            columns.append("baseline_prediction")
        series = plot_data.groupby(args.timestamp_column)[columns].mean().sort_index()
        axis = series.plot(figsize=(12, 6), title="Forecast evaluation")
        axis.set_xlabel("Timestamp")
        axis.set_ylabel(args.target_column)
        axis.figure.tight_layout()
        axis.figure.savefig(figures / f"{Path(args.predictions).stem}_forecast.png")
        plt.close(axis.figure)
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
