"""Command-line entry point for model training."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ranai.data.load import load_config
from ranai.models.train import train_model


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
    feature_path = Path(data_config["paths"]["processed"]) / f"{technology}_features.parquet"
    digest = hashlib.sha256()
    with feature_path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    frame = pd.read_parquet(feature_path)
    data_version = digest.hexdigest()
    target = model_config["target_column"]
    data_columns = data_config.get("columns", {})
    raw_kpis = set(feature_config.get("target_columns", []))
    raw_kpis.update(data_columns.get("kpis", {}).values())
    raw_kpis.update(data_columns.get("counters", []))
    raw_kpis.update(feature_config.get("ratios", {}).keys())
    raw_kpis.update({"outage", "counter_reset"})
    held_out_cells = set(data_columns.get("held_out_cells", []))
    if not held_out_cells:
        raise ValueError(
            "Set columns.held_out_cells in the data config for unseen-cell evaluation."
        )
    columns = feature_config.get("feature_columns") or [
        column
        for column in frame.select_dtypes(include="number").columns
        if column
        not in {
            target,
            feature_config["timestamp_column"],
            feature_config["entity_column"],
            *raw_kpis,
        }
    ]
    columns = [column for column in columns if column in frame.columns]
    training = model_config["training"]
    _, metrics, predictions = train_model(
        frame=frame,
        feature_columns=columns,
        target_column=target,
        timestamp_column=feature_config["timestamp_column"],
        validation_fraction=training["validation_fraction"],
        hyperparameters=model_config["hyperparameters"],
        model_output=model_config["model_output"],
        tracking_uri=model_config["tracking_uri"],
        experiment_name=model_config["experiment_name"],
        early_stopping_rounds=training["early_stopping_rounds"],
        entity_column=feature_config["entity_column"],
        held_out_cells=held_out_cells,
        data_version=data_version,
    )
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    predictions.to_csv(reports / f"{technology}_evaluation_predictions.csv", index=False)
    (reports / f"{technology}_training_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(metrics)


if __name__ == "__main__":
    main()
