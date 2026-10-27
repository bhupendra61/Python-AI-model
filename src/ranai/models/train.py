"""Chronological LightGBM training with MLflow run tracking."""

from math import isfinite
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from ranai.models.baseline import seasonal_naive
from ranai.models.evaluate import forecast_metrics
from ranai.validation.splits import cell_time_split, temporal_split


def train_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    timestamp_column: str,
    validation_fraction: float,
    hyperparameters: dict[str, Any],
    model_output: str | Path,
    tracking_uri: str,
    experiment_name: str,
    early_stopping_rounds: int = 50,
    entity_column: str = "cell_id",
    held_out_cells: set[str] | None = None,
    data_version: str | None = None,
) -> tuple[LGBMRegressor, dict[str, float | int], pd.DataFrame]:
    """Train on earlier timestamps, evaluate later rows, and log the run."""
    required = {target_column, timestamp_column, entity_column, *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Training data is missing columns: {sorted(missing)}")
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    if target_column in feature_columns or timestamp_column in feature_columns:
        raise ValueError("Target and timestamp must not be used as model features.")

    if held_out_cells:
        fit_frame, evaluation = cell_time_split(
            frame,
            entity_column,
            timestamp_column,
            held_out_cells,
            validation_fraction,
        )
        train, validation = temporal_split(fit_frame, timestamp_column, validation_fraction)
    else:
        train, validation = temporal_split(frame, timestamp_column, validation_fraction)
        evaluation = validation
    train = train.dropna(subset=[target_column, *feature_columns])
    validation = validation.dropna(subset=[target_column, *feature_columns])
    evaluation = evaluation.dropna(subset=[target_column, *feature_columns])
    if train.empty or validation.empty or evaluation.empty:
        raise ValueError("Training and validation data must have usable feature rows.")
    if train[timestamp_column].max() >= validation[timestamp_column].min():
        raise ValueError("Early-stopping timestamps must occur strictly after training.")
    if train[timestamp_column].max() >= evaluation[timestamp_column].min():
        raise ValueError("Evaluation timestamps must occur strictly after training.")

    model = LGBMRegressor(**hyperparameters)
    model.fit(
        train[feature_columns],
        train[target_column],
        eval_X=validation[feature_columns],
        eval_y=validation[target_column],
        callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
    )
    predicted = model.predict(evaluation[feature_columns])
    metrics = forecast_metrics(
        evaluation[target_column], pd.Series(predicted, index=evaluation.index)
    )
    history = frame.loc[frame[timestamp_column] < evaluation[timestamp_column].min()]
    baseline = seasonal_naive(
        evaluation,
        entity_column=entity_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
        history=history,
    )
    baseline_prediction = baseline["prediction"].set_axis(evaluation.index)
    if baseline_prediction.notna().any():
        baseline_metrics = forecast_metrics(evaluation[target_column], baseline_prediction)
    else:
        baseline_metrics = {
            "mae": float("nan"),
            "mape": float("nan"),
            "mape_excluded_count": len(evaluation),
            "wape": float("nan"),
        }
    metrics.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(hyperparameters)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("feature_columns", ",".join(feature_columns))
        mlflow.log_param("validation_fraction", validation_fraction)
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("evaluation_rows", len(evaluation))
        mlflow.log_param("early_stopping_rows", len(validation))
        mlflow.log_param("held_out_cells", ",".join(sorted(held_out_cells or set())))
        if data_version is not None:
            mlflow.log_param("data_version", data_version)
        mlflow.log_metrics(
            {
                name: float(value)
                for name, value in metrics.items()
                if name != "mape_excluded_count" and isfinite(float(value))
            }
        )
        mlflow.log_metric("mape_excluded_count", float(metrics["mape_excluded_count"]))
        mlflow.lightgbm.log_model(model, artifact_path="model")

    output = Path(model_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": feature_columns, "target": target_column},
        output,
    )
    predictions = evaluation[[entity_column, timestamp_column, target_column]].copy()
    predictions["prediction"] = predicted
    predictions["baseline_prediction"] = baseline["prediction"].to_numpy()
    return model, metrics, predictions
