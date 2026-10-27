"""Forecast and alert metrics."""

import numpy as np
import pandas as pd


def forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    mape_epsilon: float = 1e-8,
) -> dict[str, float | int]:
    """Return MAE, guarded MAPE, and WAPE for aligned actual/predicted values."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have equal lengths.")
    if mape_epsilon <= 0:
        raise ValueError("mape_epsilon must be positive.")
    valid = actual.notna() & predicted.notna()
    y_true = actual.loc[valid].to_numpy(dtype=float)
    y_pred = predicted.loc[valid].to_numpy(dtype=float)
    if not len(y_true):
        raise ValueError("No paired non-null actual and predicted values.")
    errors = np.abs(y_true - y_pred)
    nonzero = np.abs(y_true) > mape_epsilon
    denominator = np.abs(y_true).sum()
    return {
        "mae": float(errors.mean()),
        "mape": float((errors[nonzero] / np.abs(y_true[nonzero])).mean())
        if nonzero.any()
        else float("nan"),
        "mape_excluded_count": int((~nonzero).sum()),
        "wape": float(errors.sum() / denominator) if denominator > 0 else float("nan"),
    }


def alert_metrics(actual_alert: pd.Series, predicted_alert: pd.Series) -> dict[str, float]:
    """Calculate binary precision and recall, treating undefined scores as zero."""
    if len(actual_alert) != len(predicted_alert):
        raise ValueError("Actual and predicted alert values must have equal lengths.")
    truth = actual_alert.fillna(False).astype(bool)
    predicted = predicted_alert.fillna(False).astype(bool)
    true_positive = int((truth & predicted).sum())
    false_positive = int((~truth & predicted).sum())
    false_negative = int((truth & ~predicted).sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "precision": true_positive / precision_denominator if precision_denominator else 0.0,
        "recall": true_positive / recall_denominator if recall_denominator else 0.0,
    }
