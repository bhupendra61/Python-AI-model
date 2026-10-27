import pandas as pd

from ranai.models.baseline import seasonal_naive
from ranai.models.evaluate import alert_metrics, forecast_metrics
from ranai.models.predict import forecast_recursive, residual_anomalies


def test_seasonal_naive_uses_same_cell_one_week_earlier() -> None:
    history = pd.DataFrame(
        {
            "cell_id": ["a", "a", "b"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-01"]),
            "traffic": [10.0, 100.0, 20.0],
        }
    )
    requested = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "timestamp": pd.to_datetime(["2025-01-08", "2025-01-08"]),
            "traffic": [100.0, 30.0],
        }
    )

    prediction = seasonal_naive(requested, "cell_id", "timestamp", "traffic", history=history)

    assert prediction.loc[0, "prediction"] == 10.0
    assert prediction.loc[1, "prediction"] == 20.0


def test_forecast_metrics_exclude_near_zero_from_mape() -> None:
    metrics = forecast_metrics(pd.Series([0.0, 10.0]), pd.Series([5.0, 12.0]), mape_epsilon=1e-6)

    assert metrics["mae"] == 3.5
    assert metrics["mape"] == 0.2
    assert metrics["mape_excluded_count"] == 1
    assert metrics["wape"] == 0.7


def test_alert_metrics_and_residual_threshold_use_prior_residuals() -> None:
    metrics = alert_metrics(pd.Series([True, False, True]), pd.Series([True, True, False]))
    anomaly = residual_anomalies(
        pd.Series([1.0, 2.0, 100.0]),
        pd.Series([0.0, 0.0, 0.0]),
        pd.Series(["a", "a", "a"]),
        threshold_quantile=0.5,
    )

    assert metrics == {"precision": 0.5, "recall": 0.5}
    assert anomaly["is_anomaly"].tolist() == [False, True, True]


def test_recursive_forecast_feeds_predictions_into_future_lags() -> None:
    class IncrementingModel:
        def predict(self, features: pd.DataFrame) -> pd.Series:
            return features["traffic_lag_1"] + 1

    history = pd.DataFrame(
        {
            "cell_id": ["a", "a"],
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "traffic": [1.0, 2.0],
        }
    )
    result = forecast_recursive(
        {"model": IncrementingModel(), "features": ["traffic_lag_1"]},
        history,
        "cell_id",
        "timestamp",
        "traffic",
        horizon_steps=2,
        interval="1h",
        lags=[1],
        rolling_windows=[2],
    )

    assert result["prediction"].tolist() == [3.0, 4.0]
