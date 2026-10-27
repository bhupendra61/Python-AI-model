import pandas as pd

from ranai.features.build import build_features
from ranai.features.neighbor import add_neighbor_features, add_site_features


def test_lag_and_rolling_features_use_only_prior_observations() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a"] * 4,
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h"),
            "traffic": [1.0, 2.0, 1000.0, 4.0],
        }
    )

    features = build_features(
        frame,
        "timestamp",
        "cell_id",
        ["traffic"],
        lags=[1],
        rolling_windows=[2],
        rolling_statistics=["mean"],
    )

    assert features.loc[2, "traffic_lag_1"] == 2.0
    assert features.loc[2, "traffic_rolling_mean_2"] == 1.5
    assert features.loc[3, "traffic_rolling_mean_2"] == 501.0
    modified_future = frame.copy()
    modified_future.loc[3, "traffic"] = 1_000_000.0
    changed = build_features(
        modified_future,
        "timestamp",
        "cell_id",
        ["traffic"],
        lags=[1],
        rolling_windows=[2],
        rolling_statistics=["mean"],
    )
    pd.testing.assert_series_equal(
        features.loc[:2, "traffic_rolling_mean_2"],
        changed.loc[:2, "traffic_rolling_mean_2"],
    )


def test_lag_uses_exact_interval_when_observations_are_missing() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "a", "a"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 03:00"]
            ),
            "traffic": [1.0, 2.0, 3.0],
        }
    )

    features = build_features(
        frame,
        "timestamp",
        "cell_id",
        ["traffic"],
        lags=[1],
        rolling_windows=[2],
        rolling_statistics=["mean"],
        interval="1h",
    )

    assert pd.isna(features.loc[2, "traffic_lag_1"])
    assert features.loc[2, "traffic_rolling_mean_2"] == 2.0


def test_neighbor_features_exclude_self_and_align_by_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "timestamp": pd.to_datetime(["2025-01-01"] * 3),
            "traffic_lag_1": [10.0, 20.0, 40.0],
            "site_id": ["x", "x", "x"],
        }
    )

    neighbors = add_neighbor_features(
        frame,
        "cell_id",
        "timestamp",
        ["traffic_lag_1"],
        {"a": ["a", "b"], "b": ["a"], "c": ["unknown"]},
    )
    sites = add_site_features(frame, "cell_id", "timestamp", "site_id", ["traffic_lag_1"])

    assert neighbors.loc[0, "traffic_lag_1_neighbor_mean"] == 20.0
    assert pd.isna(neighbors.loc[2, "traffic_lag_1_neighbor_mean"])
    assert sites["traffic_lag_1_same_site_mean"].tolist() == [30.0, 25.0, 15.0]
