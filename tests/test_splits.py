import pandas as pd
import pytest

from ranai.validation.splits import cell_holdout_split, cell_time_split, temporal_split


def test_temporal_split_keeps_validation_after_training() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="h"),
            "value": range(10),
        }
    )

    train, validation = temporal_split(frame, "timestamp", validation_fraction=0.2)

    assert len(train) == 8
    assert len(validation) == 2
    assert train["timestamp"].max() < validation["timestamp"].min()


def test_temporal_split_keeps_equal_timestamps_together() -> None:
    timestamp = pd.Timestamp("2025-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": [
                timestamp,
                timestamp,
                timestamp + pd.Timedelta(hours=1),
                timestamp + pd.Timedelta(hours=1),
                timestamp + pd.Timedelta(hours=2),
            ],
            "cell_id": ["a", "b", "a", "b", "a"],
        }
    )

    train, validation = temporal_split(frame, "timestamp", validation_fraction=0.34)

    assert train["timestamp"].max() < validation["timestamp"].min()
    assert set(train["timestamp"]).isdisjoint(set(validation["timestamp"]))


def test_temporal_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        temporal_split(pd.DataFrame({"timestamp": [1, 2]}), "timestamp", 1.0)


def test_cell_holdout_split_has_no_entity_overlap() -> None:
    frame = pd.DataFrame({"cell_id": ["a", "b", "c"], "value": [1, 2, 3]})

    train, held_out = cell_holdout_split(frame, "cell_id", {"b"})

    assert set(train["cell_id"]).isdisjoint(set(held_out["cell_id"]))
    assert held_out["cell_id"].tolist() == ["b"]


def test_cell_time_split_excludes_held_out_cells_and_future_rows_from_training() -> None:
    timestamps = pd.date_range("2025-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps.repeat(2),
            "cell_id": ["a", "b"] * len(timestamps),
        }
    )

    train, validation = cell_time_split(
        frame, "cell_id", "timestamp", {"b"}, validation_fraction=0.2
    )

    assert "b" not in set(train["cell_id"])
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert set(validation["cell_id"]) == {"b"}
