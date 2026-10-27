import pandas as pd
import pytest

from ranai.data.clean import clean_kpi, quality_report


def test_clean_kpi_flags_counter_reset_and_declared_outage_zero() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "cell_id": ["a", "a", "a"],
            "counter": [10, 2, 3],
            "traffic": [0.0, 5.0, 6.0],
            "outage_flag": ["false", "true", "false"],
        }
    )

    cleaned = clean_kpi(
        frame,
        "timestamp",
        "cell_id",
        counter_columns=["counter"],
        outage_flag_column="outage_flag",
        zero_is_outage_columns=["traffic"],
    )

    assert cleaned["counter_reset"].tolist() == [False, True, False]
    assert cleaned["outage"].tolist() == [True, True, False]
    assert pd.isna(cleaned.loc[0, "traffic"])
    assert pd.isna(cleaned.loc[1, "traffic"])
    assert cleaned.loc[2, "traffic"] == 6.0


def test_clean_kpi_rejects_duplicate_cell_timestamps() -> None:
    frame = pd.DataFrame({"timestamp": ["2025-01-01"] * 2, "cell_id": ["a", "a"]})

    with pytest.raises(ValueError, match="at most one row"):
        clean_kpi(frame, "timestamp", "cell_id")


def test_quality_report_counts_missing_intervals_and_bounds() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 02:00"]),
            "cell_id": ["a", "a"],
            "traffic": [1.0, 10.0],
        }
    )

    report = quality_report(
        frame,
        "timestamp",
        "cell_id",
        interval="1h",
        value_columns=["traffic"],
        outlier_bounds={"traffic": (0, 5)},
    )

    assert report.loc[0, "missing_intervals"] == 1
    assert report.loc[0, "traffic_outlier_count"] == 1
