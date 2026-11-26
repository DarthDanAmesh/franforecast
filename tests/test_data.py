import io

import numpy as np
import pandas as pd
import pytest

from franforecast.data import (
    DataError,
    guess_date_column,
    guess_group_columns,
    guess_target_column,
    read_table,
    to_panel,
)


def frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_target_column_named_target_is_kept():
    df = frame(
        Date=pd.date_range("2024-01-01", periods=10, freq="D"),
        target=np.arange(10.0),
        Store=["A"] * 5 + ["B"] * 5,
    )
    panel, report = to_panel(df, "Date", "target", ("Store",), "D", "zero")

    assert list(panel.columns) == ["unique_id", "ds", "y"]
    assert len(panel) == 10
    assert report.series_count == 2
    assert panel["y"].sum() == pytest.approx(45.0)


def test_duplicate_series_periods_are_summed():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = frame(
        Date=list(dates) + [dates[0], dates[0]],
        Units=[1.0, 2.0, 3.0, 4.0, 10.0, 100.0],
        Store=["A"] * 6,
    )
    panel, report = to_panel(df, "Date", "Units", ("Store",), "D", "zero")

    assert report.merged_duplicate_rows == 2
    assert panel.loc[panel["ds"] == dates[0], "y"].iloc[0] == pytest.approx(111.0)


def test_invalid_dates_are_dropped_and_named():
    df = frame(
        Date=["2024-01-01", "not-a-date", "2024-01-03"],
        Units=[1.0, 2.0, 3.0],
    )
    panel, report = to_panel(df, "Date", "Units", (), "D", "zero")

    assert report.dropped_invalid_dates == 1
    assert len(panel) == 3  # the removed day is completed as a zero-valued gap
    assert panel["y"].sum() == pytest.approx(4.0)
    assert any("not-a-date" in note for note in report.notes)


def test_non_numeric_targets_are_dropped_and_named():
    df = frame(
        Date=pd.date_range("2024-01-01", periods=3, freq="D"),
        Units=["1", "oops", "3"],
    )
    panel, report = to_panel(df, "Date", "Units", (), "D", "zero")

    assert report.dropped_invalid_targets == 1
    assert len(panel) == 3  # the removed day is completed as a zero-valued gap
    assert panel["y"].sum() == pytest.approx(4.0)
    assert any("oops" in note for note in report.notes)


def test_missing_column_error_names_the_column():
    df = frame(Date=pd.date_range("2024-01-01", periods=3, freq="D"), Units=[1, 2, 3])
    with pytest.raises(DataError, match="Nope"):
        to_panel(df, "Date", "Nope", (), "D", "zero")


def test_upsampling_is_rejected():
    df = frame(
        Date=pd.date_range("2024-01-01", periods=6, freq="MS"),
        Units=np.arange(6.0),
    )
    with pytest.raises(DataError, match="coarser"):
        to_panel(df, "Date", "Units", (), "D", "zero")


def test_gap_strategies():
    df = frame(Date=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]), Units=[0.0, 20.0, 40.0])

    zeros, report_zero = to_panel(df, "Date", "Units", (), "D", "zero")
    assert len(zeros) == 5
    assert report_zero.gaps_filled == 2
    assert zeros["y"].tolist() == [0.0, 0.0, 20.0, 0.0, 40.0]

    interpolated, _ = to_panel(df, "Date", "Units", (), "D", "interpolate")
    assert interpolated["y"].tolist() == pytest.approx([0.0, 10.0, 20.0, 30.0, 40.0])

    dropped, report_drop = to_panel(df, "Date", "Units", (), "D", "drop")
    assert len(dropped) == 3
    assert report_drop.gaps_dropped == 2


def test_partial_week_edges_are_dropped():
    dates = pd.date_range("2024-01-03", "2024-02-14", freq="D")
    df = frame(Date=dates, Units=np.arange(len(dates), dtype=float))
    panel, report = to_panel(df, "Date", "Units", (), "W-SUN", "zero")

    assert report.dropped_partial_periods == 2
    assert panel["ds"].min() == pd.Timestamp("2024-01-14")
    assert panel["ds"].max() == pd.Timestamp("2024-02-11")


def test_missing_group_values_are_labelled():
    df = frame(
        Date=pd.date_range("2024-01-01", periods=4, freq="D"),
        Units=[1.0, 2.0, 3.0, 4.0],
        Store=["A", None, "A", None],
    )
    panel, _ = to_panel(df, "Date", "Units", ("Store",), "D", "zero")
    assert set(panel["unique_id"]) == {"A", "(missing)"}


def test_guess_helpers_on_franke_columns(sample_raw):
    assert guess_date_column(sample_raw) == "Date"
    assert guess_target_column(sample_raw) == "Units_Sold"
    assert guess_group_columns(sample_raw, "Date", "Units_Sold") == ["Item_Code", "Region"]


def test_read_table_trims_names_and_rejects_duplicates():
    good = read_table(io.StringIO(" Date , Units \n2024-01-01,2\n"))
    assert list(good.columns) == ["Date", "Units"]

    with pytest.raises(DataError, match="Duplicate"):
        read_table(io.StringIO("a, a \n1,2\n"))

    with pytest.raises(DataError, match="no rows"):
        read_table(io.StringIO("a,b\n"))
