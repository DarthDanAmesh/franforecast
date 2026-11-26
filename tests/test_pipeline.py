"""End-to-end: raw sample CSV -> panel -> backtest -> forecast."""

import numpy as np
import pytest

from franforecast.backtest import forecast_future, rolling_origin
from franforecast.data import to_panel
from franforecast.models import DEFAULT_MODELS


@pytest.fixture(scope="module")
def two_series_panel(sample_raw):
    panel, _ = to_panel(
        sample_raw,
        "Date",
        "Units_Sold",
        ("Item_Code", "Region"),
        "W-SUN",
        "zero",
    )
    keep = sorted(panel["unique_id"].unique())[:2]
    return panel[panel["unique_id"].isin(keep)].reset_index(drop=True)


def test_panel_is_contiguous_weekly(two_series_panel):
    for _, group in two_series_panel.groupby("unique_id"):
        gaps = group.sort_values("ds")["ds"].diff().dropna().unique()
        assert list(gaps) == [np.timedelta64(7, "D")]


def test_full_backtest_and_forecast(two_series_panel):
    outcome = rolling_origin(
        two_series_panel,
        list(DEFAULT_MODELS),
        horizon=4,
        folds=2,
        freq="W-SUN",
        min_train=104,
    )

    assert set(outcome.summary["Model"]) == set(DEFAULT_MODELS)
    assert outcome.summary["MASE"].notna().all()
    assert outcome.summary["Series"].eq(2).all()
    assert outcome.failures.empty
    assert outcome.best_model in DEFAULT_MODELS

    forecast = forecast_future(two_series_panel, two_series_panel["unique_id"].iloc[0], "ETS", 4, "W-SUN")
    assert len(forecast) == 4
    assert (forecast["y_pred"] >= 0).all()
