import numpy as np
import pandas as pd
import pytest

from franforecast.backtest import forecast_future, rolling_origin
from franforecast.errors import DataError


def panel_from(y, start="2023-01-01", uid="A", freq="D"):
    dates = pd.date_range(start, periods=len(y), freq=freq)
    return pd.DataFrame({"unique_id": uid, "ds": dates, "y": np.asarray(y, dtype=float)})


def seasonal_series(n=90):
    return 10.0 + (np.arange(n) % 7)


def test_fold_boundaries_and_forecast_counts(daily_panel):
    outcome = rolling_origin(daily_panel, ["Naive", "Seasonal naive"], 7, 2, "D", 14)

    assert len(outcome.folds) == 2
    assert outcome.folds["cutoff"].tolist() == [
        pd.Timestamp("2023-01-01") + pd.Timedelta(days=75),
        pd.Timestamp("2023-01-01") + pd.Timedelta(days=82),
    ]
    assert outcome.folds["train_periods"].tolist() == [76, 83]
    assert len(outcome.forecasts) == 2 * 2 * 7
    assert set(outcome.summary["Model"]) == {"Naive", "Seasonal naive"}
    assert outcome.failures.empty


def test_perfectly_seasonal_data_is_forecast_exactly(daily_panel):
    outcome = rolling_origin(daily_panel, ["Naive", "Seasonal naive"], 7, 2, "D", 14)

    seasonal = outcome.per_series[outcome.per_series["model"] == "Seasonal naive"].iloc[0]
    naive = outcome.per_series[outcome.per_series["model"] == "Naive"].iloc[0]

    assert seasonal["mae"] == pytest.approx(0.0)
    assert np.isnan(seasonal["mase"])  # scaling error is zero, so MASE is undefined
    assert naive["mae"] > 0


def test_future_values_cannot_change_past_fold_forecasts(daily_panel):
    original = rolling_origin(daily_panel, ["Naive"], 7, 2, "D", 14)

    perturbed_panel = daily_panel.copy()
    perturbed_panel.loc[perturbed_panel.index[76:], "y"] = 99999.0
    perturbed = rolling_origin(perturbed_panel, ["Naive"], 7, 2, "D", 14)

    first_fold = original.forecasts[original.forecasts["fold"] == 1].sort_values("ds")
    perturbed_first = perturbed.forecasts[perturbed.forecasts["fold"] == 1].sort_values("ds")
    np.testing.assert_allclose(first_fold["y_pred"], perturbed_first["y_pred"])


def test_short_series_are_reported_not_silently_dropped():
    short = panel_from(np.arange(20.0), uid="Too short")
    outcome = rolling_origin(short, ["Naive", "Seasonal naive"], 7, 2, "D", 14)

    assert len(outcome.failures) == 2
    assert set(outcome.failures["unique_id"]) == {"Too short"}
    assert outcome.failures["stage"].eq("skipped").all()
    assert all("needs at least" in detail for detail in outcome.failures["detail"])
    assert outcome.summary["Series"].eq(0).all()
    assert outcome.summary["Failed"].eq(1).all()
    assert outcome.summary["MASE"].isna().all()


def test_bad_configuration_raises():
    panel = panel_from(np.arange(40.0))
    with pytest.raises(DataError, match="at least 1"):
        rolling_origin(panel, ["Naive"], 0, 2, "D", 10)
    with pytest.raises(DataError, match="Select at least one model"):
        rolling_origin(panel, [], 7, 2, "D", 10)


def test_forecast_future_returns_the_next_periods():
    panel = panel_from(seasonal_series(90))
    forecast = forecast_future(panel, "A", "Seasonal naive", 5, "D")

    assert list(forecast.columns) == ["ds", "y_pred"]
    assert len(forecast) == 5
    last_actual = panel["ds"].max()
    assert forecast["ds"].iloc[0] == last_actual + pd.Timedelta(days=1)
    assert np.isfinite(forecast["y_pred"]).all()


def test_forecast_future_unknown_series():
    panel = panel_from(seasonal_series(30))
    with pytest.raises(DataError, match="not in the panel"):
        forecast_future(panel, "B", "Naive", 3, "D")
