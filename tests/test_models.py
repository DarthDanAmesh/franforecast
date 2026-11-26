import numpy as np
import pandas as pd
import pytest

from franforecast.errors import ModelError
from franforecast.models import (
    DEFAULT_MODELS,
    ETSForecaster,
    LightGBMForecaster,
    Naive,
    SeasonalNaive,
    SeriesHistory,
    make_model,
    minimum_history,
    model_catalog,
)


def history(y, uid="A", freq="D", start="2023-01-01"):
    y = np.asarray(y, dtype=float)
    dates = pd.date_range(start, periods=len(y), freq=freq)
    return SeriesHistory(uid, dates, y, freq)


def test_naive_repeats_the_last_value():
    model = Naive().fit(history([1.0, 2.0, 3.0]))
    forecast = model.predict(pd.date_range("2023-01-04", periods=3, freq="D"))
    assert forecast.tolist() == [3.0, 3.0, 3.0]


def test_seasonal_naive_tiles_the_last_season():
    model = SeasonalNaive().fit(history([1, 2, 3, 4, 5, 6, 7] * 3))
    forecast = model.predict(pd.date_range("2023-01-22", periods=8, freq="D"))
    assert forecast.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0]


def test_seasonal_naive_requires_one_full_season():
    with pytest.raises(ModelError, match="needs 7 observations"):
        SeasonalNaive().fit(history([1, 2, 3]))


def test_ets_requires_three_points():
    with pytest.raises(ModelError, match="at least 3 observations"):
        ETSForecaster().fit(history([1.0, 2.0]))


def test_ets_produces_finite_forecasts():
    n = 80
    y = 20 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + 0.1 * np.arange(n)
    model = ETSForecaster().fit(history(y))
    forecast = model.predict(pd.date_range("2023-03-22", periods=7, freq="D"))
    assert len(forecast) == 7
    assert np.isfinite(forecast).all()
    assert (forecast >= 0).all()


def test_lightgbm_learns_a_constant_series():
    model = LightGBMForecaster().fit(history(np.full(60, 10.0)))
    forecast = model.predict(pd.date_range("2023-03-02", periods=5, freq="D"))
    assert len(forecast) == 5
    np.testing.assert_allclose(forecast, 10.0, atol=1e-6)


def test_lightgbm_names_the_short_series():
    with pytest.raises(ModelError, match="SHORT"):
        LightGBMForecaster().fit(history(np.arange(20.0), uid="SHORT"))


def test_lightgbm_respects_non_negative_demand():
    y = 50 + 10 * np.sin(np.arange(90) / 3) + np.arange(90) * 0.2
    model = LightGBMForecaster().fit(history(y))
    forecast = model.predict(pd.date_range("2023-03-31", periods=8, freq="D"))
    assert (forecast >= 0).all()
    assert np.isfinite(forecast).all()


def test_catalog_and_registry():
    catalog = model_catalog()
    for name in DEFAULT_MODELS:
        assert name in catalog
    with pytest.raises(ModelError, match="Unknown model"):
        make_model("Prophet")


def test_minimum_history_matches_the_lightgbm_guard():
    assert minimum_history("LightGBM") == 58
    for name in ("Naive", "Seasonal naive", "ETS"):
        assert minimum_history(name) == 0
    # The guard in _fit uses the same numbers, so a series of this length fits.
    model = LightGBMForecaster().fit(history(np.zeros(58)))
    assert model.model is not None
