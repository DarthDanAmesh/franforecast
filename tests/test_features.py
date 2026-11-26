import numpy as np
import pandas as pd
import pytest

from franforecast.errors import ModelError
from franforecast.features import FeatureBuilder


def make_history(n: int = 60):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    y = 10.0 + np.arange(n, dtype=float)
    return dates, y


def test_training_matrix_shape_and_no_nan():
    dates, y = make_history(60)
    builder = FeatureBuilder()
    features, target, used_dates = builder.training_matrix(dates, y)

    assert features.shape[1] == len(builder.names)
    assert not np.isnan(features).any()
    assert len(target) == len(used_dates) == len(dates) - builder.warmup


def test_features_do_not_use_the_current_target():
    dates, y = make_history(60)
    builder = FeatureBuilder()
    before, _, _ = builder.training_matrix(dates, y)

    changed = y.copy()
    changed[-1] = 1_000_000.0
    after, _, _ = builder.training_matrix(dates, changed)

    # The last row's features are built from history before that period, so they
    # must be unchanged even though its target changed.
    np.testing.assert_allclose(before[-1], after[-1])


def test_forecast_is_recursive_and_non_negative():
    dates, y = make_history(60)
    builder = FeatureBuilder()
    builder.training_matrix(dates, y)
    builder.clip_non_negative = True

    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7, freq="D")
    predictions = builder.forecast(lambda row: float(row[0]) - 5.0, dates, y, future)

    assert len(predictions) == 7
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_too_short_history_raises_with_numbers():
    dates, y = make_history(10)
    builder = FeatureBuilder()
    with pytest.raises(ModelError, match="at least 29 observations"):
        builder.training_matrix(dates, y)
