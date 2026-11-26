import math

import numpy as np
import pytest

from franforecast.metrics import evaluate, mae, rmse


def test_hand_computed_metrics():
    true = [100.0, 200.0, 300.0]
    pred = [110.0, 190.0, 330.0]
    train = [90.0, 180.0, 290.0]

    result = evaluate(true, pred, y_train=train, season=1)

    assert result["n_points"] == 3
    assert mae(true, pred) == pytest.approx(50.0 / 3.0)
    assert result["rmse"] == pytest.approx(math.sqrt(1100.0 / 3.0))
    assert result["bias"] == pytest.approx(10.0)
    assert result["mape"] == pytest.approx(100.0 * (0.1 + 0.05 + 0.1) / 3.0)
    expected_smape = 200.0 * (10 / 210 + 10 / 390 + 30 / 630) / 3.0
    assert result["smape"] == pytest.approx(expected_smape)
    assert result["mase"] == pytest.approx((50.0 / 3.0) / 100.0)


def test_zero_actuals_are_excluded_from_mape_but_kept_elsewhere():
    result = evaluate([0.0, 10.0], [5.0, 10.0], y_train=[0.0, 5.0, 10.0], season=1)

    assert result["zeros_actual"] == 1
    assert result["mape_points"] == 1
    assert result["mape"] == pytest.approx(0.0)
    assert result["smape"] == pytest.approx(100.0)
    assert result["mae"] == pytest.approx(2.5)


def test_all_zero_actuals_give_nan_mape_not_infinity():
    result = evaluate([0.0, 0.0], [0.0, 0.0])
    assert np.isnan(result["mape"])
    assert result["smape"] == pytest.approx(0.0)
    assert np.isnan(result["mase"])


def test_non_finite_pairs_are_dropped():
    result = evaluate([1.0, np.nan, 3.0], [1.0, 2.0, np.nan])
    assert result["n_points"] == 1
    assert result["mae"] == pytest.approx(0.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate([1, 2, 3], [1, 2])


def test_mase_falls_back_without_a_full_season():
    # 3 training points cannot supply a seasonal lag, so the plain difference is used.
    result = evaluate([5.0], [6.0], y_train=[1.0, 3.0, 8.0], season=12)
    assert result["mase"] == pytest.approx(1.0 / 3.5)


def test_empty_input_returns_nan_metrics():
    result = evaluate([], [])
    assert result["n_points"] == 0
    assert np.isnan(result["rmse"])
