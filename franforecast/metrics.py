"""Forecast accuracy metrics with explicit zero and coverage handling."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _pairs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(true) != len(pred):
        raise ValueError(f"Length mismatch: {len(true)} actual values vs {len(pred)} predictions.")
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def mae(y_true, y_pred) -> float:
    true, pred = _pairs(y_true, y_pred)
    return float(np.mean(np.abs(pred - true))) if len(true) else float("nan")


def rmse(y_true, y_pred) -> float:
    true, pred = _pairs(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - true) ** 2))) if len(true) else float("nan")


def bias(y_true, y_pred) -> float:
    true, pred = _pairs(y_true, y_pred)
    return float(np.mean(pred - true)) if len(true) else float("nan")


def evaluate(
    y_true,
    y_pred,
    y_train=None,
    season: int | None = None,
) -> dict[str, float]:
    """All metrics for one series. Percentage errors skip zero actuals.

    MASE divides the MAE by the in-sample seasonal-naive MAE computed on
    `y_train`, so it is comparable across series of different magnitudes.
    """
    true, pred = _pairs(y_true, y_pred)
    result: dict[str, float] = {
        "n_points": float(len(true)),
        "zeros_actual": float(np.sum(np.abs(true) <= EPS)),
        "mape_points": 0.0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "mape": float("nan"),
        "smape": float("nan"),
        "mase": float("nan"),
        "bias": float("nan"),
    }
    if len(true) == 0:
        return result

    error = pred - true
    result["mae"] = float(np.mean(np.abs(error)))
    result["rmse"] = float(np.sqrt(np.mean(error**2)))
    result["bias"] = float(np.mean(error))

    nonzero = np.abs(true) > EPS
    result["mape_points"] = float(nonzero.sum())
    if nonzero.any():
        result["mape"] = float(100.0 * np.mean(np.abs(error[nonzero] / true[nonzero])))

    denominator = np.abs(true) + np.abs(pred)
    valid_smape = denominator > EPS
    perfect_zeros = (~valid_smape) & (np.abs(error) <= EPS)
    smape_terms = int(valid_smape.sum())
    if smape_terms:
        result["smape"] = float(
            200.0 * np.sum(np.abs(error[valid_smape]) / denominator[valid_smape]) / smape_terms
        )
    elif perfect_zeros.any():
        result["smape"] = 0.0

    if y_train is not None:
        train = np.asarray(y_train, dtype=float).reshape(-1)
        train = train[np.isfinite(train)]
        if len(train) > 1:
            if season and len(train) > season:
                scale = float(np.mean(np.abs(train[season:] - train[:-season])))
            else:
                scale = float(np.mean(np.abs(np.diff(train))))
            if scale > EPS:
                result["mase"] = result["mae"] / scale

    return result
