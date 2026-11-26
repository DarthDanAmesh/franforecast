"""Rolling-origin backtesting: every fold trains only on data before its cutoff."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LABELS_BY_OFFSET, default_min_train, future_dates, season_length
from .data import guard_frequency
from .errors import DataError
from .metrics import evaluate
from .models import ModelError, SeriesHistory, family_of, make_model

FORECAST_COLUMNS = ["model", "unique_id", "fold", "ds", "y_true", "y_pred"]
FAILURE_COLUMNS = ["model", "unique_id", "fold", "stage", "detail"]
FOLD_COLUMNS = [
    "unique_id",
    "fold",
    "cutoff",
    "train_periods",
    "test_start",
    "test_end",
    "test_periods",
]


@dataclass
class BacktestOutcome:
    summary: pd.DataFrame
    per_series: pd.DataFrame
    forecasts: pd.DataFrame
    failures: pd.DataFrame
    folds: pd.DataFrame

    @property
    def best_model(self) -> str | None:
        valid = self.summary.dropna(subset=["MASE"])
        if valid.empty:
            return None
        return str(valid.sort_values("MASE").iloc[0]["Model"])


def rolling_origin(
    panel: pd.DataFrame,
    model_names: list[str],
    horizon: int,
    folds: int,
    freq: str,
    min_train: int | None = None,
) -> BacktestOutcome:
    if freq not in LABELS_BY_OFFSET:
        raise DataError(f"Unknown frequency '{freq}'.")
    if horizon < 1 or folds < 1:
        raise DataError("Horizon and fold count must both be at least 1.")

    required = {"unique_id", "ds", "y"}
    missing = required - set(panel.columns)
    if missing:
        raise DataError(f"Panel is missing columns: {sorted(missing)}.")
    if panel.empty:
        raise DataError("The panel is empty; nothing to backtest.")

    names = list(dict.fromkeys(model_names))
    if not names:
        raise DataError("Select at least one model.")

    guard_frequency(panel[["unique_id", "ds", "y"]], freq)

    season = season_length(freq)
    if min_train is None:
        min_train = default_min_train(freq)
    if min_train < 3:
        raise DataError("Minimum training periods must be at least 3.")

    forecast_rows: list[dict] = []
    failure_rows: list[dict] = []
    fold_rows: list[dict] = []

    for uid, group in panel.groupby("unique_id", sort=True):
        frame = group.sort_values("ds")
        dates = pd.DatetimeIndex(frame["ds"])
        y = frame["y"].to_numpy(dtype=float)
        n = len(y)
        needed = min_train + folds * horizon

        if n < needed:
            for name in names:
                failure_rows.append(
                    {
                        "model": name,
                        "unique_id": str(uid),
                        "fold": None,
                        "stage": "skipped",
                        "detail": (
                            f"Series has {n} periods; needs at least {needed} "
                            f"(min_train={min_train} + {folds} folds x horizon={horizon})."
                        ),
                    }
                )
            continue

        cutoffs = [n - horizon * (folds - k) for k in range(folds)]
        for fold_index, cutoff in enumerate(cutoffs):
            train_dates = dates[:cutoff]
            train_y = y[:cutoff]
            test_dates = dates[cutoff : cutoff + horizon]
            test_y = y[cutoff : cutoff + horizon]

            fold_rows.append(
                {
                    "unique_id": str(uid),
                    "fold": fold_index + 1,
                    "cutoff": train_dates[-1],
                    "train_periods": cutoff,
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                    "test_periods": len(test_dates),
                }
            )

            for name in names:
                try:
                    model = make_model(name)
                    model.fit(SeriesHistory(str(uid), train_dates, train_y, freq))
                    predictions = np.asarray(model.predict(test_dates), dtype=float).reshape(-1)
                    if len(predictions) != horizon:
                        raise ModelError(
                            f"Returned {len(predictions)} predictions for a horizon of {horizon}."
                        )
                    invalid = ~np.isfinite(predictions)
                    if invalid.any():
                        raise ModelError(
                            f"{int(invalid.sum())} of {horizon} predictions were non-finite."
                        )
                except Exception as exc:  # noqa: BLE001 - one failure must not abort the run
                    failure_rows.append(
                        {
                            "model": name,
                            "unique_id": str(uid),
                            "fold": fold_index + 1,
                            "stage": "prediction",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                for target_date, actual, predicted in zip(test_dates, test_y, predictions):
                    forecast_rows.append(
                        {
                            "model": name,
                            "unique_id": str(uid),
                            "fold": fold_index + 1,
                            "ds": target_date,
                            "y_true": float(actual),
                            "y_pred": float(predicted),
                        }
                    )

    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS)
    failures = pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)
    fold_table = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)

    per_series = _per_series_metrics(panel, forecasts, folds, horizon, season)
    summary = _summary(per_series, failures, names)

    return BacktestOutcome(
        summary=summary,
        per_series=per_series,
        forecasts=forecasts,
        failures=failures,
        folds=fold_table,
    )


def _per_series_metrics(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    folds: int,
    horizon: int,
    season: int,
) -> pd.DataFrame:
    columns = [
        "model",
        "family",
        "unique_id",
        "fold_count",
        "n_points",
        "zeros_actual",
        "mape_points",
        "mase",
        "rmse",
        "mae",
        "mape",
        "smape",
        "bias",
    ]
    if forecasts.empty:
        return pd.DataFrame(columns=columns)

    scale_windows: dict[str, np.ndarray] = {}
    for uid, group in panel.groupby("unique_id", sort=True):
        y = group.sort_values("ds")["y"].to_numpy(dtype=float)
        initial_cutoff = len(y) - folds * horizon
        scale_windows[str(uid)] = y[:initial_cutoff]

    rows = []
    for (model, uid), group in forecasts.groupby(["model", "unique_id"], sort=True):
        metrics = evaluate(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
            y_train=scale_windows.get(str(uid)),
            season=season,
        )
        rows.append(
            {
                "model": model,
                "family": family_of(model),
                "unique_id": uid,
                "fold_count": int(group["fold"].nunique()),
                "n_points": int(metrics["n_points"]),
                "zeros_actual": int(metrics["zeros_actual"]),
                "mape_points": int(metrics["mape_points"]),
                "mase": metrics["mase"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "smape": metrics["smape"],
                "bias": metrics["bias"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _summary(per_series: pd.DataFrame, failures: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    columns = [
        "Rank",
        "Model",
        "Family",
        "MASE",
        "RMSE",
        "MAE",
        "MAPE",
        "SMAPE",
        "Bias",
        "Series",
        "Points",
        "Failed",
    ]
    if per_series.empty:
        summary = pd.DataFrame({"Model": names})
        for column in ["Series", "Points", "MASE", "RMSE", "MAE", "MAPE", "SMAPE", "Bias"]:
            summary[column] = np.nan
    else:
        summary = (
            per_series.groupby("model", as_index=False)
            .agg(
                Series=("unique_id", "nunique"),
                Points=("n_points", "sum"),
                MASE=("mase", "mean"),
                RMSE=("rmse", "mean"),
                MAE=("mae", "mean"),
                MAPE=("mape", "mean"),
                SMAPE=("smape", "mean"),
                Bias=("bias", "mean"),
            )
            .rename(columns={"model": "Model"})
            .set_index("Model")
            .reindex(names)
            .reset_index()
        )

    summary["Family"] = summary["Model"].map(family_of)

    if not failures.empty:
        failed = failures.groupby("model")["unique_id"].nunique()
        summary["Failed"] = summary["Model"].map(lambda m: int(failed.get(m, 0)))
    else:
        summary["Failed"] = 0

    summary["Series"] = summary["Series"].fillna(0).astype(int)
    summary["Points"] = summary["Points"].fillna(0).astype(int)
    summary["Failed"] = summary["Failed"].fillna(0).astype(int)

    summary = summary.sort_values(["MASE", "Model"], na_position="last").reset_index(drop=True)
    summary["Rank"] = pd.array(
        [str(i + 1) if pd.notna(value) else None for i, value in enumerate(summary["MASE"])],
        dtype="string",
    )

    return summary[columns]


def forecast_future(
    panel: pd.DataFrame,
    unique_id: str,
    model_name: str,
    horizon: int,
    freq: str,
) -> pd.DataFrame:
    """Fit one model on the full series and predict the next `horizon` periods."""
    if horizon < 1:
        raise DataError("Horizon must be at least 1.")
    group = panel.loc[panel["unique_id"] == unique_id].sort_values("ds")
    if group.empty:
        raise DataError(f"Series '{unique_id}' is not in the panel.")
    y = group["y"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(group["ds"])
    model = make_model(model_name)
    model.fit(SeriesHistory(unique_id, dates, y, freq))
    upcoming = future_dates(dates[-1], horizon, freq)
    predictions = np.asarray(model.predict(upcoming), dtype=float).reshape(-1)
    return pd.DataFrame({"ds": upcoming, "y_pred": predictions})
