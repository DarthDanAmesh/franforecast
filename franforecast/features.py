"""Lag and calendar features for supervised demand models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import ModelError

DEFAULT_LAGS = (1, 2, 3, 7, 14, 28)
DEFAULT_WINDOWS = (7, 28)
CALENDAR_NAMES = (
    "dow_sin",
    "dow_cos",
    "annual_sin",
    "annual_cos",
    "age_days",
    "is_weekend",
    "is_holiday",
    "is_december",
)


class FeatureBuilder:
    """Builds the same feature row for training and for recursive forecasting.

    Every predictor is computed from values strictly before the target period,
    so a feature row never sees its own outcome.
    """

    def __init__(
        self,
        lags: tuple[int, ...] = DEFAULT_LAGS,
        windows: tuple[int, ...] = DEFAULT_WINDOWS,
        country: str = "DE",
    ) -> None:
        self.lags = tuple(sorted({int(v) for v in lags}))
        self.windows = tuple(sorted({int(v) for v in windows}))
        self.country = country
        self.t0: pd.Timestamp | None = None
        self._holidays: set[pd.Timestamp] = set()
        self._holiday_years: set[int] = set()
        self.clip_non_negative = False

    @property
    def names(self) -> list[str]:
        names = [f"lag_{lag}" for lag in self.lags]
        for window in self.windows:
            names.extend([f"roll_mean_{window}", f"roll_std_{window}"])
        names.extend(CALENDAR_NAMES)
        return names

    @property
    def warmup(self) -> int:
        return max(max(self.lags), max(self.windows))

    def training_matrix(
        self, dates: pd.DatetimeIndex, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        if len(dates) != len(y):
            raise ModelError("Feature construction received mismatched date and target lengths.")
        if len(y) < self.warmup + 1:
            raise ModelError(
                f"Need at least {self.warmup + 1} observations to compute lag features, got {len(y)}."
            )

        self.t0 = pd.Timestamp(dates[0]).normalize()
        series = pd.Series(np.asarray(y, dtype=float), index=pd.DatetimeIndex(dates))

        columns: dict[str, pd.Series] = {}
        for lag in self.lags:
            columns[f"lag_{lag}"] = series.shift(lag)
        shifted = series.shift(1)
        for window in self.windows:
            columns[f"roll_mean_{window}"] = shifted.rolling(window).mean()
            columns[f"roll_std_{window}"] = shifted.rolling(window).std(ddof=0)

        calendar = np.array(
            [self._calendar_row(pd.Timestamp(d)) for d in dates], dtype=float
        )
        frame = pd.DataFrame(columns, index=series.index)
        for position, name in enumerate(CALENDAR_NAMES):
            frame[name] = calendar[:, position]

        frame = frame[self.names]
        valid = frame.notna().all(axis=1) & series.notna()
        return (
            frame.loc[valid].to_numpy(dtype=float),
            series.loc[valid].to_numpy(dtype=float),
            pd.DatetimeIndex(series.index[valid]),
        )

    def forecast(
        self,
        predict_one,
        dates: pd.DatetimeIndex,
        y: np.ndarray,
        future_dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        if self.t0 is None:
            raise ModelError("Features were not initialised; call training_matrix before forecasting.")
        history = [float(v) for v in np.asarray(y, dtype=float)]
        if len(history) < self.warmup + 1:
            raise ModelError(
                f"Need at least {self.warmup + 1} observations to forecast, got {len(history)}."
            )
        if not np.isfinite(history).all():
            raise ModelError("History contains non-finite values; clean the series first.")

        output = []
        for target in future_dates:
            row = self._feature_row(history, pd.Timestamp(target))
            prediction = float(predict_one(row))
            if not np.isfinite(prediction):
                raise ModelError(f"Model produced a non-finite prediction for {target.date()}.")
            if self.clip_non_negative:
                prediction = max(0.0, prediction)
            output.append(prediction)
            history.append(prediction)
        return np.asarray(output, dtype=float)

    def _feature_row(self, history: list[float], target: pd.Timestamp) -> np.ndarray:
        values: list[float] = []
        for lag in self.lags:
            values.append(history[-lag])
        array = np.asarray(history, dtype=float)
        for window in self.windows:
            tail = array[-window:]
            values.append(float(np.mean(tail)))
            values.append(float(np.std(tail)))
        values.extend(self._calendar_row(target))
        return np.asarray(values, dtype=float)

    def _calendar_row(self, date: pd.Timestamp) -> list[float]:
        weekday = date.dayofweek
        day_of_year = date.dayofyear
        return [
            float(np.sin(2 * np.pi * weekday / 7)),
            float(np.cos(2 * np.pi * weekday / 7)),
            float(np.sin(2 * np.pi * day_of_year / 365.25)),
            float(np.cos(2 * np.pi * day_of_year / 365.25)),
            float((date.normalize() - self.t0).days),
            1.0 if weekday >= 5 else 0.0,
            1.0 if self._is_holiday(date) else 0.0,
            1.0 if date.month == 12 else 0.0,
        ]

    def _is_holiday(self, date: pd.Timestamp) -> bool:
        year = date.year
        if year not in self._holiday_years:
            self._holiday_years.add(year)
            try:
                import holidays

                self._holidays.update(
                    pd.Timestamp(day) for day in holidays.country_holidays(self.country, years=year)
                )
            except Exception:  # holiday lookup is optional enrichment
                pass
        return date.normalize() in self._holidays
