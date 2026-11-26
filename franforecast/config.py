"""Shared constants for the forecasting pipeline."""

from __future__ import annotations

import pandas as pd

FREQUENCY_LABELS: dict[str, str] = {
    "Daily": "D",
    "Weekly": "W-SUN",
    "Monthly": "MS",
}
LABELS_BY_OFFSET: dict[str, str] = {v: k for k, v in FREQUENCY_LABELS.items()}

SEASON_LENGTH: dict[str, int] = {"D": 7, "W-SUN": 52, "MS": 12}

DEFAULT_HORIZON: dict[str, int] = {"D": 14, "W-SUN": 8, "MS": 6}
MAX_HORIZON: dict[str, int] = {"D": 60, "W-SUN": 26, "MS": 12}

GAP_STRATEGIES: dict[str, str] = {
    "Fill with zero": "zero",
    "Interpolate": "interpolate",
    "Drop the periods": "drop",
}

DEFAULT_FOLDS = 3
HOLIDAY_COUNTRY = "DE"


def season_length(freq: str) -> int:
    return SEASON_LENGTH[freq]


def default_min_train(freq: str) -> int:
    return max(2 * SEASON_LENGTH[freq], 21)


def future_dates(last_date: pd.Timestamp, horizon: int, freq: str) -> pd.DatetimeIndex:
    """The `horizon` dates that follow `last_date` at `freq`."""
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
