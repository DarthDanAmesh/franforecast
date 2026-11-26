"""Synthetic Franke-style sales data for development and demos."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

ITEMS = (
    "A600 Coffee System",
    "A800 Coffee System",
    "Cooling Unit CU4",
    "Waste Station WS2",
)
REGIONS = ("North", "South")


def generate_sales(seed: int = 7, start: str = "2021-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Daily unit sales per item and region with trend, seasonality and holiday effects."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    if len(dates) < 400:
        raise ValueError("Use at least 400 days so the sample has several seasonal cycles.")

    holiday_calendar = holidays.Germany(years=sorted({d.year for d in dates}))
    holiday_dates = pd.DatetimeIndex(sorted(holiday_calendar.keys()))

    t = np.arange(len(dates))
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()
    is_holiday = dates.isin(holiday_dates)

    frames = []
    for item in ITEMS:
        for region in REGIONS:
            base = rng.uniform(45, 150)
            growth = rng.uniform(0.02, 0.16)
            weekend_factor = rng.uniform(0.15, 0.35)
            annual_amplitude = rng.uniform(0.10, 0.25)
            phase = rng.uniform(0, 2 * np.pi)

            trend = (1 + growth) ** (t / 365.25)
            weekly = np.where(dow >= 5, weekend_factor, 1.0)
            seasonal = 1 + annual_amplitude * np.cos(2 * np.pi * (doy - 340) / 365.25 + phase * 0.05)
            december = np.where(month == 12, 1.18, 1.0)
            holiday_effect = np.where(is_holiday, 0.18, 1.0)

            promo = np.ones(len(dates))
            for promo_start in rng.integers(30, len(dates) - 10, 8):
                promo[promo_start : promo_start + int(rng.integers(2, 5))] *= rng.uniform(1.6, 2.6)

            stockout = rng.random(len(dates)) < 0.004

            level = base * trend * weekly * seasonal * december * holiday_effect * promo
            units = rng.lognormal(mean=np.log(level), sigma=0.16)
            units = np.where(stockout, 0, units)
            units = np.clip(np.round(units), 0, None).astype(int)

            frames.append(
                pd.DataFrame(
                    {
                        "Date": dates,
                        "Item_Code": item,
                        "Region": region,
                        "Units_Sold": units,
                    }
                )
            )

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Date", "Item_Code", "Region"]).reset_index(drop=True)
