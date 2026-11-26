from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franforecast.data import to_panel  # noqa: E402
from franforecast.synthetic import generate_sales  # noqa: E402


@pytest.fixture(scope="session")
def sample_raw() -> pd.DataFrame:
    return generate_sales()


@pytest.fixture(scope="session")
def weekly_panel(sample_raw: pd.DataFrame) -> pd.DataFrame:
    panel, _ = to_panel(
        sample_raw,
        "Date",
        "Units_Sold",
        ("Item_Code", "Region"),
        "W-SUN",
        "zero",
    )
    return panel


@pytest.fixture()
def daily_panel() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    return pd.DataFrame(
        {
            "unique_id": "A",
            "ds": dates,
            "y": 10.0 + (pd.Series(range(90)) % 7).to_numpy(dtype=float),
        }
    )
