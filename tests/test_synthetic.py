import pandas as pd
import pytest

from franforecast.synthetic import generate_sales


def test_sample_is_reproducible():
    first = generate_sales(seed=11)
    second = generate_sales(seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_sample_shape_and_columns():
    frame = generate_sales()
    assert list(frame.columns) == ["Date", "Item_Code", "Region", "Units_Sold"]
    assert len(frame) == 1461 * 8
    assert frame["Date"].min() == pd.Timestamp("2021-01-01")
    assert frame["Date"].max() == pd.Timestamp("2024-12-31")
    assert frame.groupby(["Item_Code", "Region"]).ngroups == 8


def test_sample_has_no_negative_demand_and_includes_zeros():
    frame = generate_sales()
    assert (frame["Units_Sold"] >= 0).all()
    assert (frame["Units_Sold"] == 0).any()


def test_sample_seed_changes_values():
    a = generate_sales(seed=1)
    b = generate_sales(seed=2)
    assert not a["Units_Sold"].equals(b["Units_Sold"])
