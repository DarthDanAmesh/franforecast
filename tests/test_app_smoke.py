from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_and_flags_short_monthly_series():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Backtest", "Forecast", "Data & quality"]

    frequency = next(box for box in at.sidebar.selectbox if box.label == "Frequency")
    frequency.set_value("Monthly")
    at.run()

    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "LightGBM needs 76 periods per series" in warnings
    assert "A600 Coffee System / North (48 periods)" in warnings
