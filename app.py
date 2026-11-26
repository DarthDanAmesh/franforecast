"""Franke Forecast Studio: backtest and project demand series."""

from __future__ import annotations

import hashlib
import io
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from franforecast.backtest import BacktestOutcome, forecast_future, rolling_origin
from franforecast.config import (
    DEFAULT_FOLDS,
    DEFAULT_HORIZON,
    FREQUENCY_LABELS,
    GAP_STRATEGIES,
    MAX_HORIZON,
    default_min_train,
)
from franforecast.data import (
    PrepReport,
    guess_date_column,
    guess_group_columns,
    guess_target_column,
    read_table,
    series_stats,
    to_panel,
)
from franforecast.errors import DataError
from franforecast.models import (
    DEFAULT_MODELS,
    chronos_hint,
    is_chronos_available,
    minimum_history,
    model_catalog,
)
from franforecast.plotting import backtest_chart, history_chart, mase_heatmap

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_franke_sales.csv"

st.set_page_config(page_title="Franke Forecast Studio", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] * {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #FAFAFB; border-right: 1px solid #E8EBF0; }
    h1 { letter-spacing: -0.03em; font-weight: 700; }
    .fk-kicker {
        color: #B4232A; text-transform: uppercase; letter-spacing: .16em;
        font-size: .72rem; font-weight: 600; margin-bottom: .15rem;
    }
    .fk-sub { color: #6B7280; max-width: 780px; font-size: .95rem; line-height: 1.55; margin-bottom: 1.4rem; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
    [data-testid="stMetricLabel"] { color: #6B7280; }
    div[data-testid="stDataFrame"] { border: 1px solid #E8EBF0; border-radius: 6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def read_upload(payload: bytes) -> pd.DataFrame:
    return read_table(io.BytesIO(payload))


def frame_signature(frame: pd.DataFrame, *parts: object) -> str:
    digest = hashlib.md5()
    if frame is not None and not frame.empty:
        hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
        digest.update(hashed.tobytes())
    for part in parts:
        digest.update(str(part).encode())
    return digest.hexdigest()


# DataFrames are excluded from the cache key (leading underscore); the explicit
# signature arguments carry the identity so results never go stale.
@st.cache_data(show_spinner=False)
def build_panel(
    _raw: pd.DataFrame,
    raw_sig: str,
    date_col: str,
    target_col: str,
    groups: tuple[str, ...],
    freq: str,
    gap_strategy: str,
) -> tuple[pd.DataFrame, PrepReport]:
    return to_panel(_raw, date_col, target_col, groups, freq, gap_strategy)


@st.cache_data(show_spinner=False)
def run_backtest(
    _panel: pd.DataFrame,
    panel_sig: str,
    models: tuple[str, ...],
    horizon: int,
    folds: int,
    min_train: int,
    freq: str,
) -> BacktestOutcome:
    return rolling_origin(_panel, list(models), horizon, folds, freq, min_train)


@st.cache_data(show_spinner=False)
def project(
    _panel: pd.DataFrame, forecast_sig: str, unique_id: str, model: str, horizon: int, freq: str
) -> pd.DataFrame:
    return forecast_future(_panel, unique_id, model, horizon, freq)


def panel_signature(panel: pd.DataFrame, *parts: object) -> str:
    return frame_signature(panel[["unique_id", "ds", "y"]] if not panel.empty else panel, *parts)


def show_error(exc: Exception, prefix: str) -> None:
    st.error(f"{prefix}: {exc}")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(exc)))


def render_data_notes(report: PrepReport) -> None:
    if report.notes:
        st.markdown("\n".join(f"- {note}" for note in report.notes))
    else:
        st.success("No data quality issues detected.")


def metric_table(summary: pd.DataFrame) -> None:
    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.TextColumn(width="small"),
            "Model": st.column_config.TextColumn(width="medium"),
            "Family": st.column_config.TextColumn(width="medium"),
            "MASE": st.column_config.NumberColumn(format="%.3f", help="Primary ranking metric: MAE / in-sample seasonal-naive MAE."),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "MAPE": st.column_config.NumberColumn(format="%.2f", help="Percent; zero actuals are excluded and counted separately."),
            "SMAPE": st.column_config.NumberColumn(format="%.2f", help="Percent, 0-200 scale."),
            "Bias": st.column_config.NumberColumn(format="%.2f", help="Mean(predicted - actual)."),
            "Series": st.column_config.NumberColumn(format="%d"),
            "Points": st.column_config.NumberColumn(format="%d"),
            "Failed": st.column_config.NumberColumn(format="%d"),
        },
    )


def main() -> None:
    st.markdown('<div class="fk-kicker">Franke · demand planning</div>', unsafe_allow_html=True)
    st.title("Forecast Studio")
    st.markdown(
        '<div class="fk-sub">Rolling-origin backtests and forward projections for item and region '
        "demand. Every score below comes from a model that saw only the periods before the one it "
        "was asked to predict.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Data")
        source = st.radio("Source", ["Sample dataset", "Upload CSV"], horizontal=True)
        if source == "Sample dataset":
            try:
                raw = load_sample()
            except FileNotFoundError:
                st.error(f"Sample data missing at {SAMPLE_PATH}. Run `python scripts/generate_sample.py`.")
                st.stop()
            st.caption(f"{len(raw):,} rows · 4 items x 2 regions · daily 2021-2024")
        else:
            uploaded = st.file_uploader("Upload a CSV", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV with a date column, a target column and optional series columns.")
                st.stop()
            try:
                raw = read_upload(uploaded.getvalue())
            except DataError as exc:
                st.error(str(exc))
                st.stop()

        columns = list(raw.columns)
        date_guess = guess_date_column(raw)
        date_col = st.selectbox(
            "Date column",
            columns,
            index=columns.index(date_guess) if date_guess in columns else 0,
        )

        target_options = [c for c in columns if c != date_col]
        target_guess = guess_target_column(raw)
        target_col = st.selectbox(
            "Target column",
            target_options,
            index=target_options.index(target_guess) if target_guess in target_options else 0,
        )

        group_options = [c for c in columns if c not in (date_col, target_col)]
        if source == "Sample dataset":
            default_groups = [g for g in guess_group_columns(raw, date_col, target_col) if g in group_options]
        else:
            default_groups = []
        group_cols = st.multiselect(
            "Series columns",
            group_options,
            default=default_groups,
            help="Each combination becomes its own series. Leave empty for one aggregate series.",
        )

        st.markdown("### Period")
        freq_label = st.selectbox("Frequency", list(FREQUENCY_LABELS), index=1)
        freq = FREQUENCY_LABELS[freq_label]
        horizon = st.number_input(
            "Forecast horizon (periods)",
            min_value=1,
            max_value=MAX_HORIZON[freq],
            value=DEFAULT_HORIZON[freq],
            key=f"horizon_{freq}",
        )
        gap_label = st.selectbox("Missing periods", list(GAP_STRATEGIES), index=0)

        st.markdown("### Models")
        catalog = model_catalog()
        models = st.multiselect(
            "Models to backtest",
            list(catalog),
            default=[m for m in DEFAULT_MODELS if m in catalog],
        )
        if is_chronos_available():
            st.caption("Chronos runs on CPU: the first forecast downloads weights and is slow.")
        else:
            st.caption(chronos_hint())

        with st.expander("Backtest settings"):
            folds = st.slider("Folds", min_value=1, max_value=6, value=DEFAULT_FOLDS, key=f"folds_{freq}")
            min_train = st.number_input(
                "Minimum training periods",
                min_value=7,
                max_value=520,
                value=default_min_train(freq),
                key=f"min_train_{freq}",
                help="Each series needs min_train + folds x horizon periods to be backtested.",
            )

    try:
        with st.spinner("Preparing series..."):
            panel, report = build_panel(
                raw,
                frame_signature(raw),
                date_col,
                target_col,
                tuple(group_cols),
                freq,
                GAP_STRATEGIES[gap_label],
            )
    except DataError as exc:
        st.error(str(exc))
        st.stop()

    all_series = sorted(panel["unique_id"].astype(str).unique().tolist())
    overview_tab, backtest_tab, forecast_tab, data_tab = st.tabs(
        ["Overview", "Backtest", "Forecast", "Data & quality"]
    )

    with overview_tab:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Series", f"{report.series_count:,}")
        k2.metric("Periods", f"{report.output_rows:,}")
        k3.metric("Window", f"{report.date_start} → {report.date_end}")
        k4.metric("Target", str(target_col))

        st.markdown("#### Series trend")
        selected_series = st.selectbox("Series", all_series, key="overview_series")
        series_frame = panel[panel["unique_id"] == selected_series]
        st.plotly_chart(
            history_chart(series_frame, selected_series, y_title=str(target_col)),
            use_container_width=True,
        )

        stats = series_stats(panel)
        st.markdown("#### Per-series summary")
        st.dataframe(
            stats,
            hide_index=True,
            use_container_width=True,
            column_config={
                "unique_id": st.column_config.TextColumn("Series", width="large"),
                "periods": st.column_config.NumberColumn("Periods", format="%d"),
                "start": st.column_config.DateColumn("Start", format="YYYY-MM-DD"),
                "end": st.column_config.DateColumn("End", format="YYYY-MM-DD"),
                "total": st.column_config.NumberColumn("Total", format="%.0f"),
                "mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                "zeros": st.column_config.NumberColumn("Zeros", format="%d"),
                "negatives": st.column_config.NumberColumn("Negatives", format="%d"),
            },
        )

        with st.expander("How the evaluation works"):
            st.markdown(
                """
- **Rolling origin.** Each fold trains on everything before a cutoff, predicts the next
  `horizon` periods, then moves the cutoff forward. The last fold always ends at the newest data.
- **MASE** is the primary metric: mean absolute error divided by the in-sample seasonal-naive
  error of the same series. Below 1 means the model beats a seasonal-naive forecast.
- **Summary values are means across series**, so one large series cannot dominate the ranking.
  The per-series table shows where a model is unstable.
- **MAPE** ignores zero actuals and reports how many points it used; **SMAPE** is on a 0–200 scale.
- **Failures are not hidden.** Any model/series combination that could not produce a valid
  forecast appears in the Failures table with the reason.
                """
            )

    with backtest_tab:
        st.markdown("#### Rolling-origin backtest")
        if not models:
            st.warning("Select at least one model in the sidebar.")
        else:
            chosen = st.multiselect(
                "Series in this backtest",
                all_series,
                default=all_series,
                help="Runtime grows with the number of series and folds.",
            )
            if len(all_series) > 20:
                st.caption(f"{len(all_series)} series available; consider a subset to keep the run short.")

            panel_use = panel[panel["unique_id"].isin(chosen)]
            fold_span = int(folds) * int(horizon)
            counts = panel_use.groupby("unique_id").size() if not panel_use.empty else pd.Series(dtype=int)
            by_requirement: dict[int, list[str]] = {}
            for model_name in models:
                requirement = max(int(min_train), minimum_history(model_name)) + fold_span
                by_requirement.setdefault(requirement, []).append(model_name)
            for requirement, model_names in sorted(by_requirement.items()):
                blocked = counts[counts < requirement]
                if len(blocked):
                    listed = ", ".join(f"{uid} ({count} periods)" for uid, count in blocked.items())
                    subject = model_names[0] if len(model_names) == 1 else ", ".join(model_names)
                    verb = "needs" if len(model_names) == 1 else "need"
                    st.warning(
                        f"{subject} {verb} {requirement} periods per series: "
                        f"{listed}. Those runs will be listed under Failures."
                    )

            sig = panel_signature(
                panel_use, tuple(models), int(horizon), int(folds), int(min_train), freq
            )
            run_clicked = st.button(
                "Run backtest",
                type="primary",
                disabled=not chosen or not models,
            )

            if run_clicked:
                with st.spinner("Fitting models across folds..."):
                    try:
                        outcome = run_backtest(
                            panel_use,
                            sig,
                            tuple(models),
                            int(horizon),
                            int(folds),
                            int(min_train),
                            freq,
                        )
                    except Exception as exc:  # noqa: BLE001 - surfaced to the user
                        st.session_state.pop("backtest", None)
                        show_error(exc, "Backtest failed")
                        outcome = None
                if outcome is not None:
                    st.session_state["backtest"] = {"sig": sig, "outcome": outcome}

            state = st.session_state.get("backtest")
            if state and state["sig"] == sig:
                outcome: BacktestOutcome = state["outcome"]
                best = outcome.best_model
                if best is None:
                    st.error("No model produced a forecast. Check the Failures tab.")
                else:
                    st.markdown(
                        f"**Best model by MASE: {best}.** Lower is better; values are averaged "
                        "equally across the backtested series."
                    )
                metric_table(outcome.summary)

                failure_count = len(outcome.failures)
                if failure_count:
                    st.warning(
                        f"{failure_count} model/series combination(s) produced no forecast. "
                        "See the Failures tab for the affected series and reasons."
                    )

                summary_tab, series_tab, detail_tab, failures_tab = st.tabs(
                    ["Summary", "By series", "Fold detail", f"Failures ({failure_count})"]
                )

                with summary_tab:
                    if not outcome.per_series.empty:
                        st.plotly_chart(mase_heatmap(outcome.per_series), use_container_width=True)
                    else:
                        st.info("No per-series metrics are available.")

                with series_tab:
                    if outcome.per_series.empty:
                        st.info("No per-series metrics are available.")
                    else:
                        view = outcome.per_series.rename(
                            columns={
                                "model": "Model",
                                "family": "Family",
                                "unique_id": "Series",
                                "fold_count": "Folds",
                                "n_points": "Points",
                                "zeros_actual": "Zero actuals",
                                "mape_points": "MAPE points",
                                "mase": "MASE",
                                "rmse": "RMSE",
                                "mae": "MAE",
                                "mape": "MAPE",
                                "smape": "SMAPE",
                                "bias": "Bias",
                            }
                        ).sort_values(["Model", "Series"])
                        st.dataframe(
                            view,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "MASE": st.column_config.NumberColumn(format="%.3f"),
                                "RMSE": st.column_config.NumberColumn(format="%.2f"),
                                "MAE": st.column_config.NumberColumn(format="%.2f"),
                                "MAPE": st.column_config.NumberColumn(format="%.2f"),
                                "SMAPE": st.column_config.NumberColumn(format="%.2f"),
                                "Bias": st.column_config.NumberColumn(format="%.2f"),
                                "Points": st.column_config.NumberColumn(format="%d"),
                            },
                        )

                with detail_tab:
                    available = [
                        name
                        for name in outcome.summary["Model"]
                        if not outcome.per_series[outcome.per_series["model"] == name].empty
                    ]
                    if not available:
                        st.info("No forecasts to plot.")
                    else:
                        left, right = st.columns(2)
                        detail_model = left.selectbox("Model", available, key="detail_model")
                        detail_series_options = sorted(
                            outcome.per_series.loc[
                                outcome.per_series["model"] == detail_model, "unique_id"
                            ].unique()
                        )
                        detail_series = right.selectbox(
                            "Series",
                            detail_series_options,
                            key=f"detail_series_{detail_model}",
                        )
                        st.plotly_chart(
                            backtest_chart(
                                outcome.forecasts,
                                detail_model,
                                detail_series,
                                outcome.folds,
                                y_title=str(target_col),
                            ),
                            use_container_width=True,
                        )
                        fold_rows = outcome.folds[outcome.folds["unique_id"] == detail_series]
                        st.dataframe(fold_rows, hide_index=True, use_container_width=True)

                with failures_tab:
                    if outcome.failures.empty:
                        st.success("Every selected model produced a valid forecast for every series.")
                    else:
                        st.dataframe(
                            outcome.failures.rename(
                                columns={
                                    "model": "Model",
                                    "unique_id": "Series",
                                    "fold": "Fold",
                                    "stage": "Stage",
                                    "detail": "Reason",
                                }
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )
            elif state:
                st.info("Backtest settings changed. Run the backtest again to refresh the results.")
            else:
                st.info("Choose series and run the backtest to score the selected models.")

    with forecast_tab:
        st.markdown("#### Project a series forward")
        st.caption(
            "A point forecast from a model fitted on the full history. Use the backtest to judge "
            "whether that model has earned your trust."
        )

        backtest_state = st.session_state.get("backtest")
        preferred = None
        if backtest_state:
            preferred = backtest_state["outcome"].best_model
        if preferred not in catalog:
            preferred = "Seasonal naive" if "Seasonal naive" in catalog else list(catalog)[0]

        c1, c2, c3 = st.columns([1.6, 1.2, 0.8])
        forecast_uid = c1.selectbox("Series", all_series, key="forecast_series")
        forecast_model = c2.selectbox(
            "Model",
            list(catalog),
            index=list(catalog).index(preferred),
            key="forecast_model",
        )
        forecast_steps = c3.number_input(
            "Periods ahead",
            min_value=1,
            max_value=MAX_HORIZON[freq],
            value=int(horizon),
            key=f"forecast_steps_{freq}",
        )

        generate = st.button("Generate forecast", type="primary")

        fsig = panel_signature(panel, forecast_uid, forecast_model, int(forecast_steps), freq)
        if generate:
            with st.spinner("Fitting on the full history..."):
                try:
                    forecast = project(
                        panel,
                        fsig,
                        forecast_uid,
                        forecast_model,
                        int(forecast_steps),
                        freq,
                    )
                    st.session_state["forecast"] = {
                        "sig": fsig,
                        "frame": forecast,
                        "uid": forecast_uid,
                        "model": forecast_model,
                    }
                except Exception as exc:  # noqa: BLE001 - surfaced to the user
                    st.session_state.pop("forecast", None)
                    show_error(exc, "Forecast failed")

        forecast_state = st.session_state.get("forecast")
        if forecast_state and forecast_state["sig"] == fsig:
            frame = forecast_state["frame"]
            history = panel[panel["unique_id"] == forecast_state["uid"]]
            st.plotly_chart(
                history_chart(
                    history,
                    f"{forecast_state['uid']} · {forecast_state['model']}",
                    frame,
                    y_title=str(target_col),
                ),
                use_container_width=True,
            )
            table = frame.rename(columns={"ds": "Period", "y_pred": f"Forecast ({target_col})"})
            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Period": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    f"Forecast ({target_col})": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.download_button(
                "Download forecast CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{forecast_state['uid'].replace(' / ', '_').replace(' ', '_')}.csv",
                mime="text/csv",
            )
            st.caption(
                f"Fitted through {history['ds'].max():%Y-%m-%d}. Point forecasts only; "
                "no prediction interval is produced."
            )
        elif forecast_state:
            st.info("Forecast settings changed. Generate again to refresh.")
        else:
            st.info("Pick a series and model, then generate a forecast.")

    with data_tab:
        st.markdown("#### Data quality")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Input rows", f"{report.input_rows:,}")
        q2.metric("Panel rows", f"{report.output_rows:,}")
        q3.metric("Dropped rows", f"{report.dropped_invalid_dates + report.dropped_invalid_targets:,}")
        q4.metric("Merged duplicates", f"{report.merged_duplicate_rows:,}")

        q5, q6, q7 = st.columns(3)
        q5.metric("Gaps filled", f"{report.gaps_filled:,}")
        q6.metric("Partial periods dropped", f"{report.dropped_partial_periods:,}")
        q7.metric("Negative targets", f"{report.negative_values:,}")

        st.markdown("#### Quality notes")
        render_data_notes(report)

        st.download_button(
            "Download completed panel CSV",
            data=panel.to_csv(index=False).encode("utf-8"),
            file_name=f"franforecast_panel_{freq_label.lower()}.csv",
            mime="text/csv",
        )

        with st.expander("Raw input preview"):
            st.dataframe(raw.head(20), use_container_width=True)

        with st.expander("Completed panel preview"):
            st.dataframe(panel.head(50), hide_index=True, use_container_width=True)

        with st.expander("Per-series statistics"):
            st.dataframe(series_stats(panel), hide_index=True, use_container_width=True)

    st.caption("Franke Forecast Studio · v2.0.0 · evaluation-first rebuild")


if __name__ == "__main__":
    main()
