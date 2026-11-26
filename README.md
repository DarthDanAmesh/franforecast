# Franke Forecast Studio

Demand forecasting with rolling-origin backtesting for item/region sales series.
Every score shown by the app or the CLI comes from a model that was fitted only
on periods before the ones it was asked to predict.

Run it:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens with a synthetic sample dataset, so it is usable immediately.
Upload your own CSV from the sidebar to replace it.

## What it does

1. **Prepare panel data.** A date column, a numeric target column and zero or
   more series columns are parsed into a `(unique_id, ds, y)` panel at daily,
   weekly (weeks ending Sunday) or monthly frequency. Duplicates are summed,
   missing periods are filled (zero, interpolate or drop) and partial buckets at
   the edges are trimmed so each period covers its full span.
2. **Backtest.** Models are scored with an expanding-window rolling-origin
   scheme: each fold trains on everything before a cutoff, predicts the next
   `horizon` periods, and the cutoff advances. Folds never overlap.
3. **Project.** The chosen model is refitted on the full history and produces a
   forward point forecast.

## Models

| Name | Family | Notes |
| --- | --- | --- |
| Naive | baseline | Last observed value repeated. |
| Seasonal naive | baseline | Value from `season_length` periods ago. |
| ETS | statistical | Holt-Winters (statsmodels), seasonal when history allows. |
| LightGBM | ML | Lags, rolling means, German holiday and calendar features; recursive multi-step. |
| Chronos | foundation | Optional; `pip install -e ".[chronos]"`. Median of sampled trajectories, CPU. |

Chronos is hidden from the model picker when it is not installed. It is not
covered by the automated tests and has not been benchmarked on this machine.

## Metrics

| Metric | Definition |
| --- | --- |
| **MASE** | MAE divided by the in-sample seasonal-naive error of the same series. Primary ranking metric; below 1 beats seasonal naive. |
| RMSE / MAE | In the data's own units. |
| MAPE | Mean absolute percentage error, excluding points where the actual is zero, with the number of points used reported. |
| SMAPE | Symmetric MAPE on a 0–200 scale. |
| Bias | Mean signed error; positive means the forecast runs high. |

Summary rows are unweighted means across series, so one large series cannot
dominate the ranking. Per-series tables show where a model is unstable. Any
model/series/fold that fails to produce a valid forecast is listed in the
Failures table with the reason instead of being silently dropped.

## Data requirements

- A date column that parses to calendar dates.
- A numeric target column; rows with unparseable dates or non-numeric targets
  are dropped and reported, never coerced silently.
- Optional series columns (for example `Item_Code` and `Region`). Without them
  the whole file is treated as one series.
- The typical spacing between dates must not be longer than the chosen period,
  because upsampling would invent detail: daily data can be aggregated to
  weekly or monthly, but monthly data cannot be turned into weeks.

The app guesses date/target/series columns and lets you override every choice.

## Command line

```bash
python scripts/run_backtest.py --freq Weekly --horizon 8 --folds 3 \
    --models "Naive,Seasonal naive,ETS,LightGBM"
python scripts/generate_sample.py        # rewrite data/sample_franke_sales.csv
```

`--csv` points at another file; `--groups "Item_Code,Region"` sets the series
columns explicitly; `--models all` runs everything available.

## Architecture

The project is a small package with one data contract between the layers, so the
Streamlit app, the CLI and the tests drive the same code.

```
CSV file
  │  read_table + guess_date_column / guess_target_column / guess_group_columns
  ▼
panel  (unique_id, ds, y)          one row per series and period
  │  rolling_origin(panel, models, horizon, folds, freq)
  ├─ per fold: fit(models) on ds < cutoff → predict the next `horizon` periods
  ├─ evaluate(y_true, y_pred, y_train, season) → MASE, RMSE, MAE, MAPE, SMAPE, Bias
  ├─ summary: mean across series, failed series counted
  └─ failures: model / series / fold / stage / reason
panel + one model ── forecast_future ──▶ future point forecast
```

| Module | Responsibility |
| --- | --- |
| `config.py` | Canonical offsets (`D`, `W-SUN`, `MS`), season lengths, horizon limits, gap strategies, `future_dates()`. |
| `data.py` | `read_table`, column guessing, `to_panel` (parse, dedupe, resample, trim, gap handling), `guard_frequency`, `series_stats`, `PrepReport`. |
| `features.py` | `FeatureBuilder`: lags, rolling mean/std, calendar and holiday features; builds the training matrix and replays the same rows recursively at inference. |
| `models.py` | `Forecaster` base class plus Naive, Seasonal naive, ETS, LightGBM and Chronos; `model_catalog`, `minimum_history`. |
| `backtest.py` | `rolling_origin`, `BacktestOutcome` (summary, per series, forecasts, folds, failures), `forecast_future`. |
| `metrics.py` | `evaluate()` — the only place metric definitions live. |
| `plotting.py` | Plotly builders: history, backtest with fold bands, MASE heatmap. |
| `synthetic.py` | Reproducible sample dataset (seasonality, German holidays, promos, stockouts). |
| `app.py` | Streamlit UI: sidebar mapping, four tabs, cached runs keyed by content fingerprints. |
| `scripts/` | `run_backtest.py` (headless report) and `generate_sample.py`. |

### Contracts and invariants

- **Panel.** Exactly `unique_id`, `ds`, `y`; one row per series and period;
  sorted, contiguous grid at the chosen offset. Duplicates are summed and gaps
  are resolved while the panel is built, so every downstream step sees regular
  series. `PrepReport` records each row dropped, merged, filled or trimmed.
- **Model.** `fit(SeriesHistory)` receives that series only, up to the cutoff;
  `predict(future_dates)` must return one finite value per date. The backtester
  checks length and finiteness and turns anything else into a failure row
  instead of aborting the run or scoring it as a good forecast.
- **No leakage.** Lags use values strictly before the target, rolling windows
  are shifted by one period, and a fold slices `dates[:cutoff]`. A test
  perturbs every value after the cutoff and asserts fold-1 forecasts are
  unchanged.
- **Comparable MASE.** The seasonal-naive scale is computed once from the first
  training window of each series and reused for all its folds, so folds and
  series stay on the same footing.
- **Honest summary.** Rows are unweighted means across series; a model with no
  valid points keeps a blank MASE and ranks last; `Failed` counts series with
  at least one failure, not individual folds.
- **Frequency guard.** The typical spacing between dates must not exceed the
  chosen period (7 / 10.5 / 45.6 days for daily, weekly, monthly). Finer data
  can be aggregated to a coarser frequency; coarser data is rejected rather
  than split into buckets the data never had.
- **Non-negative demand.** When a series' training history is non-negative,
  forecasts are clipped at zero.
- **UI state.** Expensive steps are cached on a hash of the panel content plus
  settings, and results are held in `session_state` behind the same signature,
  so changing a widget never shows a result computed for different inputs.

Lag and rolling features are expressed in periods, not calendar days: the
defaults are tuned for daily data, and on monthly series the 14- and 28-period
lags consume most of a short history.

## Tests

```bash
pip install -r requirements.txt
pytest
```

The suite covers panel construction and its failure modes, metric edge cases,
feature construction (including a test that proves future values cannot leak
into a fold's forecast), backtest failure reporting, an end-to-end two-series
backtest, and a headless render of the Streamlit app.

## Status and limits

- Validated against the synthetic sample: ETS ranks best (weekly MASE ≈ 1.12),
  then LightGBM (≈ 1.20), seasonal naive (≈ 1.34) and naive (≈ 1.37), with no
  failed forecasts.
- LightGBM needs 58 observations before its first fold; on short series (for
  example four years of monthly data) it cannot train. The app warns before the
  run, naming every affected series, and the failures are reported, not hidden.
- Accuracy on real Franke data has not been measured; the sample generator is
  plausible, not calibrated.
- Forecasts are point forecasts only. No prediction intervals.
- Everything is CPU-only and single-process by design.
