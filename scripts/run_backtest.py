"""Headless rolling-origin backtest, e.g.:

    python scripts/run_backtest.py --freq Weekly --horizon 8 --folds 3

Prints the full summary, per-series metrics and any per-model failures.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franforecast.backtest import rolling_origin  # noqa: E402
from franforecast.config import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_HORIZON,
    FREQUENCY_LABELS,
    default_min_train,
)
from franforecast.data import (  # noqa: E402
    guess_date_column,
    guess_group_columns,
    guess_target_column,
    read_table,
    to_panel,
)
from franforecast.models import DEFAULT_MODELS, model_catalog  # noqa: E402

DEFAULT_CSV = ROOT / "data" / "sample_franke_sales.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--freq", choices=list(FREQUENCY_LABELS), default="Weekly")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--min-train", type=int, default=None)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names, or 'all'.",
    )
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated series columns. Empty means auto-detect.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freq = FREQUENCY_LABELS[args.freq]
    horizon = args.horizon or DEFAULT_HORIZON[freq]
    min_train = args.min_train or default_min_train(freq)

    raw = read_table(args.csv)
    date_col = guess_date_column(raw)
    target_col = guess_target_column(raw)
    if not date_col or not target_col:
        print(f"Could not identify date/target columns. Columns: {list(raw.columns)}")
        return 1

    if args.groups.strip():
        groups = tuple(part.strip() for part in args.groups.split(",") if part.strip())
    else:
        groups = tuple(guess_group_columns(raw, date_col, target_col))

    panel, report = to_panel(raw, date_col, target_col, groups, freq, "zero")

    catalog = model_catalog()
    if args.models.strip().lower() == "all":
        models = list(catalog)
    else:
        models = [part.strip() for part in args.models.split(",") if part.strip()]
    unknown = [m for m in models if m not in catalog]
    if unknown:
        print(f"Unknown or unavailable models: {unknown}. Available: {list(catalog)}")
        return 1

    print(f"Source        : {args.csv}")
    print(f"Series        : {report.series_count} ({', '.join(groups) if groups else 'single'})")
    print(f"Frequency     : {args.freq} ({freq})")
    print(f"Rows          : {report.input_rows:,} in -> {report.output_rows:,} panel rows")
    print(f"Period        : {report.date_start} to {report.date_end}")
    print(f"Gaps filled   : {report.gaps_filled}")
    for note in report.notes:
        print(f"Note          : {note}")
    print(f"Models        : {models}")
    print(f"Backtest      : {args.folds} folds, horizon {horizon}, min_train {min_train}")

    started = time.perf_counter()
    outcome = rolling_origin(panel, models, horizon, args.folds, freq, min_train)
    elapsed = time.perf_counter() - started

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print("\n=== SUMMARY (metrics are means across series; MASE is primary) ===")
    print(outcome.summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\nBest model by MASE: {outcome.best_model}   ({elapsed:.1f}s)")

    print("\n=== PER SERIES ===")
    if outcome.per_series.empty:
        print("No successful forecasts.")
    else:
        print(
            outcome.per_series[
                ["model", "unique_id", "mase", "rmse", "mae", "mape", "bias", "n_points"]
            ].to_string(index=False, float_format=lambda v: f"{v:,.3f}")
        )

    print("\n=== FAILURES ===")
    if outcome.failures.empty:
        print("None.")
    else:
        print(outcome.failures.to_string(index=False))

    return 0 if not outcome.per_series.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
