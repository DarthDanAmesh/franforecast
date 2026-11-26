"""Turn an uploaded CSV into a clean panel of (unique_id, ds, y)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FREQUENCY_LABELS, LABELS_BY_OFFSET
from .errors import DataError

DATE_HINTS = ("date", "ds", "datetime", "timestamp", "time", "period", "order_date")
TARGET_HINTS = (
    "target",
    "y",
    "units_sold",
    "sales",
    "quantity",
    "qty",
    "demand",
    "units",
    "volume",
    "revenue",
)


def read_table(source) -> pd.DataFrame:
    """Read a CSV from a path or file-like object and normalise column names."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DataError("The CSV contains no rows.") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
        raise DataError(f"Could not read the CSV: {exc}") from exc

    if df.empty:
        raise DataError("The CSV contains no rows.")

    df.columns = [str(c).strip() for c in df.columns]
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        raise DataError(
            f"Duplicate column names after trimming spaces: {duplicates}. "
            "Rename the columns so each one is unique."
        )
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.select_dtypes(include=[np.number]).columns]


def _parse_candidates(series: pd.Series, limit: int = 25) -> float:
    sample = series.dropna().head(limit)
    if sample.empty:
        return 0.0
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def guess_date_column(df: pd.DataFrame) -> str | None:
    scored: list[tuple[float, int, str]] = []
    for position, column in enumerate(df.columns):
        lowered = str(column).lower()
        hint = 2 if any(h in lowered for h in DATE_HINTS) else 0
        score = _parse_candidates(df[column])
        if score > 0.8:
            scored.append((score + hint * 0.1, -position, str(column)))
    return max(scored)[2] if scored else None


def guess_target_column(df: pd.DataFrame) -> str | None:
    numeric = numeric_columns(df)
    if not numeric:
        return None
    for hint in TARGET_HINTS:
        for column in numeric:
            if str(column).lower() == hint:
                return column
    for hint in TARGET_HINTS:
        for column in numeric:
            if hint in str(column).lower():
                return column
    return numeric[0]


def guess_group_columns(df: pd.DataFrame, date_col: str, target_col: str) -> list[str]:
    candidates = [
        str(c)
        for c in df.columns
        if str(c) not in (date_col, target_col) and df[c].nunique(dropna=False) <= max(50, len(df) // 20)
    ]
    return candidates[:2]


@dataclass
class PrepReport:
    input_rows: int
    output_rows: int
    series_count: int
    frequency: str
    dropped_invalid_dates: int = 0
    dropped_invalid_targets: int = 0
    merged_duplicate_rows: int = 0
    gaps_filled: int = 0
    gaps_dropped: int = 0
    dropped_partial_periods: int = 0
    negative_values: int = 0
    date_start: str | None = None
    date_end: str | None = None
    notes: list[str] = field(default_factory=list)


def _examples(series: pd.Series, mask: pd.Series, limit: int = 3) -> str:
    values = series[mask].astype(str).head(limit).tolist()
    rows = [str(i) for i in series[mask].head(limit).index.tolist()]
    return f"values {values} at row indices {rows}"


def to_panel(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_cols: tuple[str, ...] = (),
    freq: str = "D",
    gap_strategy: str = "zero",
) -> tuple[pd.DataFrame, PrepReport]:
    """Validate, aggregate and complete the input into a long time-series panel."""
    if freq not in FREQUENCY_LABELS.values():
        raise DataError(f"Unsupported frequency '{freq}'. Use one of {list(FREQUENCY_LABELS.values())}.")
    if gap_strategy not in {"zero", "interpolate", "drop"}:
        raise DataError(f"Unsupported gap strategy '{gap_strategy}'.")

    required = [date_col, target_col, *group_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(
            f"Columns not found: {missing}. Available columns: {list(df.columns)}."
        )
    if date_col == target_col:
        raise DataError("The date column and the target column must be different.")
    if target_col in group_cols:
        raise DataError("The target column cannot also be a series column.")

    notes: list[str] = []
    report = PrepReport(
        input_rows=len(df),
        output_rows=0,
        series_count=0,
        frequency=LABELS_BY_OFFSET[freq],
    )

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    bad_dates = parsed_dates.isna()
    if bad_dates.any():
        report.dropped_invalid_dates = int(bad_dates.sum())
        notes.append(
            f"Dropped {report.dropped_invalid_dates} row(s) with an unparseable date "
            f"({_examples(df[date_col], bad_dates)})."
        )
    if parsed_dates.notna().sum() == 0:
        raise DataError(
            f"Column '{date_col}' contains no parseable dates. "
            f"First values: {df[date_col].astype(str).head(5).tolist()}."
        )

    numeric_target = pd.to_numeric(df[target_col], errors="coerce")
    bad_targets = numeric_target.isna()
    if bad_targets.any():
        report.dropped_invalid_targets = int(bad_targets.sum())
        notes.append(
            f"Dropped {report.dropped_invalid_targets} row(s) with a non-numeric target "
            f"({_examples(df[target_col], bad_targets)})."
        )

    keep = ~(bad_dates | bad_targets)
    if not keep.any():
        raise DataError("No rows are left after removing invalid dates and targets.")

    if group_cols:
        parts = [
            df[c].map(lambda value: "(missing)" if pd.isna(value) else str(value).strip())
            for c in group_cols
        ]
        unique_id = parts[0].astype(str)
        for part in parts[1:]:
            unique_id = unique_id + " / " + part.astype(str)
    else:
        unique_id = pd.Series("All series", index=df.index, dtype="object")

    work = pd.DataFrame(
        {
            "unique_id": unique_id[keep].astype(str),
            "ds": parsed_dates[keep],
            "y": numeric_target[keep].astype(float),
        }
    )

    finite = np.isfinite(work["y"].to_numpy())
    if not finite.all():
        dropped = int((~finite).sum())
        report.dropped_invalid_targets += dropped
        notes.append(f"Dropped {dropped} row(s) whose target was infinite.")
        work = work.loc[finite]

    report.negative_values = int((work["y"] < 0).sum())
    if report.negative_values:
        notes.append(
            f"Kept {report.negative_values} negative target value(s); they are treated as returns."
        )

    before_dedup = len(work)
    work = (
        work.groupby(["unique_id", "ds"], as_index=False, sort=True)["y"]
        .sum()
        .sort_values(["unique_id", "ds"])
    )
    report.merged_duplicate_rows = before_dedup - len(work)

    guard_frequency(work, freq)

    frames: list[pd.DataFrame] = []
    for uid, group in work.groupby("unique_id", sort=True):
        series = group.set_index("ds")["y"].sort_index()
        resampled = series.resample(freq).sum(min_count=1)
        grid = pd.date_range(resampled.index.min(), resampled.index.max(), freq=freq)
        if len(resampled) != len(grid):
            resampled = resampled.reindex(grid)

        resampled, partial = _trim_partial_periods(resampled, series.index, freq)
        report.dropped_partial_periods += partial

        gaps = int(resampled.isna().sum())
        report.gaps_filled += gaps
        if gaps:
            if gap_strategy == "zero":
                resampled = resampled.fillna(0.0)
            elif gap_strategy == "interpolate":
                resampled = resampled.interpolate(method="time").ffill().bfill()
            else:
                resampled = resampled.dropna()
                report.gaps_dropped += gaps
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": str(uid),
                    "ds": resampled.index,
                    "y": resampled.to_numpy(dtype=float),
                }
            )
        )

    if not frames:
        raise DataError("No series could be built from the input.")

    panel = pd.concat(frames, ignore_index=True).sort_values(["unique_id", "ds"])
    panel = panel.reset_index(drop=True)

    report.output_rows = len(panel)
    report.series_count = int(panel["unique_id"].nunique())
    report.date_start = panel["ds"].min().date().isoformat()
    report.date_end = panel["ds"].max().date().isoformat()
    if report.dropped_partial_periods:
        notes.append(
            f"Dropped {report.dropped_partial_periods} partial period(s) at the edges so every "
            f"{LABELS_BY_OFFSET[freq].lower()} bucket covers a complete period."
        )
    report.notes = notes
    return panel, report


def _trim_partial_periods(
    resampled: pd.Series, observed: pd.DatetimeIndex, freq: str
) -> tuple[pd.Series, int]:
    """Drop first/last week or month when the source data does not cover all of it."""
    if freq not in {"W-SUN", "MS"} or len(resampled) <= 2:
        return resampled, 0

    first_label = resampled.index[0]
    last_label = resampled.index[-1]
    if freq == "W-SUN":
        first_start = first_label - pd.Timedelta(days=6)
        last_end = last_label
    else:
        first_start = first_label
        last_end = last_label + pd.offsets.MonthEnd(0)

    span_start = pd.Timestamp(observed.min()).normalize()
    span_end = pd.Timestamp(observed.max()).normalize()
    dropped = 0
    if span_start > first_start:
        resampled = resampled.iloc[1:]
        dropped += 1
    if span_end < last_end and len(resampled) > 2:
        resampled = resampled.iloc[:-1]
        dropped += 1
    return resampled, dropped


def guard_frequency(work: pd.DataFrame, freq: str) -> None:
    """Reject upsampling: daily is the only frequency finer than the data's own."""
    unique_days = np.sort(work["ds"].unique())
    if len(unique_days) < 3:
        return
    diffs = np.diff(unique_days).astype("timedelta64[D]").astype(float)
    median_gap = float(np.median(diffs))
    max_gap_days = {"D": 7.0, "W-SUN": 10.5, "MS": 45.6}[freq]
    if median_gap > max_gap_days:
        label = LABELS_BY_OFFSET[freq]
        raise DataError(
            f"The data's typical spacing is {median_gap:.0f} day(s), which is coarser than "
            f"the selected {label} frequency. Upsampling would invent detail; pick a coarser frequency."
        )


def series_stats(panel: pd.DataFrame) -> pd.DataFrame:
    stats = (
        panel.groupby("unique_id", as_index=False)
        .agg(
            periods=("y", "size"),
            start=("ds", "min"),
            end=("ds", "max"),
            total=("y", "sum"),
            mean=("y", "mean"),
            zeros=("y", lambda s: int((s == 0).sum())),
            negatives=("y", lambda s: int((s < 0).sum())),
        )
        .sort_values("unique_id")
        .reset_index(drop=True)
    )
    return stats
