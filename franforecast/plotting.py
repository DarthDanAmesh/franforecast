"""Plotly figures with a consistent visual language."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ACCENT = "#B4232A"
INK = "#1F2430"
MUTED = "#7A8290"
GRID = "#E8EBF0"
BLUE = "#2F6690"
FILL = "#F7E9EA"

FONT = "Inter, Segoe UI, system-ui, sans-serif"


def _style(fig: go.Figure, height: int = 380, y_title: str = "Units") -> go.Figure:
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=8, r=8, t=52, b=8),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"),
        xaxis=dict(showgrid=False, linecolor=GRID, title=None),
        yaxis=dict(gridcolor=GRID, zeroline=False, title=y_title),
    )
    return fig


def history_chart(
    series: pd.DataFrame,
    title: str,
    forecast: pd.DataFrame | None = None,
    y_title: str = "Units",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["ds"],
            y=series["y"],
            mode="lines",
            name="Actual",
            line=dict(color=BLUE, width=1.8),
            hovertemplate="%{x|%d %b %Y}<br>Actual %{y:,.0f}<extra></extra>",
        )
    )
    if forecast is not None and not forecast.empty:
        anchor = pd.DataFrame({"ds": [series["ds"].iloc[-1]], "y": [series["y"].iloc[-1]]})
        line = pd.concat([anchor, forecast[["ds", "y_pred"]].rename(columns={"y_pred": "y"})])
        fig.add_trace(
            go.Scatter(
                x=line["ds"],
                y=line["y"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color=ACCENT, width=2.4, dash="dash"),
                marker=dict(size=5),
                hovertemplate="%{x|%d %b %Y}<br>Forecast %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15, color=INK)))
    return _style(fig, y_title=y_title)


def backtest_chart(
    forecasts: pd.DataFrame,
    model: str,
    unique_id: str,
    folds: pd.DataFrame | None = None,
    y_title: str = "Units",
) -> go.Figure:
    subset = forecasts[(forecasts["model"] == model) & (forecasts["unique_id"] == unique_id)]
    subset = subset.sort_values("ds")
    fig = go.Figure()
    if subset.empty:
        fig.update_layout(title=dict(text="No forecasts for this combination", x=0))
        return _style(fig, y_title=y_title)

    if folds is not None and not folds.empty:
        fold_rows = folds[folds["unique_id"] == unique_id]
        for _, row in fold_rows.iterrows():
            fig.add_vrect(
                x0=row["test_start"],
                x1=row["test_end"],
                fillcolor=FILL,
                opacity=0.55,
                line_width=0,
                layer="below",
            )

    fig.add_trace(
        go.Scatter(
            x=subset["ds"],
            y=subset["y_true"],
            mode="lines",
            name="Actual",
            line=dict(color=BLUE, width=2),
            hovertemplate="%{x|%d %b %Y}<br>Actual %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=subset["ds"],
            y=subset["y_pred"],
            mode="lines+markers",
            name=f"{model} forecast",
            line=dict(color=ACCENT, width=1.6, dash="dot"),
            marker=dict(size=5),
            customdata=subset["fold"],
            hovertemplate="%{x|%d %b %Y}<br>Forecast %{y:,.0f}<br>Fold %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"{unique_id} · {model}", x=0, font=dict(size=15, color=INK))
    )
    return _style(fig, height=420, y_title=y_title)


def mase_heatmap(per_series: pd.DataFrame) -> go.Figure:
    pivot = per_series.pivot(index="model", columns="unique_id", values="mase")
    pivot = pivot.reindex(sorted(pivot.index))
    values = pivot.to_numpy(dtype=float)
    show_text = values.size <= 120
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[
                [0.0, "#2E6F5E"],
                [0.35, "#8FBF9F"],
                [0.5, "#F2EFE6"],
                [0.75, "#E2A08C"],
                [1.0, "#B4232A"],
            ],
            zmin=0,
            zmax=float(np.nanpercentile(values, 90)) if np.isfinite(values).any() else 1,
            text=np.round(values, 2),
            texttemplate="%{text}" if show_text else None,
            hovertemplate="%{y}<br>%{x}<br>MASE %{z:.3f}<extra></extra>",
            colorbar=dict(title="MASE", thickness=12, len=0.7),
        )
    )
    fig.update_layout(
        title=dict(text="MASE by model and series (lower is better)", x=0, font=dict(size=15)),
        margin=dict(l=8, r=8, t=52, b=8),
        height=max(260, 70 + 42 * len(pivot.index)),
        font=dict(family=FONT, size=12, color=INK),
        xaxis=dict(title=None, tickangle=-20),
        yaxis=dict(title=None, autorange="reversed"),
        plot_bgcolor="white",
    )
    return fig
