# Generates summary reports and visualizations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.date_utils import generate_dates

def generate_summary_report(metrics_dict):
    """
    Generate a summary report of evaluation metrics.
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing evaluation metrics for each model.
    Returns
    -------
    pd.DataFrame
        Summary report as a DataFrame.
    """
    summary = []
    for model, metrics in metrics_dict.items():
        metrics["Model"] = model
        summary.append(metrics)
    return pd.DataFrame(summary)



def plot_actual_vs_predicted(actual, predicted, model_name, aggregation_level, start_date=None):
    """
    Enhanced version with proper date handling
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    model_name : str
        Name of model for title
    aggregation_level : str
        "Daily", "Weekly", or "Monthly"
    start_date : datetime, optional
        Start date for the time series
    """
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
    
    # Generate dates if not provided
    if start_date is None:
        dates = list(range(len(actual)))  # Fallback to indices
    else:
        dates = generate_dates(
            start_date=start_date,
            end_date=None,
            aggregation_level=aggregation_level,
            num_periods=len(actual)
        )
    
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Configure x-axis based on aggregation level
    xaxis_config = {
        "type": "date" if start_date else "linear",
        "tickmode": "auto",
        "nticks": min(20, len(dates)),
        "showgrid": True,
        "tickangle": 45
    }
    
    if start_date:  # Only apply date formatting if we have real dates
        if aggregation_level == "Daily":
            xaxis_config.update({
                "tickformat": "%b %d, %Y",
                "hoverformat": "%a, %b %d, %Y"
            })
        elif aggregation_level == "Weekly":
            xaxis_config.update({
                "tickformat": "Week of %b %d, %Y",
                "hoverformat": "Week %U, %Y"
            })
        else:  # Monthly
            xaxis_config.update({
                "tickformat": "%b %Y",
                "hoverformat": "%B %Y"
            })
    
    fig.update_layout(
        title=f"{aggregation_level} Actual vs Predicted ({model_name})",
        xaxis_title="Time Period",
        yaxis_title="Value",
        xaxis=xaxis_config,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        height=500,
        margin=dict(t=50, b=100)
    )
    
    return fig




def generate_comparison_plot(metrics_dict):
    """
    Generate a comparison plot of evaluation metrics across models using Plotly.
    """
    metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
    metrics_df.rename(columns={"index": "Model"}, inplace=True)

    # Create Plotly figure
    fig = go.Figure()

    for metric in ["MAPE", "SMAPE", "RMSE", "MAE"]:
        fig.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df[metric],
            name=metric
        ))

    # Update layout
    fig.update_layout(
        title="Model Comparison",
        xaxis_title="Model",
        yaxis_title="Metric Value",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
        height=400
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)