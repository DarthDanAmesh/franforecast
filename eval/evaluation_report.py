# Generates summary reports and visualizations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.date_utils import generate_dates

_ = np

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


#compare actuals and predicted
def plot_actual_vs_predicted_deprecatedon14th(actual, predicted, model_name, aggregation_level, start_date=None):
    """
    Enhanced version that properly handles all aggregation levels
    """
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
    
    # Create date sequence
    if start_date is None:
        dates = list(range(len(actual)))
    else:
        dates = pd.date_range(
            start=start_date,
            periods=len(actual),
            freq=aggregation_level[0]  # 'D', 'W', 'ME'
        )
    
    fig = go.Figure()
    
    # Add traces with different styling per aggregation
    line_width = 2 if aggregation_level == "Daily" else 3
    marker_size = 4 if aggregation_level == "Daily" else 6
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=line_width),
        marker=dict(size=marker_size)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=line_width, dash='dash'),
        marker=dict(size=marker_size)
    ))
    
    # Configure x-axis based on aggregation level
    xaxis_config = {
        "tickmode": "auto",
        "nticks": min(12, len(dates)),  # Fewer ticks for weekly/monthly
        "showgrid": True,
        "tickangle": 45
    }
    
    if start_date is not None:
        xaxis_config["type"] = "date"
        if aggregation_level == "Daily":
            xaxis_config.update({
                "tickformat": "%b %d\n%Y",
                "hoverformat": "%A, %b %d, %Y"
            })
        elif aggregation_level == "Weekly":
            xaxis_config.update({
                "tickformat": "Week %U\n%Y",
                "hoverformat": "Week %U (%b %d-%d), %Y"
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=500,
        margin=dict(t=50, b=100)
    )
    
    return fig



def plot_actual_vs_predicted(actual, predicted, model_name, aggregation_level, start_date=None):
    """
    Enhanced version that properly handles all aggregation levels and different forecast shapes
    """
    # Handle 2D predicted arrays by selecting appropriate values
    if isinstance(predicted, np.ndarray) and predicted.ndim > 1:
        # If predicted is 2D, we need to decide which values to use
        # Option 1: Take the first forecast for each time step
        predicted_1d = predicted[:, 0]
        
        # Option 2: If the shape doesn't match expected pattern, try flattening or reshaping
        if len(predicted_1d) != len(actual) and predicted.size >= len(actual):
            # Try to reshape to match actual length
            predicted_1d = predicted.flatten()[:len(actual)]
    else:
        predicted_1d = predicted
    
    if len(actual) != len(predicted_1d):
        min_len = min(len(actual), len(predicted_1d))
        actual = actual[:min_len]
        predicted_1d = predicted_1d[:min_len]
   
    # Create date sequence
    if start_date is None:
        dates = list(range(len(actual)))
    else:
        freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
        freq = freq_map.get(aggregation_level, 'D')
        dates = pd.date_range(
            start=start_date,
            periods=len(actual),
            freq=freq
        )
   
    fig = go.Figure()
   
    # Add traces with different styling per aggregation
    line_width = 2 if aggregation_level == "Daily" else 3
    marker_size = 4 if aggregation_level == "Daily" else 6
   
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=line_width),
        marker=dict(size=marker_size)
    ))
   
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted_1d,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=line_width, dash='dash'),
        marker=dict(size=marker_size)
    ))
   
    # Configure x-axis based on aggregation level
    xaxis_config = {
        "tickmode": "auto",
        "nticks": min(12, len(dates)),  # Fewer ticks for weekly/monthly
        "showgrid": True,
        "tickangle": 45
    }
   
    if start_date is not None:
        xaxis_config["type"] = "date"
        if aggregation_level == "Daily":
            xaxis_config.update({
                "tickformat": "%b %d\n%Y",
                "hoverformat": "%A, %b %d, %Y"
            })
        elif aggregation_level == "Weekly":
            xaxis_config.update({
                "tickformat": "Week %U\n%Y",
                "hoverformat": "Week %U (%b %d-%d), %Y"
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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