# Generates summary reports and visualizations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go

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


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """
    Enhanced plotting that properly handles DeepAR's single prediction output using Plotly.
    """
    # Convert to numpy arrays if not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Special handling for DeepAR single prediction case
    if model_name == "DeepAR" and len(y_pred) == 1 and len(y_true) > 1:
        y_pred = np.full(len(y_true), y_pred[0])  # Repeat single prediction
        
    # Handle different shapes for other models
    elif y_pred.ndim > 1:
        # If predictions are 2D (n_samples, horizon), use first step
        y_pred = y_pred[:, 0]
    
    # Ensure equal lengths
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[-min_length:]  # Take most recent values
    y_pred = y_pred[-min_length:]
    
    # Calculate errors
    errors = y_true - y_pred
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add Actual vs Predicted traces
    fig.add_trace(go.Scatter(
        x=list(range(len(y_true))),
        y=y_true,
        mode='lines+markers',
        name='Actual',
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred))),
        y=y_pred,
        mode='lines+markers',
        name='Predicted',
        line=dict(color="#ff7f0e", width=2, dash='dash'),
        marker=dict(symbol='x', size=6)
    ))
    
    # Update layout for main plot
    fig.update_layout(
        title=f"Actual vs Predicted ({model_name})",
        xaxis_title="Time Period",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
        height=400
    )
    
    # Show the main plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Create an error plot
    error_fig = go.Figure()
    error_fig.add_trace(go.Scatter(
        x=list(range(len(errors))),
        y=errors,
        mode='lines+markers',
        name='Error',
        line=dict(color="#d62728", width=1.5)
    ))
    error_fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    # Update layout for error plot
    error_fig.update_layout(
        title="Prediction Errors",
        xaxis_title="Time Period",
        yaxis_title="Error",
        margin=dict(t=50),
        height=300
    )
    
    # Show the error plot
    st.plotly_chart(error_fig, use_container_width=True)
    st.subheader("Prediction Data")
    with st.expander("Table Data"):
    # Show data table with statistics
        
        pred_df = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred,
            "Error": errors
        })
    
        # Add summary statistics row
        stats_row = pd.DataFrame({
            "Actual": ["Mean: {:.2f}".format(np.mean(y_true))],
            "Predicted": ["Mean: {:.2f}".format(np.mean(y_pred))],
            "Error": ["Mean: {:.2f}".format(np.mean(errors))]
        })
        pred_df = pd.concat([pred_df, stats_row], ignore_index=True)
        
        st.dataframe(pred_df.style.apply(
            lambda x: ['background: lightyellow' if x.name == len(pred_df)-1 else '' for i in x],
            axis=1
        ))


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