# Generates summary reports and visualizations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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
    Robust plotting of actual vs predicted values with proper type handling
    """
    # Convert to numpy arrays if not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle different shapes
    if y_pred.ndim > 1:
        # If predictions are 2D (n_samples, horizon), use first step
        y_pred = y_pred[:, 0]
    
    # Ensure equal lengths
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[-min_length:]  # Take most recent values
    y_pred = y_pred[-min_length:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with improved styling
    ax.plot(y_true, label="Actual", color="#1f77b4", linewidth=2)
    ax.plot(y_pred, label="Predicted", color="#ff7f0e", linewidth=2, linestyle='--')
    
    # Formatting
    ax.set_title(f"Actual vs Predicted ({model_name})", fontsize=14, pad=20)
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Also show the data as a table without formatting
    st.subheader("Prediction Data")
    pred_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred,
        "Error": y_true - y_pred
    })
    st.dataframe(pred_df)

    
def generate_comparison_plot(metrics_dict):
    """
    Generate a comparison plot of evaluation metrics across models.
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing evaluation metrics for each model.
    """
    metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
    metrics_df.rename(columns={"index": "Model"}, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ["MAPE", "SMAPE", "RMSE", "MAE"]:
        ax.bar(metrics_df["Model"], metrics_df[metric], label=metric)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Metric Value")
    ax.legend()
    st.pyplot(fig)