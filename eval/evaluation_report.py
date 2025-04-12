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
    Enhanced plotting that properly handles DeepAR's single prediction output
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
    
    # Create figure with improved layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot - Actual vs Predicted
    ax1.plot(y_true, label="Actual", color="#1f77b4", linewidth=2, marker='o', markersize=5)
    ax1.plot(y_pred, label="Predicted", color="#ff7f0e", linewidth=2, 
            linestyle='--', marker='x', markersize=5)
    ax1.set_title(f"Actual vs Predicted ({model_name})", fontsize=14, pad=10)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Error plot below
    errors = y_true - y_pred
    ax2.plot(errors, label="Error", color="#d62728", linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel("Time Period", fontsize=12)
    ax2.set_ylabel("Error", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show data table with statistics
    st.subheader("Prediction Data")
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