# Generates summary reports and visualizations
import pandas as pd
import matplotlib.pyplot as plt
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
#st.dataframe(generate_summary_report({selected_model: metrics}))

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """
    Plot actual vs predicted values for a given model.
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth target series.
    y_pred : pd.Series or np.ndarray
        Prediction series.
    model_name : str
        Name of the model.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    st.pyplot(plt)

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