# Handles visualizations like actual vs predicted plots, regressor impact, etc.

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

_ = pd

def plot_actual_vs_predicted_deprecated(y_true, y_pred, title="Actual vs Predicted"):
    """
    Plot actual vs predicted values.
    Parameters
    ----------
    y_true : pd.Series or list
        Ground truth target series.
    y_pred : pd.Series or list
        Prediction series.
    title : str
        Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    st.pyplot(plt)

def plot_regressor_impact(df, regressor_col, target_col):
    """
    Plot the impact of a regressor on the target variable.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing regressor and target columns.
    regressor_col : str
        Name of the regressor column.
    target_col : str
        Name of the target column.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df[regressor_col], df[target_col], alpha=0.5)
    plt.title(f"Impact of {regressor_col} on {target_col}")
    plt.xlabel(regressor_col)
    plt.ylabel(target_col)
    st.pyplot(plt)

def plot_holiday_effect(df, holiday_col, target_col):
    """
    Plot the effect of holidays on the target variable.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing holiday and target columns.
    holiday_col : str
        Name of the holiday column.
    target_col : str
        Name of the target column.
    """
    holiday_effect = df.groupby(holiday_col)[target_col].mean()
    plt.figure(figsize=(10, 6))
    holiday_effect.plot(kind="bar", color="skyblue")
    plt.title(f"Holiday Effect on {target_col}")
    plt.xlabel("Holiday")
    plt.ylabel(f"Average {target_col}")
    st.pyplot(plt)