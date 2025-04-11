# Computes evaluation metrics like MAPE, SMAPE, RMSE, etc.
import numpy as np
import pandas as pd

_ = pd

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth target series.
    y_pred : pd.Series or np.ndarray
        Prediction series.
    Returns
    -------
    float
        MAPE value (as a percentage).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth target series.
    y_pred : pd.Series or np.ndarray
        Prediction series.
    Returns
    -------
    float
        SMAPE value (as a percentage).
    """
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth target series.
    y_pred : pd.Series or np.ndarray
        Prediction series.
    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth target series.
    y_pred : pd.Series or np.ndarray
        Prediction series.
    Returns
    -------
    float
        MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))

def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics.
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth target series.
    y_pred : pd.Series or np.ndarray
        Prediction series.
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """
    return {
        "MAPE": calculate_mape(y_true, y_pred),
        "SMAPE": calculate_smape(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred)
    }