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
    Calculate metrics with shape validation
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle different shapes
    if y_pred.ndim > 1:
        # If predictions are 2D (n_samples, horizon), use first step
        y_pred = y_pred[:, 0]
    
    # Validate shapes match
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    # Filter zeros for percentage errors
    nonzero_mask = y_true != 0
    y_true_nonzero = y_true[nonzero_mask]
    y_pred_nonzero = y_pred[nonzero_mask]
    
    return {
        "MAPE": calculate_mape(y_true_nonzero, y_pred_nonzero) if len(y_true_nonzero) > 0 else np.nan,
        "SMAPE": calculate_smape(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred)
    }