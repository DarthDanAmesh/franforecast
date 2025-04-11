# Utility functions used across the evaluation module

import pandas as pd

def add_time_information(df, date_col):
    """
    Add time information (day, week, quarter, year) to the DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Name of the date column.
    Returns
    -------
    pd.DataFrame
        DataFrame with additional time information columns.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df["Day"] = df[date_col].dt.day
    df["Week"] = df[date_col].dt.isocalendar().week
    df["Quarter"] = df[date_col].dt.quarter
    df["Year"] = df[date_col].dt.year
    return df

def format_metrics(metrics_dict, decimals=2):
    """
    Format metrics to a specified number of decimal places.
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing evaluation metrics.
    decimals : int
        Number of decimal places to round to.
    Returns
    -------
    dict
        Formatted metrics dictionary.
    """
    formatted_metrics = {}
    for model, metrics in metrics_dict.items():
        formatted_metrics[model] = {k: round(v, decimals) for k, v in metrics.items()}
    return formatted_metrics