# Utility functions for formatting and exporting reports

import pandas as pd

_ = pd

def format_metrics_for_report(metrics_df, decimals=2):
    """
    Format metrics for inclusion in a report.
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing evaluation metrics.
    decimals : int
        Number of decimal places to round to.
    Returns
    -------
    pd.DataFrame
        Formatted metrics dataframe.
    """
    return metrics_df.round(decimals)

def save_dataframe_to_csv(df, filename="data.csv"):
    """
    Save a dataframe to a CSV file.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    filename : str
        Name of the output CSV file.
    """
    df.to_csv(filename, index=False)