# Formats date columns, aggregates data, and applies transformations

import pandas as pd
import streamlit as st

def format_date_column(df, date_col, freq="ME"):
    """
    Format the date column to datetime and resample the data.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the date column.
    freq : str
        Resampling frequency (e.g., "D" for daily, "W" for weekly, "M" for monthly).
    Returns
    -------
    pd.DataFrame
        Dataframe with formatted date column and resampled data.
    """
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df = df.resample(freq).sum()  # Aggregate by summing up values
        return df
    except Exception as e:
        st.error(f"Error formatting date column: {e}")
        return df

def one_hot_encode(df, cols):
    """
    Apply one-hot encoding to specified columns.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list
        List of columns to one-hot encode.
    Returns
    -------
    pd.DataFrame
        One-hot encoded dataframe.
    """
    return pd.get_dummies(df, columns=cols)