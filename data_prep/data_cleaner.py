# Cleans the data (e.g., removing duplicates, handling missing values)
import pandas as pd
import numpy as np
import streamlit as st

def clean_data(df, cleaning_specs):
    """
    Clean the input dataframe according to cleaning specifications.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that has to be cleaned.
    cleaning_specs : Dict
        Cleaning specifications (e.g., columns to drop, fill NA methods).
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col, method in cleaning_specs.get("fill_na", {}).items():
        if method == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif method == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif method == "drop":
            df = df.dropna(subset=[col])

    # Apply log transformation if specified
    if cleaning_specs.get("log_transform", False):
        try:
            df["target"] = np.log1p(df["target"])
        except Exception as e:
            st.error(f"Error applying log transformation: {e}")

    return df