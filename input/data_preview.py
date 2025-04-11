# Allows users to preview the imported data and filter columns

import streamlit as st
import pandas as pd

def preview_and_filter_data(df):
    """
    Allow users to preview the data and select columns for filtering.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    Returns
    -------
    pd.DataFrame
        Filtered dataframe based on user selections.
    """
    st.write("Preview of the uploaded data:")
    st.dataframe(df.head())

    st.write("Select columns to include in the analysis:")
    columns_to_keep = st.multiselect(
        "Choose columns:",
        options=df.columns,
        default=list(df.columns)
    )

    filtered_df = df[columns_to_keep]
    st.write("Filtered Data Preview:")
    st.dataframe(filtered_df.head())

    return filtered_df