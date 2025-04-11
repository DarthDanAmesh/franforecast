# Utility functions used across the data preparation steps

import streamlit as st

def display_message(cols_removed):
    """
    Display a message in the Streamlit dashboard if columns were removed.
    Parameters
    ----------
    cols_removed : list
        List of columns that have been removed.
    """
    if cols_removed:
        st.warning(f"The following columns were removed: {', '.join(cols_removed)}")
    else:
        st.info("No columns were removed.")

def stop_if_not_enough_rows(df, min_rows):
    """
    Stop the Streamlit app if the dataframe does not have enough rows.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    min_rows : int
        Minimum number of rows required.
    """
    if len(df) < min_rows:
        st.error(f"Not enough rows in the dataset. Minimum required: {min_rows}")
        st.stop()