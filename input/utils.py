# Utility functions used across the input module

import streamlit as st

def display_message(message, type="info"):
    """
    Display a message in the Streamlit dashboard.
    Parameters
    ----------
    message : str
        Message to display.
    type : str
        Type of message ("info", "warning", "error").
    """
    if type == "info":
        st.info(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)

def stop_if_invalid(validation_result):
    """
    Stop the Streamlit app if the validation result indicates invalid inputs.
    Parameters
    ----------
    validation_result : Tuple[bool, str]
        Validation result (validity flag, error message).
    """
    is_valid, error_message = validation_result
    if not is_valid:
        st.error(error_message)
        st.stop()