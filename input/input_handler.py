# Handles user inputs for filtering, aggregation, and model selection
import streamlit as st
from datetime import date

def get_user_input():
    """
    Collect user inputs via Streamlit widgets.
    Returns
    -------
    Dict
        Dictionary containing user inputs (e.g., filters, aggregation level, model selection).
    """
    st.sidebar.header("Data Filtering Options")
    item_code = st.sidebar.text_input("Enter Item Code (optional):", "").strip()
    start_date = st.sidebar.date_input("Start Date:", date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date:", date.today())

    st.sidebar.header("Model Selection")
    model_options = ["Random Forest", "XGBoost", "TFT"]  # Updated to match allowed models
    selected_model = st.sidebar.selectbox("Select Model:", model_options)

    st.sidebar.header("Aggregation Options")
    aggregation_level = st.sidebar.selectbox(
        "Select Aggregation Level:",
        ["Daily", "Weekly", "Monthly"]
    )

    return {
        "item_code": item_code,
        "start_date": start_date,
        "end_date": end_date,
        "selected_model": selected_model,
        "aggregation_level": aggregation_level
    }

def validate_inputs(user_inputs):
    """
    Validate user inputs to ensure they are compatible with the system.
    Parameters
    ----------
    user_inputs : Dict
        Dictionary containing user inputs.
    Returns
    -------
    bool
        Whether the inputs are valid.
    str
        Error message if validation fails.
    """
    if user_inputs["start_date"] > user_inputs["end_date"]:
        return False, "Start date cannot be later than end date."
    
    # Optional: Add more validations (e.g., item_code format)
    if user_inputs["item_code"] and not user_inputs["item_code"].isalnum():
        return False, "Item code must be alphanumeric."
    
    return True, ""