# Handles user inputs for filtering, aggregation, and model selection
import streamlit as st
import pandas as pd
from datetime import date

def get_user_input1():
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
    model_options = ["Random Forest", "XGBoost", "TFT", "DeepAR"]  # Updated to match allowed models
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


def get_user_input(df_columns):
    """
    Collect user inputs with flexible grouping based on available columns
    Parameters:
    -----------
    df_columns : list
        List of column names from your DataFrame
    
    Returns:
    --------
    Dict
        Dictionary containing all user inputs
    """
    st.sidebar.header("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", date.today())
    
    st.sidebar.header("🔍 Filter & Group")
    # Dynamic grouping options based on available columns
    available_groupings = [col for col in df_columns if col not in ['Date', 'target']]
    group_by = st.sidebar.multiselect(
        "Group by (optional):",
        options=available_groupings,
        default=None,
        help="Analyze trends by different categories"
    )
    
    item_filter = st.sidebar.text_input(
        "Filter by Item Code (optional):", 
        "",
        help="Leave blank to include all items"
    ).strip()
    
    st.sidebar.header("📊 Aggregation")
    aggregation_level = st.sidebar.radio(
        "Time Period:",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True
    )
    
    st.sidebar.header("🤖 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        ["Random Forest", "XGBoost", "TFT", "DeepAR"],
        index=0
    )
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "group_by": group_by,
        "item_filter": item_filter,
        "aggregation_level": aggregation_level,
        "selected_model": selected_model
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
    
    # Update this to use item_filter instead of item_code
    if "item_filter" in user_inputs and user_inputs["item_filter"] and not user_inputs["item_filter"].isalnum():
        return False, "Item code must be alphanumeric."
    
    return True, ""


def aggregate_data(df, aggregation_level, group_columns=None):
    """
    Enhanced aggregation that preserves grouping columns
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    if aggregation_level == "Daily":
        return df
    
    # Set date as index for resampling
    df.set_index('Date', inplace=True)
    
    # Determine aggregation columns - preserve all grouping columns
    agg_dict = {'target': 'sum'}  # Main aggregation
    if group_columns:
        agg_dict.update({col: 'first' for col in group_columns})
    
    if aggregation_level == "Weekly":
        agg_df = df.resample('W-MON').agg(agg_dict)
    elif aggregation_level == "Monthly":
        agg_df = df.resample('MS').agg(agg_dict)
    else:
        agg_df = df.resample('D').agg(agg_dict)
    
    return agg_df.reset_index()