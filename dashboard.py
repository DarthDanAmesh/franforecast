import streamlit as st
import pandas as pd
from data_prep.data_loader import load_data, preview_data
from data_prep.data_cleaner import clean_data
from data_prep.data_transformer import format_date_column, one_hot_encode
from data_prep.holiday_utils import get_holidays
from models.model_trainer import train_random_forest, train_xgboost, train_tft
from models.model_saver import save_model, load_model
from models.forecast_generator import generate_forecast
from models.utils import prepare_time_series_dataset
from eval.metric_calculator import calculate_all_metrics
from eval.evaluation_report import generate_summary_report, plot_actual_vs_predicted
from input.input_handler import get_user_input, validate_inputs
from utils.date_utils import add_time_features
from utils.streamlit_utils import display_message, stop_if_invalid
from reports.report_generator import generate_html_report, generate_streamlit_report


# Configuration
config = {
    "min_rows": 100,
    "default_aggregation": "Monthly",
    "prediction_length": 6,
    "model_params": {
        "RandomForest": {"n_estimators": 100},
        "XGBoost": {"n_estimators": 100, "learning_rate": 0.1},
        "TFT": {
            "max_epochs": 50,
            "learning_rate": 0.03,
            "hidden_size": 160,
            "attention_head_size": 4,
            "dropout": 0.1,
            "batch_size": 64
        }
    }
}

def main():
    st.title("Forecasting Application")
    
    # Step 1: Upload Data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if not uploaded_file:
        st.stop()

    # Step 2: Load Data (now with standardized column names)
    df = load_data(uploaded_file)
    preview_data(df)
    if df is None:
        st.stop()
        
    # Step 3: Verify we have required columns
    required_columns = {'Date', 'target'}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Step 4: Get User Inputs
    user_inputs = get_user_input()
    validation_result = validate_inputs(user_inputs)
    stop_if_invalid(validation_result)

    # Step 5: Filter Data
    try:
        # Convert user dates to datetime
        start_date = pd.to_datetime(user_inputs["start_date"])
        end_date = pd.to_datetime(user_inputs["end_date"])
        
        # Filter by date range
        date_mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df = df.loc[date_mask].copy()
        
        # Filter by item code if specified
        if user_inputs["item_code"] and 'Item_code' in df.columns:
            df = df[df['Item_code'] == user_inputs["item_code"]]
            
    except Exception as e:
        st.error(f"Data filtering failed: {str(e)}")
        st.stop()

    # Add time features and drop the original "Date" column
    df = add_time_features(df, date_col='Date')
    df = df.drop(columns=["Date"], errors='ignore')  # Drop the original "Date" column

    # Step 7: Train-Test Split
    train_size = int(len(df) * 0.8)
    train_df, test_df = df[:train_size], df[train_size:]

    # Step 8: Model Training
    selected_model = user_inputs["selected_model"]

    # Validate selected_model
    allowed_models = {"Random Forest", "XGBoost", "TFT"}
    if selected_model not in allowed_models:
        st.error(f"Invalid model selection: {selected_model}. Please choose from {allowed_models}.")
        st.stop()

    try:
        if selected_model == "Random Forest":
            # Select only numeric columns for Random Forest
            X_train = train_df.select_dtypes(include=['number']).drop(columns=["target"], errors='ignore')
            model = train_random_forest(
                X_train,
                train_df["target"],
                config["model_params"]["RandomForest"]
            )
        elif selected_model == "XGBoost":
            # Similar preprocessing for XGBoost
            X_train = train_df.select_dtypes(include=['number']).drop(columns=["target"], errors='ignore')
            model = train_xgboost(
                X_train,
                train_df["target"],
                config["model_params"]["XGBoost"]
            )
        elif selected_model == "TFT":
            train_dataset = prepare_time_series_dataset(
                train_df,
                target_col="target",
                date_col="Date",
                freq=user_inputs["aggregation_level"][0],
                prediction_length=config["prediction_length"]
            )
            model = train_tft(train_dataset, **config["model_params"]["TFT"])
            # Step 9: Save Model
            save_model(model, f"{selected_model.lower()}_model.pkl")
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        st.stop()
    

    # Step 10: Generate Forecasts
    try:
        if selected_model in ["Random Forest", "XGBoost"]:
            # Select only numeric columns for forecasting
            X_test = test_df.select_dtypes(include=['number']).drop(columns=["target"], errors='ignore')
            forecasts = generate_forecast(model, X_test)
        else:
            test_dataset = prepare_time_series_dataset(
                test_df,
                target_col="target",
                date_col="Date",
                freq=user_inputs["aggregation_level"][0],
                prediction_length=config["prediction_length"]
            )
            forecasts = generate_forecast(model, test_dataset)
    except Exception as e:
        st.error(f"Forecast generation failed: {str(e)}")
        st.stop()

    # Step 11: Evaluation
    metrics = calculate_all_metrics(test_df["target"], forecasts)


    # Convert metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics, index=[0])  # Ensure it's a DataFrame with one row



    st.dataframe(generate_summary_report({selected_model: metrics}))
    plot_actual_vs_predicted(test_df["target"], forecasts, selected_model)

    # Generate reports
    #generate_html_report(metrics_df, "report.html")# uncomment only if you want to have the html generated. It is not the prettiest of htmls
    generate_streamlit_report(metrics, forecasts)

    display_message("Forecasting completed successfully!", "success")

if __name__ == "__main__":
    main()