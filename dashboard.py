import streamlit as st
import pandas as pd
import traceback
from data_prep.data_loader import load_data, preview_data
from data_prep.data_cleaner import clean_data
from data_prep.data_transformer import format_date_column, one_hot_encode
from data_prep.holiday_utils import get_holidays
from models.model_trainer import train_random_forest, train_xgboost, train_tft, train_deepar
from models.model_saver import save_model, load_model, get_data_fingerprint, load_tft_model, save_tft_model
from models.forecast_generator import generate_forecast
from models.utils import prepare_time_series_dataset, preprocess_test_df
from eval.metric_calculator import calculate_all_metrics
from eval.evaluation_report import generate_summary_report, plot_actual_vs_predicted
from input.input_handler import get_user_input, validate_inputs
from utils.date_utils import add_time_features
from utils.streamlit_utils import display_message, stop_if_invalid
from reports.report_generator import generate_html_report, generate_streamlit_report
from models.model_manager import ModelManager
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


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
        },
        "DeepAR": {  # Add this section
            "max_epochs": 100,
            "learning_rate": 1e-4,
            "hidden_size": 32,
            "dropout": 0.1,
            "batch_size": 64
        }
    }
}




# In your main function:


def main():

    st.title("Forecasting Application")
    model_manager = ModelManager()

    use_saved_model = st.checkbox("Use saved model if available", value=True)
    model = None

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if not uploaded_file:
        st.stop()

    df = load_data(uploaded_file)
    preview_data(df.head())
    if df is None:
        st.stop()

    required_columns = {'Date', 'target'}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()
    
    user_inputs = get_user_input()
    validation_result = validate_inputs(user_inputs)
    stop_if_invalid(validation_result)

    try:
        start_date = pd.to_datetime(user_inputs["start_date"])
        end_date = pd.to_datetime(user_inputs["end_date"])
        date_mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df = df.loc[date_mask].copy()

        if user_inputs["item_code"] and 'Item_code' in df.columns:
            df = df[df['Item_code'] == user_inputs["item_code"]]

    except Exception as e:
        st.error(f"Data filtering failed: {str(e)}")
        st.stop()

    df = add_time_features(df, date_col='Date')

    train_size = int(len(df) * 0.8)
    train_df, test_df = df[:train_size], df[train_size:]

    selected_model = user_inputs["selected_model"]
    allowed_models = {"Random Forest", "XGBoost", "TFT", "DeepAR"}
    if selected_model not in allowed_models:
        st.error(f"Invalid model selection: {selected_model}. Please choose from {allowed_models}.")
        st.stop()

    current_fingerprint = model_manager.get_fingerprint(df, config, user_inputs)
    model = model_manager.load_cached_model(current_fingerprint, selected_model)

    if model:
        st.success("Loaded cached model matching current data configuration")

    if model is None:
        try:
            if selected_model == "Random Forest":
                X_train = train_df.select_dtypes(include=['number']).drop(columns=["target"], errors='ignore')
                model = train_random_forest(X_train, train_df["target"], config["model_params"]["RandomForest"])

            elif selected_model == "XGBoost":
                X_train = train_df.select_dtypes(include=['number']).drop(columns=["target"], errors='ignore')
                model = train_xgboost(X_train, train_df["target"], config["model_params"]["XGBoost"])

            elif selected_model == "TFT":
                train_dataset, full_df, cutoff = prepare_time_series_dataset(
                    df,
                    target_col="target",
                    date_col="Date",
                    freq=user_inputs["aggregation_level"][0],
                    prediction_length=config["prediction_length"]
                )

                tft_model = None
                if use_saved_model:
                    try:
                        tft_model = model_manager.load_tft_model("tft_model.pt", train_dataset)
                    except Exception as e:
                        st.warning(f"Failed to load saved TFT model: {e}")
                        tft_model = None

                if tft_model is None:
                    tft_model = train_tft(
                        train_dataset,
                        full_data=full_df,
                        training_cutoff=cutoff,
                        **config["model_params"]["TFT"]
                    )
                    model_manager.save_tft_model(tft_model, train_dataset, "tft_model.pt")

                model = tft_model

            elif selected_model == "DeepAR":
                if len(train_df) < config["prediction_length"] * 2:
                    st.error(f"Not enough data for DeepAR. Need at least {config['prediction_length'] * 2} samples.")
                    st.stop()
                    
                # Calculate context length more carefully
                context_length = min(
                    24, 
                    max(1, len(train_df) // 4),  # Ensure at least 1
                    len(train_df) - config["prediction_length"]
                )
                train_dataset, full_df, cutoff = prepare_time_series_dataset(
                    df,
                    target_col="target",
                    date_col="Date",
                    freq=user_inputs["aggregation_level"][0],
                    prediction_length=config["prediction_length"]
                )
                deepar_model = None
                if use_saved_model:
                    try:
                        deepar_model = model_manager.load_deepar_model("deepar_model.pt", train_dataset)
                    except Exception as e:
                        st.warning(f"Failed to load saved DeepAR model: {e}")
                        deepar_model = None
                if deepar_model is None:
                    # Determine context length based on data or config
                    context_length = min(24, len(train_dataset) // 4)  # Example: use 24 or 1/4 of dataset length
                    deepar_model = train_deepar(
                        train_df=full_df,
                        full_data=full_df,
                        training_cutoff=cutoff, 
                        context_length=context_length,
                        prediction_length=config["prediction_length"],
                        **config["model_params"]["DeepAR"]
                    )
                    model_manager.save_deepar_model(deepar_model, train_dataset, "deepar_model.pt")
                model = deepar_model
            model_manager.save_cached_model(model, current_fingerprint, selected_model)

        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            st.write("Details:")
            st.code(traceback.format_exc())
            st.stop()

    try:
        if selected_model in ["Random Forest", "XGBoost"]:
            X_test = test_df.select_dtypes(include=['number']).drop(columns=["target"], errors='ignore')
            forecasts = generate_forecast(model, X_test)
        else:
            # Preprocess test_df to match train_df (time_idx, group_id, etc.)
            test_df = preprocess_test_df(test_df, df)

            # Rename target column if needed
            #test_df.rename(columns={user_inputs["target_column"]: "target"}, inplace=True)


            # Safely get feature lists with fallback to empty list if None
            def safe_get(attr):
                return getattr(train_dataset, attr, []) or []

            required_columns = list(set(
                safe_get('reals') +
                safe_get('static_categoricals') +
                safe_get('time_varying_known_categoricals') +
                safe_get('time_varying_known_reals') +
                ['time_idx'] + safe_get('group_ids')
            ))

            test_df = test_df[[col for col in required_columns if col in test_df.columns]]

            # Create test dataset
            test_dataset = TimeSeriesDataSet.from_dataset(
                train_dataset,
                test_df,
                predict=True,
                stop_randomization=True
            )

            # Generate forecasts
            forecasts = generate_forecast(model, test_dataset)
    except Exception as e:
        st.error(f"Forecast generation failed: {str(e)}")
        st.write("Details:")
        st.code(traceback.format_exc())
        st.stop()

    try:
        # Ensure proper alignment of forecasts and actual values
        if selected_model in ["DeepAR", "TFT"]:
            # For time series models, we may need to align the predictions
            test_targets = test_df["target"].values[-len(forecasts):]
        else:
            test_targets = test_df["target"].values
        
        metrics = calculate_all_metrics(test_targets, forecasts)
        metrics_df = pd.DataFrame([metrics])
        metrics_df["Model"] = selected_model
        st.dataframe(metrics_df)

    except Exception as e:
        st.error(f"Metric calculation failed: {str(e)}")
        st.write("Shapes during error:")
        st.write(f"Actual values shape: {test_df['target'].shape}")
        st.write(f"Forecasts shape: {forecasts.shape}")
        st.code(traceback.format_exc())
        st.stop()

    st.dataframe(generate_summary_report({selected_model: metrics}))
    plot_actual_vs_predicted(test_df["target"], forecasts, selected_model)
    generate_streamlit_report(metrics, forecasts)

    display_message("Forecasting completed successfully!", "success")

if __name__ == "__main__":
    main()