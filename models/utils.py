from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd

def prepare_time_series_dataset(df, target_col, date_col, freq, prediction_length, max_encoder_length=60):
    """
    Prepare a PyTorch Forecasting-compatible dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target column.
    date_col : str
        Name of the date column.
    freq : str
        Frequency of the time series (e.g., "M" for monthly).
    prediction_length : int
        Length of the forecast horizon.
    max_encoder_length : int
        Maximum length of the lookback window.
        
    Returns
    -------
    TimeSeriesDataSet
        PyTorch Forecasting-compatible dataset.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure the date column is in datetime format
    df.loc[:, date_col] = pd.to_datetime(df[date_col])
    
    # Format frequency properly (add "1" if it's just a letter)
    if len(freq) == 1 or not any(char.isdigit() for char in freq):
        freq = "1" + freq
    
    # Sort by date to ensure correct time sequence
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Add a time index column properly - this might have gaps
    time_delta = pd.Timedelta(freq)
    df.loc[:, "time_idx"] = ((df[date_col] - df[date_col].min()) / time_delta).astype(int)
    
    # Define group IDs (if applicable) - use actual grouping if available
    if 'Item_code' in df.columns:
        df.loc[:, "group_id"] = df['Item_code'].astype(str)
    else:
        df.loc[:, "group_id"] = "0"  # Default when no grouping is available
    
    # Check which columns are actually available in the dataframe
    available_columns = set(df.columns)
    
    # Define known categoricals and reals based on what's available
    known_categoricals = [col for col in ["holiday"] if col in available_columns]
    known_reals = [col for col in ["month", "day", "dayofweek", "year"] if col in available_columns]
    
    # Add any additional available numeric columns that might be useful
    for col in available_columns:
        if col not in [date_col, target_col, "time_idx", "group_id"] + known_categoricals + known_reals:
            if pd.api.types.is_numeric_dtype(df[col]):
                known_reals.append(col)
    
    # Create the dataset with proper parameters
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=known_categoricals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[target_col],
        target_normalizer="auto",  # Auto-select normalizer based on data
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,  # This is the key parameter that needs to be added
    )
    
    return dataset

def generate_forecast(model, test_data, is_gluonts=False):
    """
    Generate forecasts using the selected model.
    
    Parameters
    ----------
    model : object
        Trained model.
    test_data : pd.DataFrame or ListDataset
        Test data.
    is_gluonts : bool
        Whether the model is a GluonTS model (e.g., DeepAR).
        
    Returns
    -------
    pd.DataFrame or array
        Forecasted values.
    """
    if is_gluonts:
        forecast_it, _ = model.predict(test_data)
        forecasts = list(forecast_it)
        return [f.mean for f in forecasts]
    elif hasattr(model, 'predict'):
        # Standard sklearn models
        return model.predict(test_data)
    else:
        # PyTorch-Forecasting TFT model
        predictions = model.predict(test_data, return_index=True, return_decoder_lengths=True)
        # For TFT models, we typically want the mean prediction
        if hasattr(predictions, 'mean'):
            return predictions.mean
        return predictions