from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
import pandas as pd
import numpy as np


def preprocess_test_df(test_df, full_df, date_col="Date"):
    test_df = test_df.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    test_df = test_df.sort_values(date_col).reset_index(drop=True)
    test_df["time_idx"] = (test_df[date_col] - full_df[date_col].min()).dt.days
    test_df["time_idx"] = test_df["time_idx"].astype(np.int64)
    test_df["group_id"] = test_df.get("Item_code", "0")
    return test_df


def prepare_time_series_dataset(df, target_col, date_col, freq, prediction_length, max_encoder_length=60):
    """
    Robust time series dataset preparation with proper DeepAR configuration
    """
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # 1. Ensure proper datetime conversion and sorting
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # 2. Create proper integer time index
    df["time_idx"] = (df[date_col] - df[date_col].min()).dt.days
    df["time_idx"] = df["time_idx"].astype(np.int64)
    
    # 3. Handle group IDs - DeepAR requires this
    group_id_col = "group_id"
    if 'Item_code' in df.columns:
        df[group_id_col] = df['Item_code'].astype(str)
    else:
        df[group_id_col] = "0"  # Single group if no item codes
    
    # 4. Calculate training cutoff (leave room for prediction)
    training_cutoff = df["time_idx"].max() - prediction_length
    
    # 5. Create the dataset with proper variable specification
    dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=[group_id_col],
        min_encoder_length=max(max_encoder_length // 2, 1),
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=prediction_length,
        
        # Critical configuration for DeepAR
        static_categoricals=[group_id_col],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[target_col],
        
        # Proper normalization configuration
        target_normalizer=GroupNormalizer(
            groups=[group_id_col],
            method="robust",  # More robust to outliers than "standard"
            transformation="log1p"  # Ensures positive values for DeepAR
        ),
        
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )
    
    return dataset, df, training_cutoff

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