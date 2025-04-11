# Generates forecasts using the selected model

def generate_forecast(model, test_data, is_gluonts=False):
    """
    Generate forecasts using the selected model.
    
    Parameters
    ----------
    model : object
        Trained model (Random Forest, XGBoost, TFT, or GluonTS model).
    test_data : pd.DataFrame, TimeSeriesDataSet, or ListDataset
        Test data in the appropriate format for the model.
    is_gluonts : bool
        Whether the model is a GluonTS model (e.g., DeepAR).
        
    Returns
    -------
    np.array or list
        Forecasted values.
    """
    # Import needed here to avoid circular imports
    from pytorch_forecasting.models import TemporalFusionTransformer
    
    # Case 1: GluonTS models (e.g., DeepAR)
    if is_gluonts:
        forecast_it, *_ = model.predict(test_data)  # Fixed syntax with *_ to unpack rest
        forecasts = list(forecast_it)
        return [f.mean for f in forecasts]
    
    # Case 2: PyTorch Forecasting models (e.g., TFT)
    elif isinstance(model, TemporalFusionTransformer):
        # Convert to dataloader if needed
        if hasattr(test_data, 'to_dataloader'):
            dataloader = test_data.to_dataloader(train=False, batch_size=64)
        else:
            # Assuming it's already a dataloader
            dataloader = test_data
            
        # Get predictions - TFT returns quantile predictions
        predictions = model.predict(dataloader)
        
        # Return median predictions (0.5 quantile)
        return predictions.median().detach().cpu().numpy()
    
    # Case 3: Handle other PyTorch models like N-BEATS or DeepAR if using pytorch_forecasting
    elif hasattr(model, 'predict') and hasattr(model, 'training'):  # Likely a PyTorch model
        # Similar to TFT case
        if hasattr(test_data, 'to_dataloader'):
            dataloader = test_data.to_dataloader(train=False, batch_size=64)
        else:
            dataloader = test_data
            
        predictions = model.predict(dataloader)
        
        # Check if predictions is a distribution or direct values
        if hasattr(predictions, 'mean'):
            return predictions.mean().detach().cpu().numpy()
            #return predictions.mean.detach().cpu().numpy()
        else:
            return predictions.detach().cpu().numpy()
    
    # Case 4: Standard ML models (Random Forest, XGBoost)
    else:
        return model.predict(test_data)