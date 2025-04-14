import numpy as np
import torch
def generate_forecast(model, test_data, is_gluonts=False):
    """
    Generate forecasts with proper shape handling for DeepAR
    """
    from pytorch_forecasting.models import DeepAR
    
    if isinstance(model, DeepAR):
        try:
            # Convert to dataloader if needed
            if hasattr(test_data, 'to_dataloader'):
                dataloader = test_data.to_dataloader(
                    train=False, 
                    batch_size=64, 
                    num_workers=0
                )
            else:
                dataloader = test_data
            
            # Get predictions - returns (n_samples, prediction_length)
            predictions = model.predict(dataloader)

            if len(predictions) == 1 and len(test_data)>1:
                return np.tile(predictions[0], len(test_data))
            
            # Convert to 1D array by taking the first prediction step
            # Alternatively use mean/median across prediction horizon
            return predictions.flatten()
            
        except Exception as e:
            print(f"[ERROR] DeepAR prediction failed: {str(e)}")
            raise
    
    elif is_gluonts:
        forecast_it, _ = model.predict(test_data)
        return np.array([f.mean for f in forecast_it])
    
    elif hasattr(model, 'predict'):
        return model.predict(test_data)
    
    raise ValueError("Unsupported model type")


def chronos_predict(model, data, prediction_length, quantiles=[0.1, 0.5, 0.9]):
    """
    Make predictions with a Chronos model.
    """
    try:
        predictions = {}
        num_samples = 20  # Number of samples for probabilistic forecast
        
        # Convert DataFrame to Chronos expected format
        if 'unique_id' not in data.columns:
            # Single time series - convert to numpy array
            ts_values = data['value'].values.astype(np.float32)
            # Convert to PyTorch tensor
            ts_tensor = torch.from_numpy(ts_values)
            # Generate samples
            forecast_samples = model.predict(
                ts_tensor,
                prediction_length,
                num_samples=num_samples
            )
            # Calculate quantiles from samples
            predictions["0"] = {
                q: np.quantile(forecast_samples.cpu().numpy(), q=q, axis=0)
                for q in quantiles
            }
        else:
            # Multiple time series
            for uid, group in data.groupby('unique_id'):
                ts_values = group['value'].values.astype(np.float32)
                ts_tensor = torch.from_numpy(ts_values)
                forecast_samples = model.predict(
                    ts_tensor,
                    prediction_length,
                    num_samples=num_samples
                )
                predictions[str(uid)] = {
                    q: np.quantile(forecast_samples.cpu().numpy(), q=q, axis=0)
                    for q in quantiles
                }
                
        return predictions
        
    except Exception as e:
        print(f"Chronos prediction failed: {str(e)}")
        raise