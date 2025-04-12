import numpy as np

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