# Generates forecasts using the selected model

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
    pd.DataFrame
        Forecasted values.
    """
    if is_gluonts:
        forecast_it, _ = model.predict(test_data)
        forecasts = list(forecast_it)
        return [f.mean for f in forecasts]
    else:
        return model.predict(test_data)