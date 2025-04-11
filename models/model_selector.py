# Recommends the best model based on evaluation metrics
def select_best_model(metrics_dict):
    """
    Select the best model based on evaluation metrics (e.g., MAPE, SMAPE, RMSE)..
    Parameters
    ----------
    metrics_dict : Dict
        Dictionary containing metrics for each model.
    Returns
    -------
    str
        Name of the best model.
    """
    # Example: Select the model with the lowest SMAPE
    best_model = min(metrics_dict, key=lambda k: metrics_dict[k]["SMAPE"])
    return best_model