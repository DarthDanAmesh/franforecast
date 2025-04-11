# Saves and loads trained models
import joblib
import os

def save_model(model, filepath):
    """
    Save a trained model to disk.
    Parameters
    ----------
    model : object
        Trained model (e.g., Random Forest, XGBoost, DeepAR).
    filepath : str
        Path to save the model.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the model using joblib
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load a trained model from disk.
    Parameters
    ----------
    filepath : str
        Path to the saved model.
    Returns
    -------
    object
        Loaded model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    
    # Load the model using joblib
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model