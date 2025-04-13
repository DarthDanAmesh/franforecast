import os
import pickle
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer
import torch
import hashlib
import json
import pandas as pd


def model_cache_path(model_name):
    return Path("saved_models") / f"{model_name.lower()}_cache.pkl"

def save_model_with_fingerprint(model, fingerprint, model_name):
    cache_data = {
        "model": model,
        "fingerprint": fingerprint,
        "timestamp": pd.Timestamp.now()
    }
    return save_model(cache_data, model_cache_path(model_name))

def load_model_with_fingerprint(fingerprint, model_name):
    cache = load_model(model_cache_path(model_name))
    if cache and cache.get("fingerprint") == fingerprint:
        return cache.get("model")
    return None


def get_data_fingerprint(df, config, user_inputs):
    """Create a unique hash representing the data and configuration"""
    data_str = df.to_json() + json.dumps(config) + json.dumps(user_inputs)
    return hashlib.md5(data_str.encode()).hexdigest()


def save_tft_model(model, filename):
    """Special handling for TFT models"""
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
    models_dir = Path("saved_models/tft_models")
    models_dir.mkdir(exist_ok=True)
    
    # Save both model and training state
    save_data = {
        'model_state': model.state_dict(),
        'training_config': model.get_parameters()
    }
    
    with open(models_dir / filename, 'wb') as f:
        torch.save(save_data, f)

def load_tft_model(filename, train_dataset=None):
    """Load TFT model with optional re-initialization"""
    filepath = Path("saved_models/tft_models") / filename
    
    if not filepath.exists():
        return None
        
    saved_data = torch.load(filepath)
    
    if train_dataset:
        # Reinitialize model if we have the dataset
        model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            **saved_data['training_config']
        )
        model.load_state_dict(saved_data['model_state'])
        return model
    else:
        return saved_data


def save_model(model, filename):
    """
    Save model to disk with proper path handling
    """
    try:
        # Create models directory if it doesn't exist
        models_dir = Path("saved_models")
        models_dir.mkdir(exist_ok=True)
        
        # Construct full filepath
        filepath = models_dir / filename
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_model(filename):
    """
    Load model from disk with proper path handling
    """
    try:
        filepath = Path("saved_models") / filename
        
        if not filepath.exists():
            return None
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None