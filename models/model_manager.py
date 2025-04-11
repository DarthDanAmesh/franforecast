import joblib
import torch
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
import hashlib
import json
import pandas as pd
import torch.serialization

class ModelManager:
    def __init__(self, model_dir="saved_models", tft_subdir="tft_models"):
        self.model_dir = Path(model_dir)
        self.tft_dir = self.model_dir / tft_subdir
        self.model_dir.mkdir(exist_ok=True)
        self.tft_dir.mkdir(exist_ok=True)

    def get_fingerprint(self, df, config, user_inputs):
        """Generate a unique fingerprint based on data and config"""
        def _json_safe(obj):
            if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return obj.isoformat()
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            return str(obj)

        data_str = df.to_json() + json.dumps(config, default=_json_safe) + json.dumps(user_inputs, default=_json_safe)

        return hashlib.md5(data_str.encode()).hexdigest()

    def get_model_cache_path(self, model_name):
        return self.model_dir / f"{model_name.lower()}_cache.pkl"

    def load_cached_model(self, fingerprint, model_name):
        cache_file = self.get_model_cache_path(model_name)
        if not cache_file.exists():
            return None

        try:
            cached = joblib.load(cache_file)
            if cached.get("fingerprint") == fingerprint:
                return cached.get("model")
        except Exception as e:
            print(f"Error loading cached model: {e}")
        return None

    def save_cached_model(self, model, fingerprint, model_name):
        cache_data = {
            "model": model,
            "fingerprint": fingerprint,
            "timestamp": pd.Timestamp.now()
        }
        cache_file = self.get_model_cache_path(model_name)
        try:
            joblib.dump(cache_data, cache_file)
        except Exception as e:
            print(f"Error saving cached model: {e}")

    # --- TFT specific handling ---
    def save_tft_model(self, model, train_dataset, filename="tft_model.pt"):
        if not isinstance(train_dataset, TimeSeriesDataSet):
            raise ValueError(f"Expected TimeSeriesDataSet, got {type(train_dataset)}")
        
        save_data = {
            'model_state': model.state_dict(),
            'training_config': train_dataset.get_parameters()
        }
        torch.save(save_data, self.tft_dir / filename)
        
    

    def load_tft_model(self, filename, train_dataset):
        path = self.tft_dir / filename
        if not path.exists():
            return None
        
        try:
            # Load the saved data containing state dict and training config
            save_data = torch.load(path)
            
            # Recreate the model using the training dataset
            model = TemporalFusionTransformer.from_dataset(
                train_dataset,
                **save_data.get('training_config', {})
            )
            
            # Load the state dictionary into the model
            model.load_state_dict(save_data['model_state'])
            
            return model
        except Exception as e:
            print(f"Error loading TFT model: {e}")
            return None

