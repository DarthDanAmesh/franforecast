# Handles training of models

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, QuantileLoss
from pytorch_forecasting.data.encoders import GroupNormalizer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import pandas as pd

# Remove these if present:
# from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger
# import pytorch_lightning as pl

_ = pd


config = {
    "min_rows": 100,
    "default_aggregation": "Monthly",
    "prediction_length": 6,
    "model_params": {
        "RandomForest": {"n_estimators": 100},
        "XGBoost": {"n_estimators": 100, "learning_rate": 0.1},
        "TFT": {
            "max_epochs": 50,
            "learning_rate": 0.03,
            "hidden_size": 160,
            "attention_head_size": 4,
            "dropout": 0.1,
            "batch_size": 64
        }
    }
}

def train_random_forest(X_train, y_train, params):
    """
    Train a Random Forest model.
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    params : Dict
        Model hyperparameters.
    Returns
    -------
    RandomForestRegressor
        Trained Random Forest model.
    """
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, params):
    """
    Train an XGBoost model.
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    params : Dict
        Model hyperparameters.
    Returns
    -------
    XGBRegressor
        Trained XGBoost model.
    """
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_tft(train_dataset, full_data, training_cutoff, **kwargs):
    """
    Final working version with proper Lightning compatibility
    """
    try:
        # 1. Create validation dataset
        val_data = full_data[full_data["time_idx"] > training_cutoff]
        val_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset,
            val_data,
            predict=True,
            stop_randomization=True
        )
        
        # 2. Create dataloaders with proper batch sizes
        batch_size = min(kwargs.get("batch_size", 64), len(train_dataset))
        train_dataloader = train_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )
        val_dataloader = val_dataset.to_dataloader(
            train=False,
            batch_size=batch_size * 2,  # Larger batch for validation
            num_workers=0
        )
        
        # 3. Initialize TFT model with proper parameters
        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            hidden_size=kwargs.get("hidden_size", 160),
            attention_head_size=kwargs.get("attention_head_size", 4),
            dropout=kwargs.get("dropout", 0.1),
            hidden_continuous_size=kwargs.get("hidden_continuous_size", 80),
            loss=QuantileLoss(),
            learning_rate=kwargs.get("learning_rate", 0.03),
            optimizer="Adam"
        )
        
        # 4. Configure trainer with proper callbacks
        trainer = pl.Trainer(
            max_epochs=kwargs.get("max_epochs", 50),
            accelerator="auto",
            gradient_clip_val=0.1,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min"),
                LearningRateMonitor()
            ],
            enable_model_summary=True,
            enable_checkpointing=True,
        )
        
        # 5. Train the model
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        return tft
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        print("Debug info:")
        print(f"Training cutoff: {training_cutoff}")
        print(f"Time index dtype: {full_data['time_idx'].dtype}")
        print(f"Time index range: {full_data['time_idx'].min()} to {full_data['time_idx'].max()}")
        raise