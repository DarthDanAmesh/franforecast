# Handles training of models

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, QuantileLoss
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder, TorchNormalizer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import DeepAR, NormalDistributionLoss
import lightning.pytorch as pl
import pandas as pd

_ = pd


# Add to your imports section:


# Add DeepAR parameters to your config
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
        },
        "DeepAR": {
            "max_epochs": 100,
            "learning_rate": 1e-4,
            "hidden_size": 32,
            "dropout": 0.1,
            "batch_size": 64
        }
    }
}


def train_deepar(train_df, full_data, training_cutoff, context_length, prediction_length, **kwargs):
    """
    Train a DeepAR model for time series forecasting.
    """
    try:
        # Ensure we have proper group_ids
        if "group_id" not in train_df.columns:
            train_df["group_id"] = 0  # Default group if not specified
            
        # 1. Create training dataset with proper configuration
        training_dataset = TimeSeriesDataSet(
            train_df.loc[train_df.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            min_encoder_length=max(1, context_length // 2),  # More flexible encoder length
            max_encoder_length=context_length,
            min_prediction_length=1,
            max_prediction_length=prediction_length,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["target"],
            target_normalizer=TorchNormalizer(method="standard"),  # Changed normalizer
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None,
            allow_missing_timesteps=True  # Changed to False for more strict checking
        )        
        # 2. Create validation dataset
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            full_data,
            min_prediction_idx=training_cutoff + 1
        )
        
        # 3. Create dataloaders
        batch_size = kwargs.get("batch_size", 64)
        train_dataloader = training_dataset.to_dataloader(
            train=True, 
            batch_size=batch_size, 
            num_workers=0
        )
        val_dataloader = validation_dataset.to_dataloader(
            train=False, 
            batch_size=batch_size, 
            num_workers=0
        )
        
        # 4. Initialize DeepAR model
        pl.seed_everything(42)  # For reproducibility
        model = DeepAR.from_dataset(
            training_dataset,
            hidden_size=kwargs.get("hidden_size", 32),
            dropout=kwargs.get("dropout", 0.1),
            loss=NormalDistributionLoss(),
            learning_rate=kwargs.get("learning_rate", 1e-4),
            log_interval=10,
            log_val_interval=1,
        )
        
        # 5. Configure trainer with callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            patience=5, 
            mode="min"
        )
        
        trainer = pl.Trainer(
            max_epochs=kwargs.get("max_epochs", 100),
            accelerator="auto",
            gradient_clip_val=0.1,
            callbacks=[
                early_stop_callback,
                LearningRateMonitor()
            ],
            enable_model_summary=True,
            enable_checkpointing=True,
        )
        
        # 6. Train the model
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        return model
        
    except Exception as e:
        print(f"DeepAR training failed: {str(e)}")
        print("Debug info:")
        print(f"Training cutoff: {training_cutoff}")
        print(f"Time index dtype: {full_data['time_idx'].dtype}")
        print(f"Time index range: {full_data['time_idx'].min()} to {full_data['time_idx'].max()}")
        raise

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