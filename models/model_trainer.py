# Handles training of models

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd

_ = pd

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

def train_tft(train_dataset, **kwargs):
    """
    Train a Temporal Fusion Transformer model.

    Parameters
    ----------
    train_dataset : TimeSeriesDataSet
        Training dataset.
    **kwargs : dict
        Additional parameters for model configuration:
        - batch_size (int): Batch size for training and validation. Default is 64.
        - hidden_size (int): Size of the hidden layers. Default is 160.
        - attention_head_size (int): Number of attention heads. Default is 4.
        - dropout (float): Dropout rate. Default is 0.1.
        - hidden_continuous_size (int): Size of hidden layers for continuous variables. Default is hidden_size // 2.
        - learning_rate (float): Learning rate. Default is 0.03.
        - patience (int): Early stopping patience. Default is 5.
        - max_epochs (int): Maximum number of epochs. Default is 50.
        - limit_train_batches (int or float): Limit the number of training batches per epoch. Default is None (full training).
        - gradient_clip_val (float): Gradient clipping value. Default is 0.1.

    Returns
    -------
    TemporalFusionTransformer
        Trained TFT model.
    """
    try:
        # Create train and validation dataloaders
        train_dataloader = train_dataset.to_dataloader(
            batch_size=kwargs.get("batch_size", 64),
            shuffle=True
        )
        
        validation = TimeSeriesDataSet.from_dataset(
            train_dataset,
            train_dataset.split_by_ratio(0.9)[1],  # 10% for validation
            predict=True,
            stop_randomization=True
        )
        val_dataloader = validation.to_dataloader(
            batch_size=kwargs.get("batch_size", 64),
            shuffle=False
        )
        
        # Configure the network
        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            hidden_size=kwargs.get("hidden_size", 160),
            attention_head_size=kwargs.get("attention_head_size", 4),
            dropout=kwargs.get("dropout", 0.1),
            hidden_continuous_size=kwargs.get("hidden_continuous_size", 160 // 2),
            learning_rate=kwargs.get("learning_rate", 0.03),
            log_interval=10,
            log_val_interval=1,
            optimizer="Adam"
        )
        
        # Configure training
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=kwargs.get("patience", 5),
            verbose=False,
            mode="min"
        )
        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("logs", name="tft_training")
        trainer = pl.Trainer(
            max_epochs=kwargs.get("max_epochs", 50),
            accelerator="auto",
            callbacks=[lr_logger, early_stop_callback],
            gradient_clip_val=kwargs.get("gradient_clip_val", 0.1),
            limit_train_batches=kwargs.get("limit_train_batches", None),
            logger=logger,
        )
        
        # Fit the network
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        return tft
        
    except Exception as e:
        print(f"TFT Training Error: {e}")
        raise