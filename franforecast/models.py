"""Forecasters with a single fit/predict interface used by the backtester."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec

import numpy as np
import pandas as pd

from .config import season_length
from .errors import ModelError
from .features import DEFAULT_LAGS, DEFAULT_WINDOWS, FeatureBuilder


@dataclass(frozen=True)
class SeriesHistory:
    unique_id: str
    dates: pd.DatetimeIndex
    y: np.ndarray
    freq: str


class Forecaster(ABC):
    name: str = "forecaster"
    family: str = "other"

    def __init__(self) -> None:
        self.history: SeriesHistory | None = None

    def fit(self, history: SeriesHistory) -> "Forecaster":
        if len(history.y) == 0:
            raise ModelError(f"{self.name} received an empty series ({history.unique_id}).")
        self.history = SeriesHistory(
            unique_id=history.unique_id,
            dates=pd.DatetimeIndex(history.dates),
            y=np.asarray(history.y, dtype=float),
            freq=history.freq,
        )
        self._fit(self.history)
        return self

    def _fit(self, history: SeriesHistory) -> None:
        return None

    def _non_negative(self) -> bool:
        return bool(self.history is not None and np.nanmin(self.history.y) >= 0)

    def _clip(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if self._non_negative():
            values = np.clip(values, 0.0, None)
        return values

    @abstractmethod
    def predict(self, future_dates: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError


class Naive(Forecaster):
    name = "Naive"
    family = "Baseline"

    def _fit(self, history: SeriesHistory) -> None:
        if len(history.y) < 1:
            raise ModelError("Naive needs at least one observation.")

    def predict(self, future_dates: pd.DatetimeIndex) -> np.ndarray:
        return np.repeat(float(self.history.y[-1]), len(future_dates))


class SeasonalNaive(Forecaster):
    name = "Seasonal naive"
    family = "Baseline"

    def _fit(self, history: SeriesHistory) -> None:
        self.period = season_length(history.freq)
        if len(history.y) < self.period:
            raise ModelError(
                f"Seasonal naive needs {self.period} observations for one full season, "
                f"got {len(history.y)} ({history.unique_id})."
            )

    def predict(self, future_dates: pd.DatetimeIndex) -> np.ndarray:
        pattern = np.asarray(self.history.y[-self.period :], dtype=float)
        index = np.arange(len(future_dates)) % self.period
        return pattern[index]


class ETSForecaster(Forecaster):
    name = "ETS"
    family = "Statistical"

    def _fit(self, history: SeriesHistory) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        period = season_length(history.freq)
        use_seasonal = len(history.y) >= 2 * period
        if len(history.y) < 3:
            raise ModelError(f"ETS needs at least 3 observations, got {len(history.y)}.")

        try:
            model = ExponentialSmoothing(
                history.y,
                trend="add",
                seasonal="add" if use_seasonal else None,
                seasonal_periods=period if use_seasonal else None,
                initialization_method="heuristic",
            )
            self.result = model.fit(optimized=True)
        except Exception as exc:  # noqa: BLE001 - solver failures become model errors
            raise ModelError(f"ETS failed to fit ({history.unique_id}): {exc}") from exc

    def predict(self, future_dates: pd.DatetimeIndex) -> np.ndarray:
        return self._clip(np.asarray(self.result.forecast(len(future_dates)), dtype=float))


class LightGBMForecaster(Forecaster):
    name = "LightGBM"
    family = "Machine learning"
    min_training_rows = 30

    def __init__(
        self,
        lags: tuple[int, ...] = DEFAULT_LAGS,
        windows: tuple[int, ...] = DEFAULT_WINDOWS,
        n_estimators: int = 400,
    ) -> None:
        super().__init__()
        self.builder = FeatureBuilder(lags=lags, windows=windows)
        self.n_estimators = n_estimators

    def _fit(self, history: SeriesHistory) -> None:
        import lightgbm as lgb

        if len(history.y) < self.builder.warmup + self.min_training_rows:
            raise ModelError(
                f"LightGBM needs at least {self.builder.warmup + self.min_training_rows} observations for lag "
                f"features plus {self.min_training_rows} training rows; series {history.unique_id} has {len(history.y)}. "
                "Use a baseline for series this short."
            )

        features, target, _ = self.builder.training_matrix(history.dates, history.y)
        if len(target) < self.min_training_rows:
            raise ModelError(
                f"LightGBM got {len(target)} usable training rows for {history.unique_id}; "
                f"at least {self.min_training_rows} are required."
            )

        self.builder.clip_non_negative = self._non_negative()
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=0,
            # One thread: thread-pool startup dominates on a few hundred rows.
            n_jobs=1,
            force_col_wise=True,
            verbose=-1,
        )
        self.model.fit(
            pd.DataFrame(features, columns=self.builder.names),
            target,
        )

    def predict(self, future_dates: pd.DatetimeIndex) -> np.ndarray:
        def predict_one(row: np.ndarray) -> float:
            frame = pd.DataFrame([row], columns=self.builder.names)
            return float(self.model.predict(frame)[0])

        return self.builder.forecast(
            predict_one,
            self.history.dates,
            self.history.y,
            future_dates,
        )


@lru_cache(maxsize=2)
def _chronos_pipeline(model_id: str):
    import torch
    from chronos import ChronosPipeline

    return ChronosPipeline.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )


@lru_cache(maxsize=1)
def is_chronos_available() -> bool:
    return find_spec("chronos") is not None and find_spec("torch") is not None


def chronos_hint() -> str:
    return (
        "Chronos is not installed. Install it with: pip install chronos-forecasting==1.5.0 "
        "(about 300 MB with PyTorch; CPU inference is slow)."
    )


class ChronosForecaster(Forecaster):
    name = "Chronos tiny"
    family = "Foundation model"

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-tiny",
        context_length: int = 128,
        num_samples: int = 20,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples

    def _fit(self, history: SeriesHistory) -> None:
        if len(history.y) < 8:
            raise ModelError(
                f"Chronos needs at least 8 historical points, got {len(history.y)} ({history.unique_id})."
            )

    def predict(self, future_dates: pd.DatetimeIndex) -> np.ndarray:
        import torch

        try:
            pipeline = _chronos_pipeline(self.model_id)
            context = np.asarray(self.history.y[-self.context_length :], dtype=np.float32)
            samples = pipeline.predict(
                torch.tensor(context),
                prediction_length=len(future_dates),
                num_samples=self.num_samples,
            )
            median = np.quantile(samples[0].to(torch.float32).numpy(), 0.5, axis=0)
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - download/inference failures become model errors
            raise ModelError(f"Chronos inference failed: {exc}") from exc
        return self._clip(median)


MODEL_CLASSES: dict[str, type[Forecaster]] = {
    "Naive": Naive,
    "Seasonal naive": SeasonalNaive,
    "ETS": ETSForecaster,
    "LightGBM": LightGBMForecaster,
    "Chronos tiny": ChronosForecaster,
}

DEFAULT_MODELS = ("Naive", "Seasonal naive", "ETS", "LightGBM")


def model_catalog(include_chronos: bool = True) -> dict[str, str]:
    """Model name -> family, in display order, excluding unavailable models."""
    catalog = {
        name: cls.family
        for name, cls in MODEL_CLASSES.items()
        if name != "Chronos tiny"
    }
    if include_chronos and is_chronos_available():
        catalog["Chronos tiny"] = ChronosForecaster.family
    return catalog


def make_model(name: str) -> Forecaster:
    try:
        return MODEL_CLASSES[name]()
    except KeyError as exc:
        raise ModelError(f"Unknown model '{name}'.") from exc


def family_of(name: str) -> str:
    cls = MODEL_CLASSES.get(name)
    return cls.family if cls else "Unknown"


def minimum_history(model_name: str) -> int:
    """Observations a model needs before its first fold can be trained."""
    if MODEL_CLASSES.get(model_name) is LightGBMForecaster:
        warmup = max(max(DEFAULT_LAGS), max(DEFAULT_WINDOWS))
        return warmup + LightGBMForecaster.min_training_rows
    return 0
