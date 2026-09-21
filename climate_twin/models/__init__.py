"""Model registry for the multi-model climate prediction framework.

Provides factory functions to instantiate and load any registered model by name.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base_model import ClimateModel, ModelMetrics, TARGET_COLUMNS

logger = logging.getLogger(__name__)

# Registry of available model classes
_MODEL_CLASSES: dict[str, type[ClimateModel]] = {}


def _register_models() -> None:
    """Lazily register all available model classes."""
    if _MODEL_CLASSES:
        return

    from .random_forest_model import RandomForestClimateModel
    from .xgboost_model import XGBoostClimateModel
    from .hist_gradient_boosting_model import HistGBClimateModel

    _MODEL_CLASSES["random_forest"] = RandomForestClimateModel
    _MODEL_CLASSES["xgboost"] = XGBoostClimateModel
    _MODEL_CLASSES["hist_gradient_boosting"] = HistGBClimateModel

    try:
        from .lstm_model import LSTMClimateModel, TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            _MODEL_CLASSES["lstm"] = LSTMClimateModel
        else:
            logger.info("LSTM model excluded: PyTorch not installed.")
    except ImportError:
        logger.info("LSTM model excluded: import failed.")


def get_model_class(name: str) -> type[ClimateModel]:
    """Get a model class by name."""
    _register_models()
    if name not in _MODEL_CLASSES:
        available = ", ".join(sorted(_MODEL_CLASSES.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_CLASSES[name]


def create_model(name: str, **kwargs: Any) -> ClimateModel:
    """Instantiate a model by name with optional hyperparameters."""
    cls = get_model_class(name)
    return cls(**kwargs)


def list_available_models() -> list[str]:
    """Return names of all available models."""
    _register_models()
    return sorted(_MODEL_CLASSES.keys())


def load_model(name: str, directory: Path) -> ClimateModel:
    """Load a trained model from disk."""
    model = create_model(name)
    model.load(directory)
    return model


__all__ = [
    "ClimateModel",
    "ModelMetrics",
    "TARGET_COLUMNS",
    "create_model",
    "get_model_class",
    "list_available_models",
    "load_model",
]
