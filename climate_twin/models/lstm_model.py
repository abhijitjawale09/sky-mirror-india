"""LSTM Neural Network for temporal climate forecasting.

Constructs proper sequential time-series input windows from historical
observations. Uses a 14-day lookback window to predict the next day's
rainfall, max temperature, and min temperature.

If PyTorch is not installed, this module gracefully degrades and the
LSTM model is excluded from the comparison framework.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base_model import ClimateModel, TARGET_COLUMNS

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. LSTM model will not be available.")


# --- Sequence construction utilities ---

def create_sequences(
    X: np.ndarray, y: np.ndarray, lookback: int = 14
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping time-series windows from tabular data.

    Given features X of shape (T, F) and targets y of shape (T, 3),
    returns:
        X_seq: (T - lookback, lookback, F) — input windows
        y_seq: (T - lookback, 3) — target for the day after each window

    IMPORTANT: This assumes X and y are already sorted chronologically
    within a single region. Cross-region boundaries must be handled by
    the caller (split by region before calling this function).
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback : i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# --- PyTorch LSTM Module ---

if TORCH_AVAILABLE:

    class ClimateLSTMNetwork(nn.Module):
        """LSTM network for multi-output climate regression."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            output_size: int = 3,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            # Use the last time step's hidden state
            last_hidden = lstm_out[:, -1, :]
            out = self.dropout(last_hidden)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out


class LSTMClimateModel(ClimateModel):
    """LSTM-based climate forecasting model with proper temporal windowing.

    This model constructs 14-day lookback windows from the feature matrix
    to create proper sequential inputs for the LSTM. It does NOT blindly
    feed tabular rows — it respects temporal ordering.

    If PyTorch is not installed, fit() raises a clear error.
    """

    name = "lstm"
    display_name = "LSTM"

    def __init__(
        self,
        lookback: int = 14,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.network: Any = None
        self.scaler_mean_X: np.ndarray | None = None
        self.scaler_std_X: np.ndarray | None = None
        self.scaler_mean_y: np.ndarray | None = None
        self.scaler_std_y: np.ndarray | None = None
        self.lookback = lookback

        self._hyperparameters = {
            "lookback": lookback,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "random_state": random_state,
        }

    def _check_torch(self) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the LSTM model. "
                "Install with: pip install torch>=2.0"
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self._check_torch()
        self.feature_columns = feature_names or []

        hp = self._hyperparameters
        torch.manual_seed(hp["random_state"])
        np.random.seed(hp["random_state"])

        # Normalize features and targets
        self.scaler_mean_X = X_train.mean(axis=0)
        self.scaler_std_X = X_train.std(axis=0) + 1e-8
        self.scaler_mean_y = y_train.mean(axis=0)
        self.scaler_std_y = y_train.std(axis=0) + 1e-8

        X_norm = (X_train - self.scaler_mean_X) / self.scaler_std_X
        y_norm = (y_train - self.scaler_mean_y) / self.scaler_std_y

        # Create sequences
        X_seq, y_seq = create_sequences(X_norm, y_norm, self.lookback)

        if len(X_seq) < 50:
            raise ValueError(
                f"Insufficient data for LSTM: only {len(X_seq)} sequences "
                f"after windowing with lookback={self.lookback}. "
                f"Need at least 50 sequences for meaningful training."
            )

        # Build PyTorch datasets
        device = torch.device("cpu")
        X_tensor = torch.FloatTensor(X_seq).to(device)
        y_tensor = torch.FloatTensor(y_seq).to(device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=hp["batch_size"], shuffle=True
        )

        # Validation set
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = (X_val - self.scaler_mean_X) / self.scaler_std_X
            y_val_norm = (y_val - self.scaler_mean_y) / self.scaler_std_y
            X_val_seq, y_val_seq = create_sequences(
                X_val_norm, y_val_norm, self.lookback
            )
            if len(X_val_seq) > 0:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_seq).to(device),
                    torch.FloatTensor(y_val_seq).to(device),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=hp["batch_size"], shuffle=False
                )

        # Initialize network
        input_size = X_train.shape[1]
        self.network = ClimateLSTMNetwork(
            input_size=input_size,
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=hp["learning_rate"]
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(hp["epochs"]):
            self.network.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.network(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if val_loader is not None:
                self.network.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        output = self.network(batch_X)
                        val_loss += criterion(output, batch_y).item()
                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.clone() for k, v in self.network.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= hp["patience"]:
                    logger.info(f"LSTM early stopping at epoch {epoch + 1}")
                    break
            else:
                # Without validation, just save latest
                best_state = {
                    k: v.clone() for k, v in self.network.state_dict().items()
                }

        # Restore best model
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from already-windowed or flat feature arrays.

        If X is 2D (n_samples, n_features): creates windows using the last
        `lookback` rows as context for each prediction.
        If X is 3D (n_samples, lookback, n_features): uses directly.
        """
        self._check_torch()
        if self.network is None:
            raise RuntimeError("LSTM model has not been trained.")

        X_norm = (X - self.scaler_mean_X) / self.scaler_std_X

        if X_norm.ndim == 2:
            X_seq, _ = create_sequences(
                X_norm, np.zeros((len(X_norm), 3)), self.lookback
            )
            if len(X_seq) == 0:
                # Not enough data for sequences — fall back to repeating
                X_seq = np.tile(X_norm[-1], (1, self.lookback, 1))
        else:
            X_seq = X_norm

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq)
            y_norm = self.network(X_tensor).numpy()

        # Inverse transform
        y_pred = y_norm * self.scaler_std_y + self.scaler_mean_y
        return y_pred

    def get_feature_importance(self) -> dict[str, float] | None:
        """LSTM does not have meaningful per-feature importance.

        Neural network weights should not be falsely labeled as feature
        importance. Return None to indicate this is not supported.
        """
        return None

    def save(self, directory: Path) -> Path:
        self._check_torch()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "lstm_model.pt"

        save_dict = {
            "network_state": self.network.state_dict() if self.network else None,
            "scaler_mean_X": self.scaler_mean_X,
            "scaler_std_X": self.scaler_std_X,
            "scaler_mean_y": self.scaler_mean_y,
            "scaler_std_y": self.scaler_std_y,
            "feature_columns": self.feature_columns,
            "hyperparameters": self._hyperparameters,
            "training_time_s": self.training_time_s,
        }
        torch.save(save_dict, path)
        return path

    def load(self, directory: Path) -> None:
        self._check_torch()
        path = directory / "lstm_model.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)

        self.feature_columns = data["feature_columns"]
        self._hyperparameters = data["hyperparameters"]
        self.scaler_mean_X = data["scaler_mean_X"]
        self.scaler_std_X = data["scaler_std_X"]
        self.scaler_mean_y = data["scaler_mean_y"]
        self.scaler_std_y = data["scaler_std_y"]
        self.training_time_s = data.get("training_time_s", 0.0)

        hp = self._hyperparameters
        input_size = len(self.feature_columns)
        self.network = ClimateLSTMNetwork(
            input_size=input_size,
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        )
        if data["network_state"] is not None:
            self.network.load_state_dict(data["network_state"])
        self.network.eval()
        self.is_fitted = True
