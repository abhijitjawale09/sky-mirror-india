"""Temporal Transformer Neural Network for Time-Series Climate Forecasting.

Uses multi-head self-attention with temporal positional encodings and
learned attention pooling across 14-day historical lookback windows to capture
both short-term synoptic variations and long-range seasonal transitions.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .base_model import ClimateModel, TARGET_COLUMNS
from .lstm_model import create_sequences

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Temporal Transformer will not be available.")


if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for sequence time-steps."""

        def __init__(self, d_model: int, max_len: int = 50) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1), :]

    class ClimateTransformerNetwork(nn.Module):
        """Transformer encoder with attention-pooling for multi-target climate regression."""

        def __init__(
            self,
            input_size: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.15,
            output_size: int = 3,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=60)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False
            )

            # Attention pooling over temporal dimension
            self.pool_attn = nn.Linear(d_model, 1)

            # Regression head
            self.head = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            h = self.input_proj(x)
            h = self.pos_encoder(h)
            encoded = self.transformer_encoder(h)

            # Temporal attention pooling
            attn_scores = torch.softmax(self.pool_attn(encoded), dim=1)
            pooled = torch.sum(encoded * attn_scores, dim=1)

            out = self.head(pooled)
            return out


class TemporalTransformerClimateModel(ClimateModel):
    """Temporal Transformer model with multi-head self-attention and sequence windowing."""

    name = "temporal_transformer"
    display_name = "Temporal Transformer"

    def __init__(
        self,
        lookback: int = 14,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.15,
        learning_rate: float = 0.001,
        epochs: int = 80,
        batch_size: int = 64,
        patience: int = 12,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.network: Any = None
        self.scaler_mean_X: np.ndarray | None = None
        self.scaler_std_X: np.ndarray | None = None
        self.scaler_mean_y: np.ndarray | None = None
        self.scaler_std_y: np.ndarray | None = None
        self.lookback = lookback
        self.residual_stds: dict[str, float] = {}

        self._hyperparameters = {
            "lookback": lookback,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
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
                "PyTorch is required for TemporalTransformerClimateModel. "
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

        # Compute z-score scaling from training data only
        self.scaler_mean_X = np.nanmean(X_train, axis=0)
        self.scaler_std_X = np.nanstd(X_train, axis=0)
        self.scaler_std_X[self.scaler_std_X < 1e-6] = 1.0

        self.scaler_mean_y = np.nanmean(y_train, axis=0)
        self.scaler_std_y = np.nanstd(y_train, axis=0)
        self.scaler_std_y[self.scaler_std_y < 1e-6] = 1.0

        X_train_norm = np.nan_to_num((X_train - self.scaler_mean_X) / self.scaler_std_X)
        y_train_norm = (y_train - self.scaler_mean_y) / self.scaler_std_y

        X_train_seq, y_train_seq = create_sequences(
            X_train_norm, y_train_norm, hp["lookback"]
        )

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=hp["batch_size"], shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = np.nan_to_num((X_val - self.scaler_mean_X) / self.scaler_std_X)
            y_val_norm = (y_val - self.scaler_mean_y) / self.scaler_std_y
            X_val_seq, y_val_seq = create_sequences(
                X_val_norm, y_val_norm, hp["lookback"]
            )
            if len(X_val_seq) > 0:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq)
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=hp["batch_size"], shuffle=False
                )

        input_size = X_train.shape[1]
        self.network = ClimateTransformerNetwork(
            input_size=input_size,
            d_model=hp["d_model"],
            nhead=hp["nhead"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"],
            dropout=hp["dropout"],
            output_size=3,
        )

        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=hp["learning_rate"], weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.network.train()
        for epoch in range(hp["epochs"]):
            self.network.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.network(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

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
                    best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= hp["patience"]:
                    break
            else:
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()

        # Compute empirical residual standard deviations
        train_preds = self._predict_sequences(X_train_seq)
        y_train_actual = y_train_seq * self.scaler_std_y + self.scaler_mean_y
        for i, target in enumerate(TARGET_COLUMNS):
            diff = y_train_actual[:, i] - train_preds[:, i]
            self.residual_stds[target] = float(np.std(diff))

        self.is_fitted = True

    def _predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X_seq)
            y_norm = self.network(tensor).numpy()
        preds = y_norm * self.scaler_std_y + self.scaler_mean_y
        preds[:, 0] = np.maximum(0.0, preds[:, 0])
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_torch()
        if self.network is None or not self.is_fitted:
            raise RuntimeError("Temporal Transformer has not been trained.")

        X_norm = np.nan_to_num((X - self.scaler_mean_X) / self.scaler_std_X)

        if X_norm.ndim == 2:
            if len(X_norm) >= self.lookback:
                X_seq, _ = create_sequences(
                    X_norm, np.zeros((len(X_norm), 3)), self.lookback
                )
            else:
                repeats = math.ceil(self.lookback / len(X_norm))
                tiled = np.tile(X_norm, (repeats, 1))[-self.lookback :]
                X_seq = np.expand_dims(tiled, axis=0)
        else:
            X_seq = X_norm

        preds = self._predict_sequences(X_seq)

        # If flat 2D input was passed, ensure output matches sample length
        if X.ndim == 2 and len(preds) != len(X):
            diff = len(X) - len(preds)
            if diff > 0:
                head = np.tile(preds[0], (diff, 1))
                preds = np.vstack([head, preds])
        return preds

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean_preds = self.predict(X)
        z_80 = 1.28155

        lower = np.zeros_like(mean_preds)
        upper = np.zeros_like(mean_preds)

        for i, target in enumerate(TARGET_COLUMNS):
            sigma = self.residual_stds.get(target, 1.0)
            lower[:, i] = mean_preds[:, i] - z_80 * sigma
            upper[:, i] = mean_preds[:, i] + z_80 * sigma

        lower[:, 0] = np.maximum(0.0, lower[:, 0])
        upper[:, 0] = np.maximum(0.0, upper[:, 0])
        return mean_preds, lower, upper

    def save(self, directory: Path) -> Path:
        self._check_torch()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "transformer_weights.pt"
        torch.save(self.network.state_dict(), path)

        meta_path = directory / "transformer_meta.json"
        meta = {
            "scaler_mean_X": self.scaler_mean_X.tolist() if self.scaler_mean_X is not None else None,
            "scaler_std_X": self.scaler_std_X.tolist() if self.scaler_std_X is not None else None,
            "scaler_mean_y": self.scaler_mean_y.tolist() if self.scaler_mean_y is not None else None,
            "scaler_std_y": self.scaler_std_y.tolist() if self.scaler_std_y is not None else None,
            "feature_columns": self.feature_columns,
            "hyperparameters": self._hyperparameters,
            "training_time_s": self.training_time_s,
            "residual_stds": self.residual_stds,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    def load(self, directory: Path) -> None:
        self._check_torch()
        meta_path = directory / "transformer_meta.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.scaler_mean_X = np.array(meta["scaler_mean_X"])
        self.scaler_std_X = np.array(meta["scaler_std_X"])
        self.scaler_mean_y = np.array(meta["scaler_mean_y"])
        self.scaler_std_y = np.array(meta["scaler_std_y"])
        self.feature_columns = meta["feature_columns"]
        self._hyperparameters = meta["hyperparameters"]
        self.training_time_s = meta.get("training_time_s", 0.0)
        self.residual_stds = meta.get("residual_stds", {})
        hp = self._hyperparameters

        self.network = ClimateTransformerNetwork(
            input_size=len(self.feature_columns) or 16,
            d_model=hp["d_model"],
            nhead=hp["nhead"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"],
            dropout=hp["dropout"],
            output_size=3,
        )

        weights_path = directory / "transformer_weights.pt"
        self.network.load_state_dict(
            torch.load(weights_path, weights_only=True, map_location=torch.device("cpu"))
        )
        self.network.eval()
        self.is_fitted = True
