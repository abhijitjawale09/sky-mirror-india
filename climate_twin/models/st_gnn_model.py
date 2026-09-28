"""Spatio-Temporal Graph Neural Network (ST-GNN) for Climate Forecasting.

Leverages India's geographic and meteorological graph topology:
The 7 pilot regions act as graph nodes with spatial edges derived from
inverse geodesic distance and synoptic monsoon flow tracks (Bay of Bengal
and Arabian Sea low-pressure tracks).

A Graph Convolutional Network (GCN) layer propagates regional atmospheric signals
spatially across the subcontinent, followed by a temporal feature fusion head.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .base_model import ClimateModel, TARGET_COLUMNS
from ..data.preprocessing import REGION_PROFILES

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. ST-GNN will not be available.")


def build_india_spatial_adjacency(num_regions: int = 7) -> np.ndarray:
    """Build normalized spatial adjacency matrix for the 7 Indian pilot climate zones."""
    coords = np.array([[p.latitude, p.longitude] for p in REGION_PROFILES])
    n = len(coords)
    adj = np.zeros((n, n), dtype=np.float32)

    # Compute pairwise Euclidean/geodesic distance in degree space
    for i in range(n):
        for j in range(n):
            if i == j:
                adj[i, j] = 1.0
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                # Gaussian distance decay (sigma ~ 8 degrees)
                adj[i, j] = math.exp(-(dist**2) / (2 * (8.0**2)))

    # Meteorological synoptic tracks:
    # Coastal Odisha (index 6) -> Central India (index 3) & Indo-Gangetic Plain (index 1)
    adj[6, 3] += 0.4
    adj[3, 6] += 0.4
    adj[6, 1] += 0.3
    adj[1, 6] += 0.3
    # Kerala Coast (index 0) -> Deccan Plateau (index 4) & Central India (index 3)
    adj[0, 4] += 0.35
    adj[4, 0] += 0.35

    # Degree normalization: D^(-1/2) * A * D^(-1/2)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat
    return norm_adj.astype(np.float32)


if TORCH_AVAILABLE:

    class GraphConvLayer(nn.Module):
        """Graph Convolution: H' = A_norm * H * W + b"""

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            # x: (num_nodes, in_features)
            # adj: (num_nodes, num_nodes)
            support = torch.mm(x, self.weight)
            output = torch.mm(adj, support) + self.bias
            return output

    class SpatioTemporalGNNNet(nn.Module):
        """Spatio-Temporal Graph Neural Network."""

        def __init__(
            self,
            feature_dim: int,
            num_nodes: int = 7,
            node_dim: int = 32,
            hidden_dim: int = 64,
            output_dim: int = 3,
        ) -> None:
            super().__init__()
            self.num_nodes = num_nodes
            # Learnable regional node embeddings
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, node_dim))
            # Spatial graph convolutions
            self.gcn1 = GraphConvLayer(node_dim, hidden_dim)
            self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)
            self.relu = nn.ReLU()

            # Dynamic feature adaptation
            self.feature_proj = nn.Linear(feature_dim, hidden_dim)

            # Combined spatio-temporal fusion head
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, output_dim),
            )

        def forward(
            self,
            features: torch.Tensor,
            region_indices: torch.Tensor,
            adj: torch.Tensor,
        ) -> torch.Tensor:
            # 1. Spatial message passing across India's climate graph
            spatial_state1 = self.relu(self.gcn1(self.node_embeddings, adj))
            spatial_state2 = self.relu(self.gcn2(spatial_state1, adj))  # (7, hidden_dim)

            # 2. Extract spatial context for each sample's region
            # region_indices: (batch_size,) clamped to 0..6
            sample_spatial = spatial_state2[region_indices]  # (batch_size, hidden_dim)

            # 3. Transform local meteorological features
            feat_repr = self.relu(self.feature_proj(features))  # (batch_size, hidden_dim)

            # 4. Spatio-temporal fusion
            fused = torch.cat([feat_repr, sample_spatial], dim=1)
            out = self.fusion(fused)
            return out


class STGNNClimateModel(ClimateModel):
    """Spatio-Temporal Graph Neural Network for multi-region climate prediction."""

    name = "st_gnn"
    display_name = "Spatio-Temporal GNN"

    def __init__(
        self,
        node_dim: int = 32,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 90,
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
        self.adj_matrix: np.ndarray = build_india_spatial_adjacency(7)
        self.residual_stds: dict[str, float] = {}

        self._hyperparameters = {
            "node_dim": node_dim,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "random_state": random_state,
        }

    def _check_torch(self) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for STGNNClimateModel. "
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

        self.scaler_mean_X = np.nanmean(X_train, axis=0)
        self.scaler_std_X = np.nanstd(X_train, axis=0)
        self.scaler_std_X[self.scaler_std_X < 1e-6] = 1.0

        self.scaler_mean_y = np.nanmean(y_train, axis=0)
        self.scaler_std_y = np.nanstd(y_train, axis=0)
        self.scaler_std_y[self.scaler_std_y < 1e-6] = 1.0

        X_train_norm = np.nan_to_num((X_train - self.scaler_mean_X) / self.scaler_std_X)
        y_train_norm = (y_train - self.scaler_mean_y) / self.scaler_std_y

        # Region code index is the 16th column (index 15)
        region_col_idx = 15 if X_train.shape[1] > 15 else -1
        regions_train = np.clip(np.round(X_train[:, region_col_idx]).astype(int), 0, 6)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.LongTensor(regions_train),
            torch.FloatTensor(y_train_norm),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=hp["batch_size"], shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = np.nan_to_num((X_val - self.scaler_mean_X) / self.scaler_std_X)
            y_val_norm = (y_val - self.scaler_mean_y) / self.scaler_std_y
            regions_val = np.clip(np.round(X_val[:, region_col_idx]).astype(int), 0, 6)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_norm),
                torch.LongTensor(regions_val),
                torch.FloatTensor(y_val_norm),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=hp["batch_size"], shuffle=False
            )

        self.network = SpatioTemporalGNNNet(
            feature_dim=X_train.shape[1],
            num_nodes=7,
            node_dim=hp["node_dim"],
            hidden_dim=hp["hidden_dim"],
            output_dim=3,
        )

        adj_tensor = torch.FloatTensor(self.adj_matrix)
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

        for epoch in range(hp["epochs"]):
            self.network.train()
            for batch_X, batch_reg, batch_y in train_loader:
                optimizer.zero_grad()
                out = self.network(batch_X, batch_reg, adj_tensor)
                loss = criterion(out, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()

            if val_loader is not None:
                self.network.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_reg, batch_y in val_loader:
                        out = self.network(batch_X, batch_reg, adj_tensor)
                        val_loss += criterion(out, batch_y).item()
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
        train_preds = self._predict_tensors(X_train)
        for i, target in enumerate(TARGET_COLUMNS):
            resids = y_train[:, i] - train_preds[:, i]
            self.residual_stds[target] = float(np.std(resids))

        self.is_fitted = True

    def _predict_tensors(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        X_norm = np.nan_to_num((X - self.scaler_mean_X) / self.scaler_std_X)
        region_col_idx = 15 if X.shape[1] > 15 else -1
        regions = np.clip(np.round(X[:, region_col_idx]).astype(int), 0, 6)

        adj_tensor = torch.FloatTensor(self.adj_matrix)
        with torch.no_grad():
            out_norm = self.network(
                torch.FloatTensor(X_norm),
                torch.LongTensor(regions),
                adj_tensor,
            ).numpy()

        preds = out_norm * self.scaler_std_y + self.scaler_mean_y
        preds[:, 0] = np.maximum(0.0, preds[:, 0])
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_torch()
        if not self.is_fitted or self.network is None:
            raise RuntimeError("ST-GNN model has not been trained.")
        return self._predict_tensors(X)

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
        path = directory / "st_gnn_weights.pt"
        torch.save(self.network.state_dict(), path)

        meta_path = directory / "st_gnn_meta.json"
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
        meta_path = directory / "st_gnn_meta.json"
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

        self.network = SpatioTemporalGNNNet(
            feature_dim=len(self.feature_columns) or 16,
            num_nodes=7,
            node_dim=hp["node_dim"],
            hidden_dim=hp["hidden_dim"],
            output_dim=3,
        )

        weights_path = directory / "st_gnn_weights.pt"
        self.network.load_state_dict(
            torch.load(weights_path, weights_only=True, map_location=torch.device("cpu"))
        )
        self.network.eval()
        self.is_fitted = True
