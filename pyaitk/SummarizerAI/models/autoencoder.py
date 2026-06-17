"""
PyTorch Models:
  - MemoryAutoencoder: compresses sparse TF-IDF vectors into dense latent space
  - EmbeddingProjector: lightweight linear projector for downstream tasks
"""

import logging
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


# ─── Autoencoder ──────────────────────────────────────────────────────────────

class MemoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MemoryDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),   # TF-IDF normalized to [0,1] after L2-norm
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MemoryAutoencoder(nn.Module):
    """
    Full autoencoder: encodes high-dim TF-IDF → latent, decodes back.
    The encoder output is the dense "memory embedding".
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.encoder = MemoryEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = MemoryDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class AutoencoderTrainer:
    """
    Trains the MemoryAutoencoder on TF-IDF feature matrices.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 16,
        device: str = "cpu",
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model: Optional[MemoryAutoencoder] = None
        self.train_losses: list = []

    def fit(self, X: np.ndarray) -> "AutoencoderTrainer":
        """X: dense numpy matrix, shape (n_samples, n_features)"""
        if issparse(X):
            X = X.toarray()

        input_dim = X.shape[1]
        self.model = MemoryAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)

        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                x_hat, _ = self.model(batch)
                loss = criterion(x_hat, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            self.train_losses.append(avg)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Autoencoder Epoch {epoch+1}/{self.epochs}  loss={avg:.5f}")

        logger.info("Autoencoder training complete.")
        return self

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Return dense latent embeddings for input X."""
        assert self.model is not None, "Call fit() first."
        if issparse(X):
            X = X.toarray()

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            z = self.model.encode(X_t)
        return z.cpu().numpy()

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "input_dim": self.model.input_dim,
            "train_losses": self.train_losses,
        }, path)
        logger.info(f"Model saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model = MemoryAutoencoder(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            latent_dim=ckpt["latent_dim"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.train_losses = ckpt.get("train_losses", [])
        logger.info(f"Model loaded ← {path}")
        return self
