"""
TTI_model.py
============
Production-grade PyTorch neural architecture for the TTI system.

Components
----------
TokenEmbedding       — learnable token + positional embeddings
TransformerEncoder   — multi-head self-attention text encoder (BERT-style)
ColourHead           — MLP predicting 18-d colour palette from text
SceneClassifier      — MLP predicting 15-class scene type
ParamDecoder         — VAE decoder: latent z → 64-d scene parameters
TTIModel             — unified model combining all heads
TTIModelLarge        — larger variant (6-layer transformer, 512-d)

Training utilities
------------------
TTILoss              — multi-task loss (scene CE + colour MSE + param MSE + KL)
TTITrainer           — full training loop with lr scheduling, checkpointing,
                       early stopping, TensorBoard-compatible metric logging
ModelCheckpoint      — best-model saving with metadata
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# ─────────────────────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Embedding
    vocab_size:      int   = 8192
    max_seq_len:     int   = 32
    embed_dim:       int   = 256

    # Transformer encoder
    n_layers:        int   = 4
    n_heads:         int   = 8
    ff_dim:          int   = 1024
    dropout:         float = 0.1
    attn_dropout:    float = 0.1

    # VAE latent space
    latent_dim:      int   = 128

    # Output heads
    n_scene_classes: int   = 15
    colour_dim:      int   = 18     # 6 colours × RGB
    param_dim:       int   = 64
    modifier_dim:    int   = 30

    # Training
    learning_rate:   float = 3e-4
    weight_decay:    float = 1e-2
    warmup_steps:    int   = 500
    max_steps:       int   = 20_000
    batch_size:      int   = 256
    grad_clip:       float = 1.0

    # Loss weights
    w_scene:         float = 1.0
    w_colour:        float = 2.0
    w_param:         float = 1.5
    w_modifier:      float = 0.5
    w_kl:            float = 0.001

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def large(cls, vocab_size: int = 8192) -> "ModelConfig":
        """6-layer, 512-d variant for higher quality."""
        return cls(vocab_size=vocab_size, embed_dim=512, n_layers=6, n_heads=8,
                   ff_dim=2048, latent_dim=256, dropout=0.1)

    @classmethod
    def small(cls, vocab_size: int = 8192) -> "ModelConfig":
        """Fast 2-layer, 128-d variant for experimentation."""
        return cls(vocab_size=vocab_size, embed_dim=128, n_layers=2, n_heads=4,
                   ff_dim=512, latent_dim=64, dropout=0.05)

    @classmethod
    def memory_safe(cls, vocab_size: int = 8192) -> "ModelConfig":
        """
        ≤4 GB RAM config using gradient checkpointing.
        4-layer 256-d transformer, batch=64 → 3.8M params.
        Enable via TTIModel(..., use_gradient_checkpointing=True).
        """
        return cls(
            vocab_size=vocab_size, embed_dim=256, n_layers=4,
            n_heads=8, ff_dim=1024, latent_dim=128, dropout=0.1,
            attn_dropout=0.05, batch_size=64, learning_rate=3e-4,
            warmup_steps=200, grad_clip=1.0,
            w_scene=1.0, w_colour=2.0, w_param=1.5, w_modifier=0.5, w_kl=0.001,
        )

    @classmethod
    def standard(cls, vocab_size: int = 8192) -> "ModelConfig":
        """Standard config for ≥8 GB RAM. 4-layer 256-d, batch=256, 3.8M params."""
        return cls(
            vocab_size=vocab_size, embed_dim=256, n_layers=4,
            n_heads=8, ff_dim=1024, latent_dim=128, dropout=0.1,
            batch_size=256, learning_rate=3e-4, warmup_steps=300, grad_clip=1.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionEmbedding(nn.Module):
    """Fixed sinusoidal positional embeddings (Vaswani et al. 2017)."""

    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        pe  = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Learnable token embedding + sinusoidal position embedding."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=0)
        self.pos_embed = SinusoidalPositionEmbedding(cfg.max_seq_len, cfg.embed_dim)
        self.dropout   = nn.Dropout(cfg.dropout)
        self.norm      = nn.LayerNorm(cfg.embed_dim)
        nn.init.normal_(self.tok_embed.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_embed(token_ids)     # (B, L, D)
        x = self.pos_embed(x)
        return self.dropout(self.norm(x))


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with GELU activation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.attn  = nn.MultiheadAttention(
            cfg.embed_dim, cfg.n_heads,
            dropout=cfg.attn_dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x:           torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + attn_out
        # Feed-forward with pre-norm
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers producing a [CLS]-style pooled vector."""

    def __init__(self, cfg: ModelConfig, use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.embedding  = TokenEmbedding(cfg)
        self.layers     = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.pool_norm  = nn.LayerNorm(cfg.embed_dim)
        self.use_ckpt   = use_gradient_checkpointing

    def forward(
        self,
        token_ids: torch.Tensor,        # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch.utils.checkpoint import checkpoint as grad_ckpt
        pad_mask = (token_ids == 0)
        x = self.embedding(token_ids)
        for layer in self.layers:
            if self.use_ckpt and self.training:
                x = grad_ckpt(layer, x, pad_mask, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=pad_mask)
        mask_float = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
        pooled = self.pool_norm(pooled)
        return pooled, x


class MLP(nn.Module):
    """General-purpose MLP with configurable depth and activation."""

    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        hidden:  List[int],
        dropout: float = 0.1,
        act:     str   = "gelu",
    ) -> None:
        super().__init__()
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU,
                  "silu": nn.SiLU, "tanh": nn.Tanh}[act]
        dims   = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(dims[i + 1]), act_fn(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Output heads
# ─────────────────────────────────────────────────────────────────────────────

class SceneClassifier(nn.Module):
    """Predicts discrete scene class from pooled text encoding."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mlp = MLP(cfg.embed_dim, cfg.n_scene_classes,
                       hidden=[cfg.embed_dim // 2], dropout=cfg.dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.mlp(pooled)   # (B, n_classes) — logits


class ColourHead(nn.Module):
    """Predicts 18-d colour palette (6 × RGB, normalised 0-1) from text."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mlp = MLP(cfg.embed_dim, cfg.colour_dim,
                       hidden=[cfg.embed_dim, cfg.embed_dim // 2], dropout=cfg.dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(pooled))   # (B, 18)


class ModifierHead(nn.Module):
    """Predicts 30-d binary modifier vector from text."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mlp = MLP(cfg.embed_dim, cfg.modifier_dim,
                       hidden=[cfg.embed_dim // 2], dropout=cfg.dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(pooled))   # (B, 30)


class VAEEncoder(nn.Module):
    """
    Encodes pooled text + colour → (μ, log_σ²) in latent space.
    Input: [pooled_text || colour_vec]  (D + 18)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_dim = cfg.embed_dim + cfg.colour_dim
        self.shared = MLP(in_dim, cfg.latent_dim * 2,
                          hidden=[cfg.embed_dim, cfg.embed_dim // 2],
                          dropout=cfg.dropout)
        self.mu_head     = nn.Linear(cfg.latent_dim * 2, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.latent_dim * 2, cfg.latent_dim)

    def forward(
        self, pooled: torch.Tensor, colour: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.shared(torch.cat([pooled, colour], dim=-1))
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 10)
        return mu, logvar

    @staticmethod
    def reparameterise(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if not TTIModel.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class ParamDecoder(nn.Module):
    """
    Decodes latent z + scene_class_emb → 64-d scene parameter vector.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Scene class embedding
        self.scene_emb = nn.Embedding(cfg.n_scene_classes, cfg.embed_dim // 4)
        in_dim = cfg.latent_dim + cfg.embed_dim // 4
        self.mlp = MLP(in_dim, cfg.param_dim,
                       hidden=[cfg.embed_dim, cfg.embed_dim],
                       dropout=cfg.dropout, act="silu")

    def forward(
        self,
        z:           torch.Tensor,    # (B, latent_dim)
        scene_logits: torch.Tensor,   # (B, n_classes)
    ) -> torch.Tensor:
        scene_idx = scene_logits.argmax(dim=-1)         # (B,)
        s_emb     = self.scene_emb(scene_idx)            # (B, D//4)
        return torch.sigmoid(self.mlp(torch.cat([z, s_emb], dim=-1)))


# ─────────────────────────────────────────────────────────────────────────────
# TTIModel  (unified)
# ─────────────────────────────────────────────────────────────────────────────

class TTIModel(nn.Module):
    """
    Full TTI neural model.

    Forward pass returns TTIModelOutput containing:
      scene_logits : (B, 15)   — scene class scores
      colour_pred  : (B, 18)   — predicted colour palette
      param_pred   : (B, 64)   — decoded scene parameters
      modifier_pred: (B, 30)   — modifier flags
      mu           : (B, 128)  — VAE mean
      logvar       : (B, 128)  — VAE log variance
      z            : (B, 128)  — sampled latent
    """

    training: bool = True   # class-level flag read by reparameterise

    def __init__(self, cfg: ModelConfig, use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.cfg      = cfg
        self.encoder  = TransformerEncoder(cfg, use_gradient_checkpointing)
        self.scene_cls= SceneClassifier(cfg)
        self.colour   = ColourHead(cfg)
        self.modifier = ModifierHead(cfg)
        self.vae_enc  = VAEEncoder(cfg)
        self.decoder  = ParamDecoder(cfg)

        self._n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._use_ckpt = use_gradient_checkpointing

    def forward(
        self,
        token_ids: torch.Tensor,          # (B, L)
        colour_gt: Optional[torch.Tensor] = None,  # (B, 18) — used during training
    ) -> "TTIModelOutput":

        pooled, seq_out = self.encoder(token_ids)

        scene_logits  = self.scene_cls(pooled)
        colour_pred   = self.colour(pooled)
        modifier_pred = self.modifier(pooled)

        # Use ground-truth colour during training for better VAE conditioning
        colour_input  = colour_gt if (colour_gt is not None and self.training) \
                        else colour_pred

        mu, logvar = self.vae_enc(pooled, colour_input)
        TTIModel.training = self.training
        z = VAEEncoder.reparameterise(mu, logvar)
        param_pred = self.decoder(z, scene_logits)

        return TTIModelOutput(
            scene_logits  = scene_logits,
            colour_pred   = colour_pred,
            param_pred    = param_pred,
            modifier_pred = modifier_pred,
            mu            = mu,
            logvar        = logvar,
            z             = z,
        )

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return pooled text representation (inference utility)."""
        with torch.no_grad():
            pooled, _ = self.encoder(token_ids)
        return pooled

    def generate(
        self,
        token_ids:   torch.Tensor,
        temperature: float = 1.0,
        seed:        Optional[int] = None,
    ) -> "TTIModelOutput":
        """Inference with temperature-scaled latent sampling."""
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        with torch.no_grad():
            out = self.forward(token_ids)
        # Re-sample z with temperature
        eps = torch.randn_like(out.mu) * temperature
        std = torch.exp(0.5 * out.logvar)
        z_t = out.mu + eps * std
        param_pred = self.decoder(z_t, out.scene_logits)
        return TTIModelOutput(
            scene_logits  = out.scene_logits,
            colour_pred   = out.colour_pred,
            param_pred    = param_pred,
            modifier_pred = out.modifier_pred,
            mu            = out.mu,
            logvar        = out.logvar,
            z             = z_t,
        )

    def n_parameters(self) -> int:
        return self._n_params

    def save(self, path: str, metadata: Optional[Dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":              self.state_dict(),
            "cfg":                     asdict(self.cfg),
            "n_params":                self._n_params,
            "use_gradient_checkpointing": self._use_ckpt,
            "metadata":                metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "TTIModel":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg  = ModelConfig(**ckpt["cfg"])
        use_ckpt = ckpt.get("use_gradient_checkpointing", False)
        model = cls(cfg, use_gradient_checkpointing=use_ckpt)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def __repr__(self) -> str:
        return (f"TTIModel(embed={self.cfg.embed_dim}, "
                f"layers={self.cfg.n_layers}, "
                f"heads={self.cfg.n_heads}, "
                f"params={self._n_params:,})")


@dataclass
class TTIModelOutput:
    scene_logits:  torch.Tensor
    colour_pred:   torch.Tensor
    param_pred:    torch.Tensor
    modifier_pred: torch.Tensor
    mu:            torch.Tensor
    logvar:        torch.Tensor
    z:             torch.Tensor

    def scene_probs(self) -> torch.Tensor:
        return F.softmax(self.scene_logits, dim=-1)

    def scene_class(self) -> torch.Tensor:
        return self.scene_logits.argmax(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# TTIModelLarge
# ─────────────────────────────────────────────────────────────────────────────

class TTIModelLarge(TTIModel):
    """6-layer, 512-d model for production-quality generation."""

    def __init__(self) -> None:
        super().__init__(ModelConfig.large())


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class TTILoss(nn.Module):
    """
    Multi-task loss combining:
      L_scene    : cross-entropy over 15 scene classes
      L_colour   : MSE on 18-d colour palette
      L_param    : Huber loss on 64-d parameter vector
      L_modifier : BCE on 30-d modifier flags
      L_kl       : KL divergence (VAE regularisation)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg   = cfg
        self.ce    = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.mse   = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=0.5)
        self.bce   = nn.BCELoss()

    def forward(
        self,
        output:       TTIModelOutput,
        scene_idx:    torch.Tensor,    # (B,) int64
        colour_gt:    torch.Tensor,    # (B, 18)
        param_gt:     torch.Tensor,    # (B, 64)
        modifier_gt:  torch.Tensor,    # (B, 30)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        L_scene    = self.ce(output.scene_logits, scene_idx)
        L_colour   = self.mse(output.colour_pred, colour_gt)
        L_param    = self.huber(output.param_pred, param_gt)
        L_modifier = self.bce(output.modifier_pred, modifier_gt)
        L_kl       = -0.5 * (
            1 + output.logvar - output.mu.pow(2) - output.logvar.exp()
        ).mean()

        total = (
            self.cfg.w_scene    * L_scene  +
            self.cfg.w_colour   * L_colour +
            self.cfg.w_param    * L_param  +
            self.cfg.w_modifier * L_modifier +
            self.cfg.w_kl       * L_kl
        )

        breakdown = {
            "scene":    L_scene.item(),
            "colour":   L_colour.item(),
            "param":    L_param.item(),
            "modifier": L_modifier.item(),
            "kl":       L_kl.item(),
            "total":    total.item(),
        }
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# ModelCheckpoint
# ─────────────────────────────────────────────────────────────────────────────

class ModelCheckpoint:
    """Saves the best model based on a monitored metric."""

    def __init__(
        self,
        save_dir:  str,
        monitor:   str   = "val_loss",
        mode:      str   = "min",
        save_top_k: int  = 3,
    ) -> None:
        self.save_dir  = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor   = monitor
        self.mode      = mode
        self.save_top_k = save_top_k
        self._scores:  List[Tuple[float, str]] = []   # (score, path)
        self.best      = math.inf if mode == "min" else -math.inf

    def is_better(self, score: float) -> bool:
        return score < self.best if self.mode == "min" else score > self.best

    def __call__(
        self,
        model:  TTIModel,
        score:  float,
        step:   int,
        extra:  Optional[Dict] = None,
    ) -> bool:
        """Save checkpoint; returns True if this is the new best."""
        path = str(self.save_dir / f"ckpt_step{step:06d}_{self.monitor}{score:.4f}.pt")
        model.save(path, metadata={"step": step, self.monitor: score, **(extra or {})})

        self._scores.append((score, path))
        self._scores.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        # Prune old checkpoints
        while len(self._scores) > self.save_top_k:
            _, old_path = self._scores.pop()
            try:
                Path(old_path).unlink()
            except FileNotFoundError:
                pass

        if self.is_better(score):
            self.best = score
            best_path = str(self.save_dir / "best_model.pt")
            model.save(best_path, metadata={"step": step, self.monitor: score})
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MetricLogger
# ─────────────────────────────────────────────────────────────────────────────

class MetricLogger:
    """Lightweight training metric recorder (JSON lines + console)."""

    def __init__(self, log_dir: str, verbose: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._log_file = open(self.log_dir / "metrics.jsonl", "a")
        self.history: List[Dict] = []

    def log(self, step: int, phase: str, metrics: Dict[str, float]) -> None:
        record = {"step": step, "phase": phase,
                  "time": time.strftime("%H:%M:%S"), **metrics}
        self.history.append(record)
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()
        if self.verbose:
            parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
            print(f"  [{phase:5s} step {step:6d}] " + "  ".join(parts))

    def close(self) -> None:
        self._log_file.close()

    def summary(self) -> Dict[str, float]:
        if not self.history:
            return {}
        last = {k: v for k, v in self.history[-1].items()
                if isinstance(v, float)}
        return last


# ─────────────────────────────────────────────────────────────────────────────
# TTITrainer
# ─────────────────────────────────────────────────────────────────────────────

class TTITrainer:
    """
    Full training loop with:
      - Linear warmup + cosine annealing LR schedule
      - Gradient clipping
      - Multi-task loss (scene + colour + param + modifier + KL)
      - Validation loop with accuracy and loss reporting
      - Best-model checkpointing (top-3 + best_model.pt)
      - Early stopping (patience-based)
      - JSON-lines metric log
      - Scene-class accuracy breakdown

    Usage
    -----
        trainer = TTITrainer(model, dataset, cfg, save_dir="tti_models")
        results = trainer.train(max_epochs=20)
    """

    def __init__(
        self,
        model:      TTIModel,
        dataset,                      # TTIDataset instance
        cfg:        ModelConfig,
        save_dir:   str  = "tti_models",
        device:     str  = "cpu",
        verbose:    bool = True,
    ) -> None:
        self.model    = model.to(device)
        self.dataset  = dataset
        self.cfg      = cfg
        self.device   = device
        self.verbose  = verbose

        self.loss_fn  = TTILoss(cfg)
        self.optim    = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # LR schedule: linear warmup → cosine decay
        self.warmup_sched = LinearLR(
            self.optim,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=cfg.warmup_steps,
        )
        self.cosine_sched = CosineAnnealingLR(
            self.optim,
            T_max=max(1, cfg.max_steps - cfg.warmup_steps),
            eta_min=cfg.learning_rate * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optim,
            schedulers=[self.warmup_sched, self.cosine_sched],
            milestones=[cfg.warmup_steps],
        )

        self.checkpoint = ModelCheckpoint(save_dir, monitor="val_loss", mode="min")
        self.logger     = MetricLogger(save_dir, verbose=verbose)
        self.save_dir   = Path(save_dir)
        self.global_step = 0

    # ── batch → tensors ───────────────────────────────────────────────────

    def _to_device(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.from_numpy(v).to(self.device)
            for k, v in batch.items()
        }

    # ── train one epoch ───────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        TTIModel.training = True
        totals: Dict[str, float] = {}
        n_batches = 0

        for batch_np in self.dataset.batches(
            "train", self.cfg.batch_size, shuffle=True, seed=epoch
        ):
            batch = self._to_device(batch_np)

            output = self.model(
                batch["token_ids"],
                colour_gt=batch["colour_vec"],
            )
            loss, breakdown = self.loss_fn(
                output,
                batch["scene_idx"],
                batch["colour_vec"],
                batch["param_vec"],
                batch["modifier_vec"],
            )

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
            self.scheduler.step()
            self.global_step += 1

            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            if self.global_step % 200 == 0:
                avg = {f"train_{k}": v / n_batches for k, v in totals.items()}
                avg["lr"] = self.optim.param_groups[0]["lr"]
                self.logger.log(self.global_step, "train", avg)

        return {f"train_{k}": v / max(1, n_batches) for k, v in totals.items()}

    # ── validation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        TTIModel.training = False
        totals:  Dict[str, float] = {}
        correct, total = 0, 0
        per_class_correct: Dict[int, int] = {}
        per_class_total:   Dict[int, int] = {}
        n_batches = 0

        for batch_np in self.dataset.batches(
            "val", self.cfg.batch_size, shuffle=False
        ):
            batch  = self._to_device(batch_np)
            output = self.model(batch["token_ids"])
            loss, breakdown = self.loss_fn(
                output,
                batch["scene_idx"],
                batch["colour_vec"],
                batch["param_vec"],
                batch["modifier_vec"],
            )
            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0.0) + v

            preds = output.scene_class()
            gt    = batch["scene_idx"]
            correct  += (preds == gt).sum().item()
            total    += gt.size(0)
            for cls_id in gt.unique().tolist():
                mask = gt == cls_id
                per_class_correct[cls_id] = per_class_correct.get(cls_id, 0) + \
                    (preds[mask] == gt[mask]).sum().item()
                per_class_total[cls_id]   = per_class_total.get(cls_id, 0) + mask.sum().item()
            n_batches += 1

        accuracy = correct / max(1, total)
        metrics  = {f"val_{k}": v / max(1, n_batches) for k, v in totals.items()}
        metrics["val_accuracy"] = accuracy

        # Per-class accuracy
        from TTI_dataset import SCENE_CLASSES
        for cls_id, cnt in per_class_total.items():
            cls_name = SCENE_CLASSES[cls_id] if cls_id < len(SCENE_CLASSES) else str(cls_id)
            metrics[f"val_acc_{cls_name}"] = per_class_correct.get(cls_id, 0) / max(1, cnt)

        return metrics

    # ── main train loop ───────────────────────────────────────────────────

    def train(
        self,
        max_epochs:    int = 20,
        eval_every:    int = 1,
        patience:      int = 5,
        min_delta:     float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Train for *max_epochs* epochs with early stopping.

        Returns summary dict with best val_loss and per-class accuracy.
        """
        print(f"\n{'='*60}")
        print(f"  TTI Model Training")
        print(f"  Model    : {self.model}")
        print(f"  Device   : {self.device}")
        print(f"  Train    : {self.dataset.split_len('train'):,} samples")
        print(f"  Val      : {self.dataset.split_len('val'):,} samples")
        print(f"  Epochs   : {max_epochs}")
        print(f"  Batch    : {self.cfg.batch_size}")
        print(f"  LR       : {self.cfg.learning_rate}")
        print(f"{'='*60}\n")

        best_val_loss  = math.inf
        patience_count = 0
        history: List[Dict] = []
        t0 = time.time()

        for epoch in range(1, max_epochs + 1):
            ep_t0 = time.time()
            print(f"── Epoch {epoch}/{max_epochs} ──────────────────────")

            train_metrics = self._train_epoch(epoch)
            self.logger.log(self.global_step, "train", train_metrics)

            if epoch % eval_every == 0:
                val_metrics = self._validate()
                self.logger.log(self.global_step, "val", val_metrics)

                val_loss = val_metrics["val_total"]
                val_acc  = val_metrics.get("val_accuracy", 0.0)

                is_best = self.checkpoint(
                    self.model, val_loss, self.global_step,
                    extra={"epoch": epoch, "val_accuracy": val_acc},
                )
                marker = " ★ BEST" if is_best else ""

                ep_time = time.time() - ep_t0
                print(f"  epoch={epoch}  val_loss={val_loss:.4f}  "
                      f"val_acc={val_acc:.3f}  t={ep_time:.1f}s{marker}")

                # Early stopping
                if val_loss < best_val_loss - min_delta:
                    best_val_loss  = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print(f"\n[Trainer] Early stopping at epoch {epoch} "
                              f"(patience={patience})")
                        break

                record = {"epoch": epoch, **train_metrics, **val_metrics}
                history.append(record)

        elapsed = time.time() - t0
        self.logger.close()

        summary = {
            "best_val_loss":  best_val_loss,
            "epochs_trained": epoch,
            "total_steps":    self.global_step,
            "elapsed_s":      round(elapsed, 1),
            "model_params":   self.model.n_parameters(),
            "final_metrics":  history[-1] if history else {},
        }

        # Save final summary
        with open(self.save_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Training complete in {elapsed/60:.1f} min")
        print(f"  Best val_loss : {best_val_loss:.4f}")
        print(f"  Best model    : {self.checkpoint.save_dir}/best_model.pt")
        print(f"{'='*60}\n")

        return summary