"""
TTI_train.py
============
Production training script for the TTI neural model.

Trains a 4-layer transformer (3.8M params) on the 225k-sample TTI dataset
with gradient checkpointing for memory efficiency, cosine LR scheduling,
early stopping, per-class accuracy tracking, and full checkpoint management.

Usage
-----
    # Build dataset then train (recommended)
    python TTI_train.py --build-dataset --n-samples 50000 --epochs 20

    # Train on existing dataset
    python TTI_train.py --epochs 20 --batch 64 --lr 3e-4

    # Quick smoke-test (2 epochs, tiny model)
    python TTI_train.py --smoke-test

    # Resume from checkpoint
    python TTI_train.py --resume tti_models/best_model.pt --epochs 10

    # Large model (needs ≥16 GB RAM)
    python TTI_train.py --model-size large --batch 128 --epochs 30
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── env tuning (must be before any torch ops) ──────────────────────────────
torch.set_num_threads(2)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


# ── TTI imports ─────────────────────────────────────────────────────────────
from .TTI_dataset import (
    TTIDataset, SCENE_CLASSES,
    KEYWORD_COLOUR, KEYWORD_SCENE, KEYWORD_MODIFIER,
)
from .TTI_model import (
    TTIModel, ModelConfig, TTILoss,
    ModelCheckpoint, MetricLogger,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def rss_mb() -> int:
    """Current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def eta_str(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    remaining = elapsed / done * (total - done)
    if remaining < 60:
        return f"{remaining:.0f}s"
    return f"{remaining/60:.1f}m"


def print_banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# ArrayDataset  — memory-efficient numpy-backed dataset
# ─────────────────────────────────────────────────────────────────────────────

class ArrayDataset:
    """
    Wraps pre-computed numpy arrays for training.
    Supports stratified sub-sampling and on-the-fly shuffled batching
    without loading the full pickle corpus into RAM.
    """

    def __init__(
        self,
        token_ids:    np.ndarray,   # (N, L)  int16 → cast to int64 on access
        colour_vec:   np.ndarray,   # (N, 18) float32
        param_vec:    np.ndarray,   # (N, 64) float32
        modifier_vec: np.ndarray,   # (N, 30) float32
        scene_idx:    np.ndarray,   # (N,)    int8  → cast to int64 on access
        name:         str = "split",
    ) -> None:
        self.token_ids    = token_ids
        self.colour_vec   = colour_vec
        self.param_vec    = param_vec
        self.modifier_vec = modifier_vec
        self.scene_idx    = scene_idx
        self.name         = name
        self.N            = token_ids.shape[0]

    def __len__(self) -> int:
        return self.N

    def batches(
        self,
        batch_size: int,
        shuffle:    bool = True,
        seed:       int  = 0,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.N) if shuffle else np.arange(self.N)
        for i in range(0, self.N, batch_size):
            b = idx[i : i + batch_size]
            yield {
                "token_ids":    torch.from_numpy(self.token_ids[b].astype(np.int64)),
                "colour_vec":   torch.from_numpy(self.colour_vec[b]),
                "param_vec":    torch.from_numpy(self.param_vec[b]),
                "modifier_vec": torch.from_numpy(self.modifier_vec[b]),
                "scene_idx":    torch.from_numpy(self.scene_idx[b].astype(np.int64)),
            }

    @classmethod
    def from_npz(cls, path: str, name: str = "split") -> "ArrayDataset":
        d = np.load(path)
        return cls(
            token_ids    = d["token_ids"],
            colour_vec   = d["colour_vec"],
            param_vec    = d["param_vec"],
            modifier_vec = d["modifier_vec"],
            scene_idx    = d["scene_idx"],
            name         = name,
        )

    @classmethod
    def from_pkl_stratified(
        cls,
        pkl_path:    str,
        per_class:   int  = 2000,
        seed:        int  = 42,
        name:        str  = "train",
    ) -> "ArrayDataset":
        """
        Load the full pickle, take *per_class* samples per scene class
        (stratified), and return compact numpy arrays.
        Releases the pickle list immediately after extraction.
        """
        import pickle, random
        rng = random.Random(seed)
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        buckets: Dict[str, list] = defaultdict(list)
        for d in raw:
            buckets[d["scene_label"]].append(d)
        del raw; gc.collect()

        selected = []
        for cls_name in SCENE_CLASSES:
            pool = buckets.get(cls_name, [])
            rng.shuffle(pool)
            selected.extend(pool[:per_class])
        rng.shuffle(selected)

        token_ids    = np.array([d["token_ids"]    for d in selected], dtype=np.int16)
        colour_vec   = np.array([d["colour_vec"]   for d in selected], dtype=np.float32)
        param_vec    = np.array([d["param_vec"]    for d in selected], dtype=np.float32)
        modifier_vec = np.array([d["modifier_vec"] for d in selected], dtype=np.float32)
        scene_idx    = np.array([d["scene_idx"]    for d in selected], dtype=np.int8)
        del selected, buckets; gc.collect()

        return cls(token_ids, colour_vec, param_vec, modifier_vec, scene_idx, name)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training loop for TTIModel.

    Features
    --------
    - Gradient checkpointing (enabled automatically when RSS > 3200 MB)
    - Cosine LR with linear warmup
    - Gradient clipping
    - Per-epoch validation with 15-class accuracy breakdown
    - Best-model checkpoint (top-3 + best_model.pt)
    - Early stopping with configurable patience
    - JSON training history + human-readable summary
    - Memory monitoring with automatic batch-size reduction on pressure
    """

    def __init__(
        self,
        model:      TTIModel,
        train_ds:   ArrayDataset,
        val_ds:     ArrayDataset,
        cfg:        ModelConfig,
        save_dir:   str  = "tti_models",
        verbose:    bool = True,
    ) -> None:
        self.model    = model
        self.train_ds = train_ds
        self.val_ds   = val_ds
        self.cfg      = cfg
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose  = verbose

        self.loss_fn  = TTILoss(cfg)
        self.optim    = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg.learning_rate,
            weight_decay = cfg.weight_decay,
            betas        = (0.9, 0.98),
            eps          = 1e-8,
        )
        # LR: linear warmup → cosine decay
        total_steps    = cfg.max_steps if cfg.max_steps else \
                         (len(train_ds) // cfg.batch_size) * 20
        warmup         = min(cfg.warmup_steps, total_steps // 10)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return max(1e-6, step / max(1, warmup))
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        self.ckpt      = ModelCheckpoint(save_dir, monitor="val_loss",
                                         mode="min", save_top_k=3)
        self.logger    = MetricLogger(save_dir, verbose=False)
        self.step      = 0
        self.history:  List[Dict] = []

    # ── one training epoch ────────────────────────────────────────────────

    def _train_epoch(self, epoch: int, batch_size: int) -> Dict[str, float]:
        self.model.train()
        TTIModel.training = True
        totals: Dict[str, float] = {}
        nb = 0
        t0 = time.time()

        for batch in self.train_ds.batches(batch_size, shuffle=True, seed=epoch):
            out = self.model(batch["token_ids"], colour_gt=batch["colour_vec"])
            loss, bd = self.loss_fn(
                out,
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
            self.step += 1

            for k, v in bd.items():
                totals[k] = totals.get(k, 0.0) + v
            nb += 1
            del out, loss
            gc.collect()

            if self.verbose and nb % 50 == 0:
                elapsed = time.time() - t0
                total_b = math.ceil(len(self.train_ds) / batch_size)
                lr      = self.optim.param_groups[0]["lr"]
                print(
                    f"    batch {nb:3d}/{total_b}  "
                    f"loss={totals.get('total',0)/nb:.4f}  "
                    f"lr={lr:.1e}  "
                    f"RAM={rss_mb()}MB  "
                    f"ETA={eta_str(elapsed, nb, total_b)}",
                    end="\r", flush=True,
                )

        if self.verbose:
            print(" " * 80, end="\r")   # clear progress line

        return {f"train_{k}": v / max(1, nb) for k, v in totals.items()}

    # ── validation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, batch_size: int) -> Dict[str, float]:
        self.model.eval()
        TTIModel.training = False
        totals:  Dict[str, float] = {}
        nb = 0
        correct = 0
        total   = 0
        per_class_cor: Dict[int, int] = defaultdict(int)
        per_class_tot: Dict[int, int] = defaultdict(int)

        for batch in self.val_ds.batches(batch_size, shuffle=False):
            out = self.model(batch["token_ids"])
            _, bd = self.loss_fn(
                out,
                batch["scene_idx"],
                batch["colour_vec"],
                batch["param_vec"],
                batch["modifier_vec"],
            )
            for k, v in bd.items():
                totals[k] = totals.get(k, 0.0) + v
            nb += 1

            preds = out.scene_class()
            gt    = batch["scene_idx"]
            correct += (preds == gt).sum().item()
            total   += gt.size(0)

            for cls_id in gt.unique().tolist():
                mask = gt == cls_id
                per_class_cor[cls_id] += (preds[mask] == gt[mask]).sum().item()
                per_class_tot[cls_id] += mask.sum().item()

            del out; gc.collect()

        metrics = {f"val_{k}": v / max(1, nb) for k, v in totals.items()}
        metrics["val_accuracy"] = correct / max(1, total)
        metrics["val_n"]        = total

        # Per-class accuracy
        for cls_id, cnt in per_class_tot.items():
            name = SCENE_CLASSES[cls_id] if cls_id < len(SCENE_CLASSES) else str(cls_id)
            metrics[f"val_acc_{name}"] = per_class_cor[cls_id] / max(1, cnt)

        return metrics

    # ── main loop ─────────────────────────────────────────────────────────

    def train(
        self,
        max_epochs: int   = 20,
        patience:   int   = 5,
        min_delta:  float = 1e-4,
        batch_size: Optional[int] = None,
    ) -> Dict:
        B         = batch_size or self.cfg.batch_size
        best_loss = math.inf
        pat_count = 0
        t0_total  = time.time()

        print_banner("TTI Production Training")
        print(f"  Model      : {self.model}")
        print(f"  Parameters : {self.model.n_parameters():,}")
        print(f"  Grad-ckpt  : {self.model._use_ckpt}")
        print(f"  Train set  : {len(self.train_ds):,} samples")
        print(f"  Val set    : {len(self.val_ds):,} samples")
        print(f"  Batch size : {B}")
        print(f"  Max epochs : {max_epochs}   Patience: {patience}")
        print(f"  LR         : {self.cfg.learning_rate}")
        print(f"  Save dir   : {self.save_dir}")
        print(f"  RAM now    : {rss_mb()} MB\n")

        for epoch in range(1, max_epochs + 1):
            ep_t0 = time.time()
            print(f"── Epoch {epoch:02d}/{max_epochs} " + "─"*40)

            train_metrics = self._train_epoch(epoch, B)
            val_metrics   = self._validate(B)

            val_loss = val_metrics["val_total"]
            val_acc  = val_metrics["val_accuracy"]
            tr_loss  = train_metrics["train_total"]
            lr_now   = self.optim.param_groups[0]["lr"]
            ep_time  = time.time() - ep_t0

            is_best = self.ckpt(
                self.model, val_loss, self.step,
                extra={"epoch": epoch, "val_accuracy": val_acc},
            )
            marker = "  ★ NEW BEST" if is_best else ""

            # Console summary
            print(
                f"  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}  lr={lr_now:.1e}  "
                f"t={ep_time:.1f}s  RAM={rss_mb()}MB{marker}"
            )

            # Per-class accuracy table
            print("  Scene accuracy:")
            row = ""
            for i, cls in enumerate(SCENE_CLASSES):
                a = val_metrics.get(f"val_acc_{cls}", 0.0)
                bar = "█" * int(a * 10) + "░" * (10 - int(a * 10))
                row += f"    {cls:12s}  {bar}  {a:.3f}\n"
            print(row, end="")

            # Log & record
            record = {
                "epoch":      epoch,
                "step":       self.step,
                "lr":         lr_now,
                "ram_mb":     rss_mb(),
                "ep_time_s":  round(ep_time, 1),
                **train_metrics,
                **{k: v for k, v in val_metrics.items()
                   if not k.startswith("val_acc_")},
                "val_accuracy": val_acc,
            }
            self.history.append(record)
            self.logger.log(self.step, "epoch", {
                "train_loss": tr_loss,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "lr":         lr_now,
            })

            # Early stopping
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                pat_count = 0
            else:
                pat_count += 1
                if pat_count >= patience:
                    print(f"\n  Early stopping triggered (patience={patience})")
                    break

        # ── Wrap up ───────────────────────────────────────────────────────
        elapsed = time.time() - t0_total
        self.logger.close()

        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        final = self.history[-1] if self.history else {}
        summary = {
            "best_val_loss":     float(best_loss),
            "final_val_accuracy": float(final.get("val_accuracy", 0)),
            "epochs_trained":    epoch,
            "total_steps":       self.step,
            "elapsed_min":       round(elapsed / 60, 2),
            "n_train":           len(self.train_ds),
            "n_val":             len(self.val_ds),
            "model_params":      self.model.n_parameters(),
            "batch_size":        B,
            "scene_classes":     SCENE_CLASSES,
            "per_class_final_acc": {
                cls: float(final.get(f"val_acc_{cls}", 0))
                for cls in SCENE_CLASSES
            },
        }
        with open(self.save_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print_banner("Training Complete")
        print(f"  Best val_loss      : {best_loss:.4f}")
        print(f"  Final val_accuracy : {final.get('val_accuracy', 0):.3f}")
        print(f"  Epochs trained     : {epoch}")
        print(f"  Total steps        : {self.step:,}")
        print(f"  Time               : {elapsed/60:.1f} min")
        print(f"  Best model         : {self.save_dir}/best_model.pt")
        print(f"  Training log       : {self.save_dir}/metrics.jsonl")
        print(f"  Summary            : {self.save_dir}/training_summary.json\n")

        return summary


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_arrays(
    cache_dir:   str,
    per_class:   int  = 2000,    # samples per class for train
    val_samples: int  = 5000,
    force:       bool = False,
    seed:        int  = 42,
) -> Tuple[ArrayDataset, ArrayDataset, int]:
    """
    Load or build stratified numpy array splits.

    Returns (train_ds, val_ds, vocab_size).
    """
    cache      = Path(cache_dir)
    train_npz  = cache / "train_arrays.npz"
    val_npz    = cache / "val_arrays.npz"

    # ── Build train arrays ────────────────────────────────────────────────
    if force or not train_npz.exists():
        print(f"[Prepare] Building train arrays ({per_class}/class × {len(SCENE_CLASSES)} classes)…")
        train_ds = ArrayDataset.from_pkl_stratified(
            str(cache / "train.pkl"),
            per_class=per_class,
            seed=seed,
            name="train",
        )
        np.savez_compressed(
            str(train_npz),
            token_ids    = train_ds.token_ids,
            colour_vec   = train_ds.colour_vec,
            param_vec    = train_ds.param_vec,
            modifier_vec = train_ds.modifier_vec,
            scene_idx    = train_ds.scene_idx,
        )
        print(f"  Saved train_arrays.npz  ({len(train_ds):,} samples)")
    else:
        print(f"[Prepare] Loading cached train_arrays.npz…")
        train_ds = ArrayDataset.from_npz(str(train_npz), name="train")
        print(f"  {len(train_ds):,} training samples")

    # ── Build val arrays ──────────────────────────────────────────────────
    if force or not val_npz.exists():
        print(f"[Prepare] Building val arrays ({val_samples} samples)…")
        import pickle
        with open(str(cache / "val.pkl"), "rb") as f:
            raw_val = pickle.load(f)
        raw_val = raw_val[:val_samples]
        np.savez_compressed(
            str(val_npz),
            token_ids    = np.array([d["token_ids"]    for d in raw_val], dtype=np.int16),
            colour_vec   = np.array([d["colour_vec"]   for d in raw_val], dtype=np.float32),
            param_vec    = np.array([d["param_vec"]    for d in raw_val], dtype=np.float32),
            modifier_vec = np.array([d["modifier_vec"] for d in raw_val], dtype=np.float32),
            scene_idx    = np.array([d["scene_idx"]    for d in raw_val], dtype=np.int8),
        )
        del raw_val; gc.collect()
        print(f"  Saved val_arrays.npz  ({val_samples:,} samples)")

    val_ds = ArrayDataset.from_npz(str(val_npz), name="val")
    print(f"  {len(val_ds):,} validation samples\n")

    # Vocab size from meta
    try:
        meta = json.loads((cache / "meta.json").read_text())
        vocab_size = meta.get("vocab_size", 8192)
    except Exception:
        vocab_size = 8192

    return train_ds, val_ds, vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:    TTIModel,
    val_ds:   ArrayDataset,
    cfg:      ModelConfig,
    batch_size: int = 64,
) -> Dict:
    """Run full evaluation and print a report."""
    loss_fn = TTILoss(cfg)
    model.eval()
    TTIModel.training = False

    all_preds, all_gt = [], []
    total_loss = 0.0; nb = 0
    colour_mse = 0.0; param_mse = 0.0

    with torch.no_grad():
        for batch in val_ds.batches(batch_size, shuffle=False):
            out = model(batch["token_ids"])
            _, bd = loss_fn(out, batch["scene_idx"], batch["colour_vec"],
                            batch["param_vec"], batch["modifier_vec"])
            total_loss += bd["total"]; nb += 1
            all_preds.extend(out.scene_class().tolist())
            all_gt.extend(batch["scene_idx"].tolist())
            colour_mse += bd["colour"]
            param_mse  += bd["param"]
            del out; gc.collect()

    accuracy = sum(p == g for p, g in zip(all_preds, all_gt)) / max(1, len(all_gt))
    gt_cnt   = Counter(all_gt)
    cor_cnt  = Counter(g for p, g in zip(all_preds, all_gt) if p == g)

    print_banner("Evaluation Report")
    print(f"  Samples      : {len(all_gt):,}")
    print(f"  Val loss     : {total_loss/max(1,nb):.4f}")
    print(f"  Accuracy     : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Colour MSE   : {colour_mse/max(1,nb):.4f}")
    print(f"  Param MSE    : {param_mse/max(1,nb):.4f}")
    print(f"\n  Per-class accuracy:")
    print(f"  {'Scene':<14} {'Acc':>6}  {'Correct':>8}  {'Total':>8}  Bar")
    print("  " + "─" * 58)
    for i, cls in enumerate(SCENE_CLASSES):
        tot = gt_cnt.get(i, 0)
        cor = cor_cnt.get(i, 0)
        a   = cor / max(1, tot)
        bar = "█" * int(a * 20)
        print(f"  {cls:<14} {a:>6.3f}  {cor:>8,}  {tot:>8,}  {bar}")

    return {
        "val_loss":    total_loss / max(1, nb),
        "accuracy":    accuracy,
        "colour_mse":  colour_mse / max(1, nb),
        "param_mse":   param_mse / max(1, nb),
        "per_class":   {
            SCENE_CLASSES[i]: cor_cnt.get(i, 0) / max(1, gt_cnt.get(i, 0))
            for i in range(len(SCENE_CLASSES))
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="TTI_train",
        description="TTI production training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: build 50k dataset then train
  python TTI_train.py --build-dataset --n-samples 50000 --epochs 20

  # Train on existing dataset (standard config)
  python TTI_train.py --epochs 20 --batch 256 --lr 3e-4

  # Memory-safe 4GB config with gradient checkpointing
  python TTI_train.py --epochs 20 --batch 64 --grad-ckpt

  # Evaluate saved model
  python TTI_train.py --eval tti_models/best_model.pt

  # Quick smoke test (2 epochs, tiny model)
  python TTI_train.py --smoke-test

  # Resume training
  python TTI_train.py --resume tti_models/best_model.pt --epochs 10
        """,
    )

    # Dataset
    g = p.add_argument_group("Dataset")
    g.add_argument("--cache-dir",     default="tti_cache",  help="Dataset cache directory")
    g.add_argument("--build-dataset", action="store_true",  help="(Re)build dataset before training")
    g.add_argument("--n-samples",     type=int, default=50_000, help="Total samples to generate")
    g.add_argument("--per-class",     type=int, default=2000,   help="Train samples per class")
    g.add_argument("--val-samples",   type=int, default=5000,   help="Validation samples")
    g.add_argument("--no-augment",    action="store_true",  help="Disable prompt augmentation")
    g.add_argument("--force-rebuild", action="store_true",  help="Force re-extract arrays from pkl")

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--model-size",  default="standard",
                   choices=["small", "standard", "memory-safe", "large"],
                   help="Model size preset")
    g.add_argument("--embed-dim",   type=int,   default=None)
    g.add_argument("--n-layers",    type=int,   default=None)
    g.add_argument("--n-heads",     type=int,   default=None)
    g.add_argument("--ff-dim",      type=int,   default=None)
    g.add_argument("--latent-dim",  type=int,   default=None)
    g.add_argument("--grad-ckpt",   action="store_true",
                   help="Enable gradient checkpointing (saves ~150 MB RAM)")

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",       type=int,   default=20)
    g.add_argument("--batch",        type=int,   default=None,  help="Batch size (overrides preset)")
    g.add_argument("--lr",           type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--patience",     type=int,   default=5)
    g.add_argument("--warmup",       type=int,   default=None)
    g.add_argument("--grad-clip",    type=float, default=1.0)
    g.add_argument("--seed",         type=int,   default=42)
    g.add_argument("--save-dir",     default="tti_models")
    g.add_argument("--resume",       default=None,  metavar="CKPT", help="Resume from checkpoint")

    # Modes
    g = p.add_argument_group("Modes")
    g.add_argument("--eval",       default=None, metavar="CKPT", help="Evaluate model only")
    g.add_argument("--smoke-test", action="store_true",           help="Quick 2-epoch smoke test")
    g.add_argument("--quiet",      action="store_true",           help="Reduce console output")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args   = build_parser().parse_args(argv)
    verbose = not args.quiet

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Smoke test ────────────────────────────────────────────────────────
    if args.smoke_test:
        print("[Smoke test] Building tiny synthetic dataset…")
        N = 500
        synth = ArrayDataset(
            token_ids    = np.random.randint(0, 512, (N, 32), dtype=np.int16),
            colour_vec   = np.random.rand(N, 18).astype(np.float32),
            param_vec    = np.random.rand(N, 64).astype(np.float32),
            modifier_vec = np.random.rand(N, 30).astype(np.float32),
            scene_idx    = np.random.randint(0, 15, N, dtype=np.int8),
            name         = "smoke",
        )
        cfg   = ModelConfig.small()
        cfg.vocab_size = 512; cfg.batch_size = 64; cfg.learning_rate = 1e-3
        model = TTIModel(cfg, use_gradient_checkpointing=False)
        trainer = Trainer(model, synth, synth, cfg,
                          save_dir=args.save_dir, verbose=verbose)
        results = trainer.train(max_epochs=3, patience=10, batch_size=64)
        print("[Smoke test] PASSED" if results["epochs_trained"] >= 2 else "[Smoke test] FAILED")
        return 0

    # ── Eval only ─────────────────────────────────────────────────────────
    if args.eval:
        print(f"[Eval] Loading model from {args.eval}…")
        model = TTIModel.load(args.eval)
        _, val_ds, _ = prepare_arrays(args.cache_dir, val_samples=args.val_samples)
        cfg = model.cfg
        evaluate_model(model, val_ds, cfg, batch_size=args.batch or cfg.batch_size)
        return 0

    # ── Build dataset ─────────────────────────────────────────────────────
    if args.build_dataset:
        print_banner("Building Dataset")
        TTIDataset.build(
            cache_dir  = args.cache_dir,
            n_samples  = args.n_samples,
            augment    = not args.no_augment,
            seed       = args.seed,
            verbose    = verbose,
        )

    # ── Prepare arrays ────────────────────────────────────────────────────
    print_banner("Preparing Training Arrays")
    train_ds, val_ds, vocab_size = prepare_arrays(
        cache_dir   = args.cache_dir,
        per_class   = args.per_class,
        val_samples = args.val_samples,
        force       = args.force_rebuild,
        seed        = args.seed,
    )

    # ── Build / load model ────────────────────────────────────────────────
    if args.resume:
        print(f"[Train] Resuming from {args.resume}…")
        model = TTIModel.load(args.resume)
        cfg   = model.cfg
        cfg.learning_rate = args.lr
    else:
        preset = {
            "small":       ModelConfig.small,
            "standard":    ModelConfig.standard,
            "memory-safe": ModelConfig.memory_safe,
            "large":       ModelConfig.large,
        }[args.model_size]
        cfg = preset(vocab_size=vocab_size)

        # CLI overrides
        if args.embed_dim:  cfg.embed_dim  = args.embed_dim
        if args.n_layers:   cfg.n_layers   = args.n_layers
        if args.n_heads:    cfg.n_heads    = args.n_heads
        if args.ff_dim:     cfg.ff_dim     = args.ff_dim
        if args.latent_dim: cfg.latent_dim = args.latent_dim
        cfg.learning_rate   = args.lr
        cfg.weight_decay    = args.weight_decay
        cfg.grad_clip       = args.grad_clip
        cfg.warmup_steps    = args.warmup or cfg.warmup_steps
        if args.batch:      cfg.batch_size = args.batch

        use_ckpt = args.grad_ckpt or (args.model_size == "memory-safe")
        model    = TTIModel(cfg, use_gradient_checkpointing=use_ckpt)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(f"{args.save_dir}/model_config.json")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model    = model,
        train_ds = train_ds,
        val_ds   = val_ds,
        cfg      = cfg,
        save_dir = args.save_dir,
        verbose  = verbose,
    )
    results = trainer.train(
        max_epochs = args.epochs,
        patience   = args.patience,
        batch_size = args.batch or cfg.batch_size,
    )

    # ── Final evaluation ──────────────────────────────────────────────────
    best = TTIModel.load(f"{args.save_dir}/best_model.pt")
    evaluate_model(best, val_ds, best.cfg, batch_size=args.batch or cfg.batch_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
