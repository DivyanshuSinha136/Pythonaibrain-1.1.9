"""
run_training.py
===============
Self-contained production training loop.
Runs 20 epochs on 30k samples, saves best model and full history.
"""
import torch
import numpy as np
import gc
import os
import sys
import json
import time
import math
import torch.nn as nn
from collections import defaultdict

torch.set_num_threads(2)

# ── helpers ──────────────────────────────────────────────────────────────────
def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def rss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return 0

# ── data ─────────────────────────────────────────────────────────────────────
log("Loading data...")
tr = np.load("tti_cache/train_arrays.npz")
TK = torch.from_numpy(tr["token_ids"].astype(np.int64))
TC = torch.from_numpy(tr["colour_vec"])
TP = torch.from_numpy(tr["param_vec"])
TM = torch.from_numpy(tr["modifier_vec"])
TS = torch.from_numpy(tr["scene_idx"].astype(np.int64))
del tr; gc.collect()

va = np.load("tti_cache/val_arrays.npz")
VK = torch.from_numpy(va["token_ids"].astype(np.int64))
VC = torch.from_numpy(va["colour_vec"])
VP = torch.from_numpy(va["param_vec"])
VM = torch.from_numpy(va["modifier_vec"])
VS = torch.from_numpy(va["scene_idx"].astype(np.int64))
del va; gc.collect()

N_train, N_val = TK.shape[0], VK.shape[0]
log(f"Train: {N_train:,}  Val: {N_val:,}  RAM: {rss()} MB")

# ── model ─────────────────────────────────────────────────────────────────────
from TTI_model import TTIModel, ModelConfig, TTILoss, ModelCheckpoint, MetricLogger
from TTI_dataset import SCENE_CLASSES

VOCAB         = 603
BATCH         = 128
MAX_EPOCHS    = 20
PATIENCE      = 5
LR            = 3e-4

cfg = ModelConfig.small(vocab_size=VOCAB)
cfg.learning_rate = LR
cfg.weight_decay  = 1e-2
cfg.warmup_steps  = 100
cfg.batch_size    = BATCH
cfg.grad_clip     = 1.0
cfg.w_scene       = 1.0
cfg.w_colour      = 2.0
cfg.w_param       = 1.5
cfg.w_modifier    = 0.5
cfg.w_kl          = 0.001

os.makedirs("tti_models", exist_ok=True)
cfg.save("tti_models/model_config.json")

model   = TTIModel(cfg, use_gradient_checkpointing=False)
loss_fn = TTILoss(cfg)
optim   = torch.optim.AdamW(
    model.parameters(), lr=LR,
    weight_decay=1e-2, betas=(0.9, 0.98), eps=1e-8,
)

total_steps = (N_train // BATCH) * MAX_EPOCHS

def lr_lambda(step):
    if step < cfg.warmup_steps:
        return max(1e-6, step / max(1, cfg.warmup_steps))
    prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return max(0.01, 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
ckpt      = ModelCheckpoint("tti_models", monitor="val_loss", mode="min", save_top_k=3)
logger    = MetricLogger("tti_models", verbose=False)

log(f"Model: {model.n_parameters():,} params  RAM: {rss()} MB")
log("=" * 60)
log("  TTI PRODUCTION TRAINING")
log(f"  Dataset : {N_train:,} train / {N_val:,} val")
log(f"  Epochs  : {MAX_EPOCHS}   Batch: {BATCH}   LR: {LR}")
log(f"  Model   : small ({model.n_parameters():,} params)")
log("=" * 60)

step       = 0
best_loss  = math.inf
pat_count  = 0
history    = []
t_total    = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    # ── Train epoch ──────────────────────────────────────────────────────
    model.train()
    TTIModel.training = True
    perm   = torch.randperm(N_train)
    totals = defaultdict(float)
    nb     = 0
    t_ep   = time.time()

    for i in range(0, N_train, BATCH):
        idx  = perm[i : i + BATCH]
        out  = model(TK[idx], colour_gt=TC[idx])
        loss, bd = loss_fn(out, TS[idx], TC[idx], TP[idx], TM[idx])
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        scheduler.step()
        for k, v in bd.items():
            totals[k] += v
        nb   += 1
        step += 1
        del out, loss
        gc.collect()

    tr_metrics = {k: v / max(1, nb) for k, v in totals.items()}

    # ── Validation ───────────────────────────────────────────────────────
    model.eval()
    TTIModel.training = False
    v_totals  = defaultdict(float)
    vb        = 0
    correct   = 0
    total     = 0
    cls_cor   = defaultdict(int)
    cls_tot   = defaultdict(int)

    with torch.no_grad():
        for j in range(0, N_val, BATCH):
            out  = model(VK[j : j + BATCH])
            _, bd = loss_fn(
                out,
                VS[j : j + BATCH],
                VC[j : j + BATCH],
                VP[j : j + BATCH],
                VM[j : j + BATCH],
            )
            for k, v in bd.items():
                v_totals[k] += v
            vb     += 1
            preds   = out.scene_class()
            gt      = VS[j : j + BATCH]
            correct += (preds == gt).sum().item()
            total   += gt.size(0)
            for ci in gt.unique().tolist():
                mask         = gt == ci
                cls_cor[ci] += (preds[mask] == gt[mask]).sum().item()
                cls_tot[ci] += mask.sum().item()
            del out
            gc.collect()

    val_loss = v_totals["total"] / max(1, vb)
    val_acc  = correct / max(1, total)
    lr_now   = optim.param_groups[0]["lr"]
    ep_time  = time.time() - t_ep

    # Checkpoint
    is_best = ckpt(model, val_loss, step,
                   extra={"epoch": epoch, "val_acc": val_acc})
    mark    = "  ★ BEST" if is_best else ""

    # Per-class accuracy
    cls_accs = {
        SCENE_CLASSES[ci]: cls_cor[ci] / max(1, cls_tot[ci])
        for ci in range(len(SCENE_CLASSES))
    }

    # Logging
    rec = {
        "epoch":        epoch,
        "step":         step,
        "lr":           lr_now,
        "ep_time_s":    round(ep_time, 1),
        "ram_mb":       rss(),
        "train_loss":   tr_metrics["total"],
        "train_scene":  tr_metrics["scene"],
        "train_colour": tr_metrics["colour"],
        "train_param":  tr_metrics["param"],
        "val_loss":     val_loss,
        "val_acc":      val_acc,
        "val_scene":    v_totals["scene"] / max(1, vb),
        "val_colour":   v_totals["colour"] / max(1, vb),
        "val_param":    v_totals["param"] / max(1, vb),
    }
    rec.update({f"acc_{k}": v for k, v in cls_accs.items()})
    history.append(rec)
    logger.log(step, "epoch", {
        "train_loss": tr_metrics["total"],
        "val_loss":   val_loss,
        "val_acc":    val_acc,
        "lr":         lr_now,
    })

    # Console
    log(
        f"Ep {epoch:2d}/{MAX_EPOCHS} | "
        f"tr={tr_metrics['total']:.4f} "
        f"(sc={tr_metrics['scene']:.3f} "
        f"col={tr_metrics['colour']:.3f} "
        f"param={tr_metrics['param']:.3f}) | "
        f"val={val_loss:.4f} | "
        f"acc={val_acc:.3f} | "
        f"lr={lr_now:.1e} | "
        f"{ep_time:.0f}s | "
        f"RAM={rss()}MB"
        + mark
    )

    # Per-class bar chart (every epoch)
    lines = []
    for i, cls in enumerate(SCENE_CLASSES):
        a   = cls_accs.get(cls, 0.0)
        bar = ("█" * int(a * 12)).ljust(12)
        lines.append(f"  {cls:12s} {bar} {a:.3f}")
        if (i + 1) % 5 == 0:
            log("  ".join(lines)); lines = []
    if lines:
        log("  ".join(lines))

    # Save history every epoch
    with open("tti_models/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Early stopping
    if val_loss < best_loss - 1e-4:
        best_loss = val_loss
        pat_count = 0
    else:
        pat_count += 1
        if pat_count >= PATIENCE:
            log(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

# ── Wrap up ───────────────────────────────────────────────────────────────────
logger.close()
elapsed = time.time() - t_total

final = history[-1]
summary = {
    "best_val_loss":       float(best_loss),
    "final_val_accuracy":  float(final["val_acc"]),
    "epochs_trained":      epoch,
    "total_steps":         step,
    "elapsed_min":         round(elapsed / 60, 2),
    "n_train":             N_train,
    "n_val":               N_val,
    "model_params":        model.n_parameters(),
    "model_size":          "small",
    "vocab_size":          VOCAB,
    "batch_size":          BATCH,
    "scene_classes":       SCENE_CLASSES,
    "per_class_accuracy":  {
        cls: float(final.get(f"acc_{cls}", 0))
        for cls in SCENE_CLASSES
    },
}
with open("tti_models/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

log("")
log("=" * 60)
log(f"  Training complete in {elapsed / 60:.1f} min")
log(f"  Best val_loss     : {best_loss:.4f}")
log(f"  Final val_acc     : {final['val_acc']:.3f}")
log(f"  Epochs trained    : {epoch}")
log(f"  Total steps       : {step:,}")
log(f"  Best model        : tti_models/best_model.pt")
log(f"  Training history  : tti_models/training_history.json")
log(f"  Training summary  : tti_models/training_summary.json")
log("=" * 60)