"""Train LeWM from scratch, following the published recipe (config/train/lewm.yaml in le-wm):
AdamW (lr 5e-5, weight decay 1e-3), linear warm-up then cosine decay, gradient clipping 1.0, batch
128, history 3 frames + 1 predicted, frameskip 5, SIGReg weight 0.09 with 1,024 projections, 90/10
train/validation split with seed 3072.

Differences, all forced by a free Kaggle T4 (16 GB, 12-hour sessions):
  * fp16 autocast with a gradient scaler instead of bf16 (T4 has no bf16 tensor cores); SIGReg is
    computed in fp32 either way;
  * a wall-clock budget: training stops cleanly before the session limit, and the learning-rate
    schedule is laid out over the epochs that fit, measured on the first epoch;
  * a checkpoint after every epoch, so a killed session loses at most one epoch.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .convert import to_official
from .model import LeWM, LeWMConfig, init_like_reference

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@dataclass
class TrainConfig:
    dataset_path: str = ""
    out_dir: str = "/kaggle/working/run"
    batch: int = 128
    lr: float = 5e-5
    weight_decay: float = 1e-3
    warmup_frac: float = 0.05
    max_epochs: int = 100
    budget_hours: float = 10.5
    clip: float = 1.0
    frameskip: int = 5
    history: int = 3
    img_size: int = 224
    seed: int = 3072
    workers: int = 4
    amp: bool = True
    max_steps: Optional[int] = None         # for benchmarks and smoke runs


class Normaliser:
    """z-scores a column with dataset statistics (NaN rows at episode boundaries excluded)."""

    def __init__(self, data: np.ndarray):
        d = data[~np.isnan(data).any(axis=1)]
        self.mean = torch.from_numpy(d.mean(0)).float()
        self.std = torch.from_numpy(d.std(0)).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean.repeat(x.shape[-1] // len(self.mean))) / self.std.repeat(x.shape[-1] // len(self.std))


def make_loaders(cfg: TrainConfig):
    import stable_worldmodel as swm

    ds = swm.data.HDF5Dataset(path=cfg.dataset_path, frameskip=cfg.frameskip, num_steps=cfg.history + 1,
                              keys_to_load=["pixels", "action"], keys_to_cache=["action"])
    act_norm = Normaliser(ds.get_col_data("action"))

    def transform(s):
        s["action"] = torch.nan_to_num(act_norm(s["action"]), 0.0)
        return s

    ds.transform = transform
    gen = torch.Generator().manual_seed(cfg.seed)
    n_val = int(round(0.1 * len(ds)))
    train, val = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    kw = dict(batch_size=cfg.batch, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
              prefetch_factor=4 if cfg.workers > 0 else None)
    return (torch.utils.data.DataLoader(train, shuffle=True, drop_last=True, generator=gen, **kw),
            torch.utils.data.DataLoader(val, shuffle=False, drop_last=False, **kw), len(ds))


def prepare_pixels(px: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 (B, T, 3, H, W) on the GPU -> ImageNet-normalised float at size x size."""
    x = px.float() / 255.0
    B, T = x.shape[:2]
    if x.shape[-1] != size:
        x = torch.nn.functional.interpolate(x.flatten(0, 1), size=(size, size), mode="bilinear", antialias=True,
                                            align_corners=False).reshape(B, T, 3, size, size)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def train(cfg: TrainConfig, log=print) -> Dict:
    torch.manual_seed(cfg.seed)
    device = "cuda"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_dl, val_dl, n = make_loaders(cfg)
    model = init_like_reference(LeWM(LeWMConfig(image_size=cfg.img_size))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    steps_per_epoch = len(train_dl)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.max_epochs
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    history, step, t_start = [], 0, time.time()
    gen = torch.Generator(device=device)
    log(f"windows {n}, steps/epoch {steps_per_epoch}, planned steps {total_steps}")
    for epoch in range(cfg.max_epochs):
        model.train()
        t_epoch, sums = time.time(), {"loss": 0.0, "pred": 0.0, "sigreg": 0.0}
        for i, batch in enumerate(train_dl):
            px = prepare_pixels(batch["pixels"].to(device, non_blocking=True), cfg.img_size)
            act = batch["action"].to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup, cfg.lr)
            gen.manual_seed(cfg.seed * 1_000_003 + step)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                loss, parts = model.loss(px, act, generator=gen)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            sums["loss"] += float(loss.detach())
            sums["pred"] += float(parts["pred"])
            sums["sigreg"] += float(parts["sigreg"])
            if step % 100 == 0:
                log(f"step {step} loss {float(loss):.4f} pred {float(parts['pred']):.4f} sigreg {float(parts['sigreg']):.3f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {(time.time() - t_start) / step:.3f}s/step")
            if cfg.max_steps and step >= cfg.max_steps:
                break
        k = i + 1
        rec = {"epoch": epoch + 1, "step": step, **{f"train_{a}": b / k for a, b in sums.items()},
               **validate(model, val_dl, cfg), "epoch_minutes": (time.time() - t_epoch) / 60}
        history.append(rec)
        log(json.dumps(rec))
        save(model, out, cfg, history, name="last")
        if epoch == 0 and not cfg.max_steps:          # fit the schedule to the time budget
            fit = int((cfg.budget_hours * 60 - rec["epoch_minutes"] * 1.5) // rec["epoch_minutes"]) + 1
            epochs = max(1, min(cfg.max_epochs, fit))
            total_steps = steps_per_epoch * epochs
            warmup = min(warmup, max(1, int(cfg.warmup_frac * total_steps)))
            cfg.max_epochs = epochs
            log(f"time budget: {epochs} epochs fit in {cfg.budget_hours} h")
        if (cfg.max_steps and step >= cfg.max_steps) or epoch + 1 >= cfg.max_epochs:
            break
    return {"history": history, "hours": (time.time() - t_start) / 3600, "steps": step, "config": asdict(cfg)}


@torch.no_grad()
def validate(model: LeWM, dl, cfg: TrainConfig, max_batches: int = 50) -> Dict:
    model.eval()
    tot, n = {"pred": 0.0, "sigreg": 0.0}, 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        px = prepare_pixels(batch["pixels"].cuda(non_blocking=True), cfg.img_size)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            _, parts = model.loss(px, batch["action"].cuda(non_blocking=True))
        tot["pred"] += float(parts["pred"])
        tot["sigreg"] += float(parts["sigreg"])
        n += 1
    return {f"val_{k}": v / max(n, 1) for k, v in tot.items()}


def save(model: LeWM, out: Path, cfg: TrainConfig, history, name: str) -> None:
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, out / f"{name}.pt")
    torch.save(to_official(sd), out / f"{name}_official_layout.pt")
    (out / "history.json").write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=2))
