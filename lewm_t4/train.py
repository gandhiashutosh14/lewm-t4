"""Train LeWM from scratch, following the published recipe (config/train/lewm.yaml in le-wm):
AdamW (lr 5e-5, weight decay 1e-3), linear warm-up then cosine decay, gradient clipping 1.0, batch
128, history 3 frames + 1 predicted, frameskip 5, SIGReg weight 0.09 with 1,024 projections, 90/10
train/validation split with seed 3072.

Differences, all forced by a free Kaggle T4 (16 GB, 12-hour sessions):
  * fp16 autocast with a gradient scaler instead of bf16 (T4 has no bf16 tensor cores); SIGReg is
    computed in fp32 either way;
  * a wall-clock budget: the first few hundred steps are timed and the warm-up + cosine schedule is
    laid out over the number of steps that fit the session, so training ends cleanly on schedule;
  * a validation pass and a checkpoint every 2,500 steps;
  * a T4 x2 session trains two seeds at once, one per GPU, on the same batches (loaded once).
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


class Replica:
    """One model, its optimiser and its gradient scaler on one GPU."""

    def __init__(self, seed: int, device: str, cfg: TrainConfig):
        torch.manual_seed(seed)
        self.seed, self.device = seed, device
        self.model = init_like_reference(LeWM(LeWMConfig(image_size=cfg.img_size))).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        self.gen = torch.Generator(device=device)
        self.last = None

    def step(self, batch, step: int, lr: float, cfg: TrainConfig) -> None:
        px = prepare_pixels(batch["pixels"].to(self.device, non_blocking=True), cfg.img_size)
        act = batch["action"].to(self.device, non_blocking=True)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.gen.manual_seed(self.seed * 1_000_003 + step)
        self.model.train()
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            loss, parts = self.model.loss(px, act, generator=self.gen)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.last = (loss.detach(), parts)             # no host sync here; read when logging


def train(cfg: TrainConfig, log=print, seeds: Optional[list] = None, devices: Optional[list] = None,
          timing_steps: int = 300, eval_every: int = 2500) -> Dict:
    """Train one model per (seed, device) on the same stream of batches (data is loaded once).

    The schedule is fitted to the time budget by steps: the first ``timing_steps`` steps are timed,
    the total number of steps that fits ``budget_hours`` is fixed from that, and warm-up plus cosine
    decay are laid out over it. ``max_steps`` overrides the budget (benchmarks, smoke runs)."""
    seeds = seeds or [cfg.seed]
    devices = devices or [f"cuda:{i}" for i in range(len(seeds))]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_dl, val_dl, n = make_loaders(cfg)
    reps = [Replica(sd, dev, cfg) for sd, dev in zip(seeds, devices)]
    steps_per_epoch = len(train_dl)
    total = cfg.max_steps or steps_per_epoch * cfg.max_epochs
    warmup = max(1, int(cfg.warmup_frac * total))
    history = {r.seed: [] for r in reps}
    step, t_start, t_budget0 = 0, time.time(), None
    log(f"windows {n}, steps/epoch {steps_per_epoch}, models {len(reps)} on {devices}")
    done = False
    while not done:
        for batch in train_dl:
            lr = lr_at(step, total, warmup, cfg.lr)
            for r in reps:
                r.step(batch, step, lr, cfg)
            step += 1
            if step == 20:
                t_budget0 = time.time()                     # exclude start-up from the timing
            if step == timing_steps and not cfg.max_steps:
                sec = (time.time() - t_budget0) / (timing_steps - 20)
                remaining = cfg.budget_hours * 3600 - (time.time() - t_start)
                total = step + int(remaining / (sec * 1.04))  # 4% for validation and checkpoints
                warmup = max(1, int(cfg.warmup_frac * total))
                log(f"timing: {sec:.3f} s/step -> {total} steps ({total / steps_per_epoch:.2f} epochs) fit in {cfg.budget_hours} h")
            if step % 100 == 0:
                msg = " | ".join(f"s{r.seed} loss {float(r.last[0]):.4f} pred {float(r.last[1]['pred']):.4f} "
                                 f"sigreg {float(r.last[1]['sigreg']):.3f}" for r in reps)
                log(f"step {step}/{total} lr {lr:.2e} {(time.time() - t_start) / step:.3f}s/step | {msg}")
            if step % eval_every == 0 or step >= total:
                for r in reps:
                    rec = {"step": step, "epoch": step / steps_per_epoch, "hours": (time.time() - t_start) / 3600,
                           "train_loss": float(r.last[0]), **validate(r.model, val_dl, cfg, device=r.device)}
                    history[r.seed].append(rec)
                    log(f"s{r.seed} {json.dumps(rec)}")
                    save(r.model, out / f"s{r.seed}", cfg, history[r.seed], name="last")
            if step >= total:
                done = True
                break
    return {"history": history, "hours": (time.time() - t_start) / 3600, "steps": step, "total_steps": total,
            "steps_per_epoch": steps_per_epoch, "config": asdict(cfg), "seeds": seeds}


@torch.no_grad()
def validate(model: LeWM, dl, cfg: TrainConfig, max_batches: int = 50, device: str = "cuda") -> Dict:
    model.eval()
    tot, n = {"pred": 0.0, "sigreg": 0.0}, 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        px = prepare_pixels(batch["pixels"].to(device, non_blocking=True), cfg.img_size)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            _, parts = model.loss(px, batch["action"].to(device, non_blocking=True))
        tot["pred"] += float(parts["pred"])
        tot["sigreg"] += float(parts["sigreg"])
        n += 1
    return {f"val_{k}": v / max(n, 1) for k, v in tot.items()}


def save(model: LeWM, out: Path, cfg: TrainConfig, history, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, out / f"{name}.pt")
    torch.save(to_official(sd), out / f"{name}_official_layout.pt")
    (out / "history.json").write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=2))
