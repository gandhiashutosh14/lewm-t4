"""Train LeWM from scratch, following the published recipe (config/train/lewm.yaml in le-wm):
AdamW (lr 5e-5, weight decay 1e-3), linear warm-up then cosine decay, gradient clipping 1.0, batch
128, history 3 frames + 1 predicted, frameskip 5, SIGReg weight 0.09 with 1,024 projections, 90/10
train/validation split with seed 3072.

The learning-rate schedule. The authors' training library, stable-pretraining (checked at version
0.1.8), uses manual optimisation and steps its LinearWarmupCosineAnnealingLR after every optimiser step
(the ``"interval": "epoch"`` in their training script is not consulted under manual optimisation): the learning
rate rises linearly from 0 over the first 1% of the total steps, then follows a cosine to 0. This
project also steps a linear warm-up from 0 and a cosine decay after every optimiser step, but the
warm-up is an absolute number of steps (``warmup_steps``) and the total is fitted to a time budget.

Differences, all forced by a free Kaggle T4 (16 GB, 12-hour sessions):
  * fp16 autocast with a gradient scaler instead of bf16 (T4 has no bf16 tensor cores); SIGReg is
    computed in fp32, with autocast off;
  * a wall-clock budget: the first ``timing_steps`` steps, all inside the warm-up, are timed, and the
    number of steps that fits the session (at most ``max_epochs`` epochs) becomes the end of the
    cosine decay, so training ends cleanly on schedule; the fit never changes the current learning rate;
  * a validation pass and a resumable checkpoint every 2,500 steps;
  * a T4 x2 session trains two seeds at once, one per GPU, on the same batches (loaded once).
"""
from __future__ import annotations

import json
import math
import os
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
_IMAGENET_ON: Dict[torch.device, tuple] = {}           # the statistics, copied once to each device
OFFICIAL_CONFIG = Path(__file__).with_name("official_config.json")   # published with quentinll/lewm-tworooms


@dataclass
class TrainConfig:
    dataset_path: str = ""
    out_dir: str = "/kaggle/working/run"
    batch: int = 128
    lr: float = 5e-5
    weight_decay: float = 1e-3
    warmup_steps: int = 500                 # linear warm-up from lr 0; the step-time fit leaves it unchanged
    max_epochs: int = 100
    budget_hours: float = 10.5
    clip: float = 1.0
    frameskip: int = 5
    history: int = 3
    img_size: int = 224
    seed: int = 3072
    workers: int = 4
    amp: bool = True
    bn_recalibrate: bool = False            # re-estimate BatchNorm statistics before each validation and save
    resume: bool = False                    # continue from out_dir/s<seed>/state.pt
    max_steps: Optional[int] = None         # for benchmarks and smoke runs


class Normaliser:
    """z-scores a column with dataset statistics (NaN rows at episode boundaries excluded)."""

    def __init__(self, data: np.ndarray):
        d = data[~np.isnan(data).any(axis=1)]
        self.mean = torch.from_numpy(d.mean(0)).float()
        self.std = torch.from_numpy(d.std(0)).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean.repeat(x.shape[-1] // len(self.mean))) / self.std.repeat(x.shape[-1] // len(self.std))


class ActionTransform:
    """The dataset transform: z-scored actions, 0 where missing (NaN at episode ends). A class, not a
    closure, so that DataLoader workers can pickle it where they are spawned (Windows, macOS)."""

    def __init__(self, norm: Normaliser):
        self.norm = norm

    def __call__(self, s: Dict) -> Dict:
        s["action"] = torch.nan_to_num(self.norm(s["action"]), 0.0)
        return s


class EpochBatches(torch.utils.data.BatchSampler):
    """The batches ``DataLoader(shuffle=True, drop_last=True, generator=gen)`` draws (one permutation from
    ``gen`` per epoch), recording the generator state each epoch's permutation starts from. To resume,
    set ``epoch_state`` and ``skip``: the next epoch redraws the interrupted epoch's order and skips the
    batches already trained on, without loading them."""

    def __init__(self, n: int, batch: int, gen: torch.Generator):
        super().__init__(torch.utils.data.RandomSampler(range(n), generator=gen), batch, drop_last=True)
        self.gen, self.epoch_state, self.skip = gen, None, None

    def __iter__(self):
        if self.skip is not None:
            self.gen.set_state(self.epoch_state)
        skip, self.skip, self.epoch_state = self.skip or 0, None, self.gen.get_state()
        for i, b in enumerate(super().__iter__()):
            if i >= skip:
                yield b


def make_loaders(cfg: TrainConfig, ds=None):
    """Training and validation loaders over ``ds`` (by default the HDF5 file at ``cfg.dataset_path``), split
    90/10 as the authors split it: ``random_split`` by fractions with a generator seeded with ``cfg.seed``,
    which then shuffles the training batches. Returns (train loader, validation loader, number of windows)."""
    if ds is None:
        import stable_worldmodel as swm
        ds = swm.data.HDF5Dataset(path=cfg.dataset_path, frameskip=cfg.frameskip, num_steps=cfg.history + 1,
                                  keys_to_load=["pixels", "action"], keys_to_cache=["action"])
    ds.transform = ActionTransform(Normaliser(ds.get_col_data("action")))
    gen = torch.Generator().manual_seed(cfg.seed)
    train, val = torch.utils.data.random_split(ds, [0.9, 0.1], generator=gen)
    kw = dict(num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
              prefetch_factor=4 if cfg.workers > 0 else None)
    return (torch.utils.data.DataLoader(train, batch_sampler=EpochBatches(len(train), cfg.batch, gen), generator=gen, **kw),
            torch.utils.data.DataLoader(val, batch_size=cfg.batch, shuffle=False, drop_last=False,
                                        generator=torch.Generator().manual_seed(cfg.seed), **kw),   # not the global RNG
            len(ds))


def prepare_pixels(px: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 (B, T, 3, H, W) on the GPU -> ImageNet-normalised float at size x size."""
    x = px.float() / 255.0
    B, T = x.shape[:2]
    if x.shape[-1] != size:
        x = torch.nn.functional.interpolate(x.flatten(0, 1), size=(size, size), mode="bilinear", antialias=True,
                                            align_corners=False).reshape(B, T, 3, size, size)
    if x.device not in _IMAGENET_ON:
        _IMAGENET_ON[x.device] = IMAGENET_MEAN.to(x.device), IMAGENET_STD.to(x.device)
    mean, std = _IMAGENET_ON[x.device]
    return (x - mean) / std


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    """Learning rate for optimiser step ``step`` (from 0): linear from 0 to ``base`` over ``warmup`` steps,
    then a cosine to 0 at ``total``. Only the cosine depends on ``total``, so re-fitting ``total`` during
    the warm-up leaves the learning rate unchanged."""
    if step < warmup:
        return base * step / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


class Replica:
    """One model, its optimiser and its gradient scaler on one device. A step is split in two, ``backward``
    (forward, backward, unscale, clip) and ``apply`` (optimiser step, scale update), so that the loop can
    queue every GPU's work before the first ``GradScaler.step``, which reads its overflow check back to
    the host and so waits for its own GPU to finish."""

    def __init__(self, seed: int, device: str, cfg: TrainConfig):
        torch.manual_seed(seed)
        self.seed, self.device, self.cuda = seed, device, torch.device(device).type == "cuda"
        self.model = init_like_reference(LeWM(LeWMConfig(image_size=cfg.img_size))).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        self.gen = torch.Generator(device=device)
        self.last = None

    def backward(self, batch, step: int, lr: float, cfg: TrainConfig) -> None:
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
        self.last = (loss.detach(), parts)             # no host sync here; read when logging

    def apply(self) -> None:
        self.scaler.step(self.opt)                     # skipped when the fp16 gradients overflowed
        self.scaler.update()

    def state(self) -> Dict:
        return {"optimizer": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                "device_rng": torch.cuda.get_rng_state(self.device) if self.cuda else None}

    def load(self, st: Dict) -> None:
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optimizer"])
        self.scaler.load_state_dict(st["scaler"])
        if self.cuda:
            torch.cuda.set_rng_state(st["device_rng"], self.device)


def train(cfg: TrainConfig, log=print, seeds: Optional[list] = None, devices: Optional[list] = None,
          timing_steps: int = 300, eval_every: int = 2500) -> Dict:
    """Train one model per (seed, device) on the same stream of batches (data is loaded once).

    The schedule is fitted to the time budget by steps: the first ``timing_steps`` steps (fewer than
    ``cfg.warmup_steps``) are timed, and the number of steps that fits ``budget_hours``, capped at
    ``max_epochs`` epochs, becomes the end of the cosine decay. The warm-up does not depend on it, so
    the learning rate is continuous across the fit. ``max_steps`` overrides the budget (benchmarks,
    smoke runs). With ``cfg.resume``, every model continues from its ``state.pt`` (weights, optimiser,
    scaler, schedule, random number generators, position in the data), as if the run had not stopped."""
    assert cfg.max_steps or 20 < timing_steps < cfg.warmup_steps, "the step-time fit must fall inside the warm-up"
    seeds = seeds or [cfg.seed]
    devices = devices or [f"cuda:{i}" for i in range(len(seeds))]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_dl, val_dl, n = make_loaders(cfg)
    batches = train_dl.batch_sampler
    reps = [Replica(sd, dev, cfg) for sd, dev in zip(seeds, devices)]
    for r in reps:                         # torch.manual_seed in Replica() reseeds every GPU: give each its own run's seed
        if r.cuda:
            with torch.cuda.device(r.device):
                torch.cuda.manual_seed(r.seed)
    bn_dl = None
    if cfg.bn_recalibrate:                 # the same 50 random training batches every time
        pick = torch.randperm(len(train_dl.dataset), generator=torch.Generator().manual_seed(cfg.seed))[:50 * cfg.batch]
        bn_dl = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dl.dataset, pick.tolist()), batch_size=cfg.batch,
                                            num_workers=cfg.workers, generator=torch.Generator().manual_seed(cfg.seed))
    steps_per_epoch = len(train_dl)
    total, warmup, step, in_epoch, seconds = cfg.max_steps or steps_per_epoch * cfg.max_epochs, cfg.warmup_steps, 0, 0, 0.0
    history = {r.seed: [] for r in reps}
    if cfg.resume:
        states = [torch.load(out / f"s{r.seed}" / "state.pt", map_location="cpu", weights_only=True) for r in reps]
        for r, st in zip(reps, states):
            r.load(st)
            history[r.seed] = st["history"]
        st = states[0]
        assert all(s["step"] == st["step"] for s in states), "the models were saved at different steps"
        step, total, warmup, in_epoch, seconds = st["step"], st["total"], st["warmup"], st["batch_in_epoch"], st["seconds"]
        assert cfg.max_steps or step >= timing_steps, "resume needs a checkpoint saved after the step-time fit"
        torch.set_rng_state(st["cpu_rng"])
        batches.epoch_state, batches.skip = st["epoch_state"], in_epoch
        log(f"resuming at step {step}/{total}, batch {in_epoch} of its epoch")
    t_start, t_budget0 = time.time() - seconds, None
    log(f"windows {n}, steps/epoch {steps_per_epoch}, models {len(reps)} on {devices}")
    while step < total:
        for batch in train_dl:
            in_epoch += 1
            lr = lr_at(step, total, warmup, cfg.lr)
            for r in reps:                         # queue every model's forward and backward pass ...
                r.backward(batch, step, lr, cfg)
            for r in reps:                         # ... before the optimiser steps, which wait for their GPU
                r.apply()
            step += 1
            if step == 20:
                t_budget0 = time.time()                     # exclude start-up from the timing
            if step == timing_steps and not cfg.max_steps:
                sec = (time.time() - t_budget0) / (timing_steps - 20)
                remaining = cfg.budget_hours * 3600 - (time.time() - t_start)
                total = min(step + max(0, int(remaining / (sec * 1.04))), steps_per_epoch * cfg.max_epochs)  # 4% for validation, saves
                log(f"timing: {sec:.3f} s/step -> {total} steps ({total / steps_per_epoch:.2f} epochs) fit in {cfg.budget_hours} h")
            if step % 100 == 0:
                msg = " | ".join(f"s{r.seed} loss {float(r.last[0]):.4f} pred {float(r.last[1]['pred']):.4f} "
                                 f"sigreg {float(r.last[1]['sigreg']):.3f}" for r in reps)
                log(f"step {step}/{total} lr {lr:.2e} {(time.time() - t_start) / step:.3f}s/step | {msg}")
            if step % eval_every == 0 or step >= total:
                for r in reps:
                    if bn_dl is not None:
                        recalibrate_batchnorm(r.model, bn_dl, 50, r.device)
                    rec = {"step": step, "epoch": step / steps_per_epoch, "hours": (time.time() - t_start) / 3600,
                           "train_loss": float(r.last[0]), **validate(r.model, val_dl, cfg, device=r.device)}
                    history[r.seed].append(rec)
                    log(f"s{r.seed} {json.dumps(rec)}")
                    save(r.model, out / f"s{r.seed}", cfg, history[r.seed], name="last",
                         state={**r.state(), "step": step, "total": total, "warmup": warmup, "seconds": time.time() - t_start,
                                "cpu_rng": torch.get_rng_state(), "epoch_state": batches.epoch_state, "batch_in_epoch": in_epoch})
            if step >= total:
                break
        else:
            in_epoch = 0
    return {"history": history, "hours": (time.time() - t_start) / 3600, "steps": step, "total_steps": total,
            "steps_per_epoch": steps_per_epoch, "config": asdict(cfg), "seeds": seeds}


@torch.no_grad()
def recalibrate_batchnorm(model: LeWM, loader, n_batches: int, device) -> None:
    """Re-estimate every BatchNorm layer's running statistics from ``n_batches`` training batches, as a
    plain average over them (momentum None) instead of the exponential average kept during training;
    each layer's momentum is restored afterwards. Only the BatchNorm layers run in training mode, so
    dropout is off, as it is when the model is evaluated, and no random numbers are drawn. Off by default
    (``TrainConfig.bn_recalibrate``): the authors do not do this, and evaluate and plan with the running
    statistics accumulated during training."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta, was_training = [bn.momentum for bn in bns], model.training
    model.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    H = model.cfg.history
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        emb = model.encode(prepare_pixels(batch["pixels"].to(device, non_blocking=True), model.cfg.image_size))
        model.predict(emb[:, :H], model.action_encoder(batch["action"].to(device, non_blocking=True))[:, :H])
    for bn, mom in zip(bns, momenta):
        bn.momentum = mom
    model.train(was_training)


@torch.no_grad()
def validate(model: LeWM, dl, cfg: TrainConfig, max_batches: int = 50, device: str = "cuda") -> Dict:
    """Eval-mode prediction loss and SIGReg over up to ``max_batches`` validation batches. SIGReg's random
    directions come from a generator seeded the same way on every call, so checkpoints are compared on
    the same directions."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    tot, n = {"pred": 0.0, "sigreg": 0.0}, 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        px = prepare_pixels(batch["pixels"].to(device, non_blocking=True), cfg.img_size)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            _, parts = model.loss(px, batch["action"].to(device, non_blocking=True), generator=gen)
        tot["pred"] += float(parts["pred"])
        tot["sigreg"] += float(parts["sigreg"])
        n += 1
    return {f"val_{k}": v / max(n, 1) for k, v in tot.items()}


def save(model: LeWM, out: Path, cfg: TrainConfig, history, name: str, state: Optional[Dict] = None) -> None:
    """Write ``<name>.pt`` (this project's names), ``<name>_official_layout.pt`` with the reference
    ``config.json`` beside it, and ``history.json``; given the training ``state``, also the resumable
    ``state.pt`` (through a temporary file, so an interrupted save keeps the previous one).
    stable-worldmodel's ``load_pretrained`` loads the official layout when given the path of that .pt
    file (the folder holds more than one)."""
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, out / f"{name}.pt")
    torch.save(to_official(sd), out / f"{name}_official_layout.pt")
    conf = json.loads(OFFICIAL_CONFIG.read_text(encoding="utf-8"))
    conf["encoder"]["image_size"] = model.cfg.image_size
    (out / "config.json").write_text(json.dumps(conf, indent=4) + "\n", encoding="utf-8")
    (out / "history.json").write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=2))
    if state is not None:
        torch.save({"model": sd, "history": history, "config": asdict(cfg), **state}, out / "state.pt.tmp")
        os.replace(out / "state.pt.tmp", out / "state.pt")
