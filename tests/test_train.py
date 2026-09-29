"""CPU checks of the training code: the learning rate across the step-time fit, action normalisation, the
authors' data split and batch order, and resuming a run from its checkpoint."""
import json

import numpy as np
import pytest
import torch

from lewm_t4 import train as tr
from lewm_t4.model import LeWM, LeWMConfig
from lewm_t4.train import EpochBatches, Normaliser, TrainConfig, lr_at, make_loaders

TINY = dict(dim=32, vit_depth=1, vit_heads=2, vit_mlp=64, proj_hidden=64, pred_depth=1, pred_heads=2, pred_head_dim=16,
            pred_mlp=64, sigreg_slices=32)


class Windows:
    """Stands in for stable-worldmodel's HDF5Dataset: ``n`` windows of 4 frames and 4 x 5 raw 2-D actions. As
    there, the transform sees the raw (20, 2) actions, which are then grouped by step into (4, 10)."""

    def __init__(self, n: int, size: int = 28):
        g = torch.Generator().manual_seed(0)
        self.px = torch.randint(0, 256, (n, 4, 3, size, size), dtype=torch.uint8, generator=g)
        self.act = torch.randn(n, 20, 2, generator=g) * torch.tensor([0.5, 2.0]) + torch.tensor([1.0, -1.0])
        self.transform = None

    def __len__(self):
        return len(self.px)

    def get_col_data(self, col):
        return self.act.reshape(-1, 2).numpy()

    def __getitem__(self, i):
        s = self.transform({"pixels": self.px[i], "action": self.act[i].clone()})
        s["action"] = s["action"].reshape(4, -1)
        return s


def test_learning_rate_is_continuous_across_the_step_time_fit():
    base, warmup, fit_at = 5e-5, 500, 300
    provisional, fitted = 513_800, 18_498             # 100 epochs of the reported run, then the steps that fit its budget
    lrs = [lr_at(s, provisional if s < fit_at else fitted, warmup, base) for s in range(fitted)]
    assert lrs[0] == 0.0 and max(lrs) == base and lrs[-1] < 1e-3 * base
    assert max(abs(b - a) for a, b in zip(lrs, lrs[1:])) <= base / warmup * (1 + 1e-9)   # no jump, at the fit or anywhere
    with pytest.raises(AssertionError):                # a fit after the warm-up would change the learning rate
        tr.train(TrainConfig(warmup_steps=300), timing_steps=300)


def test_normaliser_repeats_per_dimension_statistics_across_a_frameskipped_window():
    raw = np.random.default_rng(0).normal([1.0, -2.0], [0.5, 3.0], size=(1000, 2)).astype(np.float32)
    raw[::97] = np.nan                                 # episode boundaries
    norm = Normaliser(raw)
    kept = raw[~np.isnan(raw).any(axis=1)]
    assert norm.mean.shape == (2,) and torch.allclose(norm.mean, torch.from_numpy(kept.mean(0)))
    steps = torch.from_numpy(raw[1:21])                # 4 steps x frameskip 5 of raw (x, y) actions, none missing
    want = (steps - torch.from_numpy(kept.mean(0))) / torch.from_numpy(kept.std(0))
    assert torch.allclose(norm(steps.reshape(4, 10)), want.reshape(4, 10))   # the (2,) statistics repeated 5 times
    assert torch.allclose(norm(steps), want)


def test_split_is_random_split_by_fractions():
    cfg = TrainConfig(batch=4, workers=0)
    train_dl, val_dl, n = make_loaders(cfg, Windows(1006))
    want = torch.utils.data.random_split(range(1006), [0.9, 0.1], generator=torch.Generator().manual_seed(cfg.seed))
    assert n == 1006 and (len(train_dl.dataset), len(val_dl.dataset)) == (906, 100)   # rounding 10% would give 905 / 101
    assert train_dl.dataset.indices == want[0].indices and val_dl.dataset.indices == want[1].indices


def test_training_batches_come_in_the_shuffled_loader_order():
    ga, gb = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    ours = torch.utils.data.DataLoader(range(50), batch_sampler=EpochBatches(50, 8, ga), generator=ga)
    theirs = torch.utils.data.DataLoader(range(50), batch_size=8, shuffle=True, drop_last=True, generator=gb)
    for _ in range(3):
        assert [b.tolist() for b in ours] == [b.tolist() for b in theirs]


@torch.no_grad()
def test_batchnorm_recalibration_averages_batch_statistics():
    torch.manual_seed(0)
    m = LeWM(LeWMConfig(image_size=28, **TINY)).train()
    loader = [{"pixels": torch.randint(0, 256, (4, 4, 3, 28, 28), dtype=torch.uint8), "action": torch.randn(4, 4, 10)} for _ in range(3)]
    rng = torch.get_rng_state()
    tr.recalibrate_batchnorm(m, loader, n_batches=2, device="cpu")
    means = [m.projector.fc1(m.encoder(tr.prepare_pixels(b["pixels"], 28).flatten(0, 1))).mean(0) for b in loader[:2]]
    bn = m.projector.bn
    assert torch.allclose(bn.running_mean, torch.stack(means).mean(0), atol=1e-6) and int(bn.num_batches_tracked) == 2
    assert bn.momentum == 0.1 and m.training and torch.equal(rng, torch.get_rng_state())   # restored; no dropout draws


@pytest.mark.parametrize("seeds", [[0], [0, 1]])
def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, monkeypatch, seeds):
    monkeypatch.setattr(tr, "LeWMConfig", lambda image_size: LeWMConfig(image_size=image_size, **TINY))
    ds, make, save = Windows(48), tr.make_loaders, tr.save
    monkeypatch.setattr(tr, "make_loaders", lambda cfg: make(cfg, ds))

    def run(out, **kw):                                # 44 training windows: 11 steps an epoch, 14 steps in all
        cfg = TrainConfig(out_dir=str(out), batch=4, workers=0, amp=False, img_size=28, max_steps=14, warmup_steps=4, seed=0, **kw)
        return tr.train(cfg, log=lambda *a: None, seeds=seeds, devices=["cpu"] * len(seeds), eval_every=5)

    class Stop(Exception):
        pass

    def save_then_stop(model, out, *a, state=None, **kw):     # stop once every model has saved step 5
        save(model, out, *a, state=state, **kw)
        if state["step"] == 5 and out.name == f"s{seeds[-1]}":
            raise Stop

    whole = run(tmp_path / "whole")
    monkeypatch.setattr(tr, "save", save_then_stop)
    with pytest.raises(Stop):
        run(tmp_path / "cut")
    monkeypatch.setattr(tr, "save", save)
    resumed = run(tmp_path / "cut", resume=True)       # mid-epoch, then across the epoch boundary
    assert resumed["steps"] == 14
    for s in seeds:
        a, b = (torch.load(tmp_path / d / f"s{s}" / "last.pt", weights_only=True) for d in ("whole", "cut"))
        assert all(torch.equal(a[k], b[k]) for k in a)
        assert [h["val_pred"] for h in resumed["history"][s]] == [h["val_pred"] for h in whole["history"][s]]
    conf = json.loads((tmp_path / "cut" / "s0" / "config.json").read_text(encoding="utf-8"))
    assert conf["encoder"]["image_size"] == 28 and (tmp_path / "cut" / "s0" / "last_official_layout.pt").exists()
