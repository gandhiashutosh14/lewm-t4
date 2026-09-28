"""LeWorldModel, written from the paper (Maes et al., arXiv 2603.19312) as plain PyTorch.

One file, no framework: a ViT-tiny encoder, a BatchNorm MLP projector, an action embedder, an
autoregressive transformer predictor with AdaLN-zero conditioning on actions, a BatchNorm MLP on
the predictor's output, and the two-term loss (next-embedding MSE + 0.09 * SIGReg).

Module names are this project's own. ``convert.py`` maps the official checkpoint onto them, and
``tests/test_parity.py`` checks that the two produce the same numbers on the same inputs.

Shapes: pixels (B, T, 3, H, W) normalised with ImageNet statistics; actions (B, T, frameskip *
action_dim); embeddings (B, T, 192).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LeWMConfig:
    image_size: int = 224
    patch_size: int = 14
    dim: int = 192                 # ViT-tiny width and the embedding size
    vit_depth: int = 12
    vit_heads: int = 3
    vit_mlp: int = 768
    proj_hidden: int = 2048
    action_dim: int = 10           # frameskip (5) x raw action dim (2) for TwoRoom
    history: int = 3
    pred_depth: int = 6
    pred_heads: int = 16
    pred_head_dim: int = 64
    pred_mlp: int = 2048
    pred_dropout: float = 0.1
    sigreg_weight: float = 0.09
    sigreg_slices: int = 1024


# --------------------------------------------------------------------------------------- encoder
class ViTBlock(nn.Module):
    """Pre-norm transformer block (LayerNorm eps 1e-12 and exact GELU, as in the reference ViT)."""

    def __init__(self, dim: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim, eps=1e-12)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim, eps=1e-12)
        self.fc1 = nn.Linear(dim, mlp)
        self.fc2 = nn.Linear(mlp, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class ViT(nn.Module):
    def __init__(self, c: LeWMConfig):
        super().__init__()
        self.patch_size = c.patch_size
        self.grid = c.image_size // c.patch_size
        self.patch = nn.Conv2d(3, c.dim, kernel_size=c.patch_size, stride=c.patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, c.dim))
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid + 1, c.dim))
        self.blocks = nn.ModuleList([ViTBlock(c.dim, c.vit_heads, c.vit_mlp) for _ in range(c.vit_depth)])
        self.norm = nn.LayerNorm(c.dim, eps=1e-12)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def pos_for(self, h: int, w: int) -> torch.Tensor:
        """Position embeddings for an h x w patch grid; bicubic interpolation when it differs from training."""
        if h == self.grid and w == self.grid:
            return self.pos
        cls_pos, patch_pos = self.pos[:, :1], self.pos[:, 1:]
        D = patch_pos.shape[-1]
        grid = patch_pos.reshape(1, self.grid, self.grid, D).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
        return torch.cat([cls_pos, grid.permute(0, 2, 3, 1).reshape(1, h * w, D)], dim=1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, dim): the final-layer CLS token."""
        x = self.patch(pixels)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_for(h, w)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


class BNMLP(nn.Module):
    """Linear -> BatchNorm1d -> GELU -> Linear: the projector after the encoder and after the predictor."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = self.fc2(F.gelu(self.bn(self.fc1(x.reshape(-1, shape[-1])))))
        return x.reshape(*shape[:-1], -1)


class ActionEmbedder(nn.Module):
    """Per-step linear mix of the frameskipped action, then a SiLU MLP to the embedding width."""

    def __init__(self, action_dim: int, dim: int, scale: int = 4):
        super().__init__()
        self.mix = nn.Linear(action_dim, action_dim)
        self.fc1 = nn.Linear(action_dim, scale * dim)
        self.fc2 = nn.Linear(scale * dim, dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(self.mix(a.float()))))


# --------------------------------------------------------------------------------------- predictor
class AdaLNBlock(nn.Module):
    """Causal transformer block whose two sub-layers are modulated by the action embedding
    (AdaLN-zero: shift, scale and a gate per sub-layer, the gate initialised to zero). Each
    sub-layer also carries its own affine LayerNorm, applied after the modulation."""

    def __init__(self, c: LeWMConfig):
        super().__init__()
        D, inner = c.dim, c.pred_heads * c.pred_head_dim
        self.heads, self.head_dim, self.dropout = c.pred_heads, c.pred_head_dim, c.pred_dropout
        self.ada = nn.Linear(D, 6 * D)
        self.norm1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.attn_norm = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * inner, bias=False)
        self.attn_out = nn.Linear(inner, D)
        self.mlp_norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, c.pred_mlp)
        self.fc2 = nn.Linear(c.pred_mlp, D)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        drop = self.dropout if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        return F.dropout(self.attn_out(a.transpose(1, 2).reshape(B, T, -1)), drop, self.training)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        drop = self.dropout if self.training else 0.0
        h = F.dropout(F.gelu(self.fc1(self.mlp_norm(x))), drop, self.training)
        return F.dropout(self.fc2(h), drop, self.training)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(F.silu(c)).chunk(6, dim=-1)
        x = x + gate1 * self._attn(self.norm1(x) * (1 + scale1) + shift1)
        return x + gate2 * self._mlp(self.norm2(x) * (1 + scale2) + shift2)


class Predictor(nn.Module):
    def __init__(self, c: LeWMConfig):
        super().__init__()
        self.history = c.history
        self.pos = nn.Parameter(torch.randn(1, c.history, c.dim))
        self.blocks = nn.ModuleList([AdaLNBlock(c) for _ in range(c.pred_depth)])
        self.norm = nn.LayerNorm(c.dim)

    def forward(self, emb: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """emb, act (B, T <= history, dim) -> (B, T, dim); output t predicts the embedding at t + 1."""
        x = emb + self.pos[:, : emb.shape[1]]
        for blk in self.blocks:
            x = blk(x, act)
        return self.norm(x)


# --------------------------------------------------------------------------------------- SIGReg
def sigreg(z: torch.Tensor, slices: int = 1024, t_max: float = 3.0, knots: int = 17,
           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sliced Epps-Pulley distance of each time step's batch of embeddings from N(0, I), averaged.
    z: (T, N, D). Integral over [-t_max, t_max] with window exp(-t^2/2), as in LeJEPA."""
    a = torch.randn(z.shape[-1], slices, device=z.device, dtype=torch.float32, generator=generator)
    a = a / a.norm(dim=0, keepdim=True)
    t = torch.linspace(0.0, t_max, knots, device=z.device)
    dt = t_max / (knots - 1)
    w = torch.full((knots,), 2 * dt, device=z.device)
    w[0] = w[-1] = dt
    phi = torch.exp(-0.5 * t ** 2)
    xt = (z.float() @ a).unsqueeze(-1) * t                           # (T, N, M, K)
    err = (xt.cos().mean(-3) - phi) ** 2 + xt.sin().mean(-3) ** 2     # (T, M, K)
    return (err @ (w * phi) * z.shape[-2]).mean()


# --------------------------------------------------------------------------------------- world model
class LeWM(nn.Module):
    def __init__(self, c: Optional[LeWMConfig] = None):
        super().__init__()
        self.cfg = c = c or LeWMConfig()
        self.encoder = ViT(c)
        self.projector = BNMLP(c.dim, c.proj_hidden)
        self.action_encoder = ActionEmbedder(c.action_dim, c.dim)
        self.predictor = Predictor(c)
        self.pred_proj = BNMLP(c.dim, c.proj_hidden)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, dim)."""
        B, T = pixels.shape[:2]
        return self.projector(self.encoder(pixels.flatten(0, 1))).reshape(B, T, -1)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.pred_proj(self.predictor(emb, act_emb))

    def loss(self, pixels: torch.Tensor, actions: torch.Tensor, generator: Optional[torch.Generator] = None):
        """pixels (B, H + 1, 3, h, w), actions (B, H + 1, action_dim). Teacher-forced next-embedding
        prediction on the first H steps plus SIGReg on every frame's embedding. No stop-gradient."""
        H = self.cfg.history
        emb = self.encode(pixels)
        act = self.action_encoder(actions)
        pred = self.predict(emb[:, :H], act[:, :H])
        pred_loss = (pred - emb[:, 1:H + 1]).pow(2).mean()
        reg = sigreg(emb.transpose(0, 1), slices=self.cfg.sigreg_slices, generator=generator)
        return pred_loss + self.cfg.sigreg_weight * reg, {"pred": pred_loss.detach(), "sigreg": reg.detach()}

    @torch.no_grad()
    def rollout(self, emb0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Imagine forward. emb0 (N, h, dim): encoded history (h <= history); actions (N, h + n, action_dim)
        covering the history and n future steps. Returns (N, h + n + 1, dim) with the n + 1 predictions."""
        h = emb0.shape[1]
        act = self.action_encoder(actions)
        embs = list(emb0.unbind(1))
        for t in range(actions.shape[1] - h + 1):
            lo = max(0, h + t - self.cfg.history)
            ctx = torch.stack(embs[lo:], dim=1)
            embs.append(self.predict(ctx, act[:, lo:h + t])[:, -1])
        return torch.stack(embs, dim=1)

    @torch.no_grad()
    def plan_cost(self, emb0: torch.Tensor, goal_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Squared distance between the last imagined embedding and the goal embedding. emb0 (N, h, dim),
        goal_emb (N, dim), actions (N, h + n, action_dim) -> (N,)."""
        return (self.rollout(emb0, actions)[:, -1] - goal_emb).pow(2).sum(-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def init_like_reference(m: LeWM) -> LeWM:
    """Weight init for training from scratch: truncated normal (std 0.02) for linear layers, zero
    biases, AdaLN gates at zero (set in AdaLNBlock), as is standard for ViTs and DiT-style blocks."""
    for mod in m.modules():
        if isinstance(mod, nn.Linear) and not any(mod is b.ada for b in m.predictor.blocks):
            nn.init.trunc_normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
    fan_in = m.encoder.patch.in_channels * m.cfg.patch_size ** 2
    nn.init.trunc_normal_(m.encoder.patch.weight, std=math.sqrt(1.0 / fan_in))
    return m
