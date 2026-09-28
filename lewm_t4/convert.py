"""Map the official LeWM checkpoint (``weights.pt`` from the Hugging Face model repos, e.g.
quentinll/lewm-tworooms) onto this project's module names, and back.

The official encoder is a Hugging Face ``ViTModel`` (separate query/key/value projections); this
project fuses them into one ``qkv`` projection. The action embedder's 1x1 Conv1d becomes a Linear.
Everything else is a rename. ``to_official`` is the exact inverse, so a model trained here can be
evaluated with the reference tooling.
"""
from __future__ import annotations

import re
from typing import Dict

import torch

Tensor = torch.Tensor


def from_official(sd: Dict[str, Tensor], vit_depth: int = 12, pred_depth: int = 6) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    used = set()

    def take(k: str) -> Tensor:
        used.add(k)
        return sd[k]

    e = "encoder."
    out["encoder.patch.weight"] = take(e + "embeddings.patch_embeddings.projection.weight")
    out["encoder.patch.bias"] = take(e + "embeddings.patch_embeddings.projection.bias")
    out["encoder.cls"] = take(e + "embeddings.cls_token")
    out["encoder.pos"] = take(e + "embeddings.position_embeddings")
    for i in range(vit_depth):
        L = f"{e}encoder.layer.{i}."
        o = f"encoder.blocks.{i}."
        att = L + "attention.attention."
        out[o + "qkv.weight"] = torch.cat([take(att + n + ".weight") for n in ("query", "key", "value")], 0)
        out[o + "qkv.bias"] = torch.cat([take(att + n + ".bias") for n in ("query", "key", "value")], 0)
        for mine, theirs in (("proj", "attention.output.dense"), ("fc1", "intermediate.dense"), ("fc2", "output.dense"),
                             ("ln1", "layernorm_before"), ("ln2", "layernorm_after")):
            out[o + mine + ".weight"] = take(L + theirs + ".weight")
            out[o + mine + ".bias"] = take(L + theirs + ".bias")
    out["encoder.norm.weight"] = take(e + "layernorm.weight")
    out["encoder.norm.bias"] = take(e + "layernorm.bias")

    for mine, theirs in (("projector", "projector"), ("pred_proj", "pred_proj")):
        for a, b in (("fc1", "net.0"), ("bn", "net.1"), ("fc2", "net.3")):
            for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                k = f"{theirs}.{b}.{suffix}"
                if k in sd:
                    out[f"{mine}.{a}.{suffix}"] = take(k)

    out["action_encoder.mix.weight"] = take("action_encoder.patch_embed.weight").squeeze(-1)
    out["action_encoder.mix.bias"] = take("action_encoder.patch_embed.bias")
    for a, b in (("fc1", "embed.0"), ("fc2", "embed.2")):
        out[f"action_encoder.{a}.weight"] = take(f"action_encoder.{b}.weight")
        out[f"action_encoder.{a}.bias"] = take(f"action_encoder.{b}.bias")

    out["predictor.pos"] = take("predictor.pos_embedding")
    out["predictor.norm.weight"] = take("predictor.transformer.norm.weight")
    out["predictor.norm.bias"] = take("predictor.transformer.norm.bias")
    for i in range(pred_depth):
        L = f"predictor.transformer.layers.{i}."
        o = f"predictor.blocks.{i}."
        pairs = (("ada.weight", "adaLN_modulation.1.weight"), ("ada.bias", "adaLN_modulation.1.bias"),
                 ("attn_norm.weight", "attn.norm.weight"), ("attn_norm.bias", "attn.norm.bias"),
                 ("qkv.weight", "attn.to_qkv.weight"), ("attn_out.weight", "attn.to_out.0.weight"),
                 ("attn_out.bias", "attn.to_out.0.bias"), ("mlp_norm.weight", "mlp.net.0.weight"),
                 ("mlp_norm.bias", "mlp.net.0.bias"), ("fc1.weight", "mlp.net.1.weight"), ("fc1.bias", "mlp.net.1.bias"),
                 ("fc2.weight", "mlp.net.4.weight"), ("fc2.bias", "mlp.net.4.bias"))
        for mine, theirs in pairs:
            out[o + mine] = take(L + theirs)
    unused = sorted(set(sd) - used)
    if unused:
        raise KeyError(f"official keys not mapped: {unused[:8]}{' ...' if len(unused) > 8 else ''}")
    return out


def to_official(sd: Dict[str, Tensor], vit_depth: int = 12, pred_depth: int = 6) -> Dict[str, Tensor]:
    """Inverse of from_official (built by running the forward mapping on key names)."""
    out: Dict[str, Tensor] = {}
    D = sd["encoder.cls"].shape[-1]
    for k, v in sd.items():
        m = re.match(r"encoder\.blocks\.(\d+)\.qkv\.(weight|bias)", k)
        if m:
            i, kind = m.groups()
            for j, n in enumerate(("query", "key", "value")):
                out[f"encoder.encoder.layer.{i}.attention.attention.{n}.{kind}"] = v[j * D:(j + 1) * D]
            continue
        if k == "action_encoder.mix.weight":
            out["action_encoder.patch_embed.weight"] = v.unsqueeze(-1)
            continue
        out[_reverse_name(k)] = v
    return out


def _reverse_name(k: str) -> str:
    rules = [
        (r"^encoder\.patch\.", "encoder.embeddings.patch_embeddings.projection."),
        (r"^encoder\.cls$", "encoder.embeddings.cls_token"),
        (r"^encoder\.pos$", "encoder.embeddings.position_embeddings"),
        (r"^encoder\.norm\.", "encoder.layernorm."),
        (r"^encoder\.blocks\.(\d+)\.proj\.", r"encoder.encoder.layer.\1.attention.output.dense."),
        (r"^encoder\.blocks\.(\d+)\.fc1\.", r"encoder.encoder.layer.\1.intermediate.dense."),
        (r"^encoder\.blocks\.(\d+)\.fc2\.", r"encoder.encoder.layer.\1.output.dense."),
        (r"^encoder\.blocks\.(\d+)\.ln1\.", r"encoder.encoder.layer.\1.layernorm_before."),
        (r"^encoder\.blocks\.(\d+)\.ln2\.", r"encoder.encoder.layer.\1.layernorm_after."),
        (r"^(projector|pred_proj)\.fc1\.", r"\1.net.0."),
        (r"^(projector|pred_proj)\.bn\.", r"\1.net.1."),
        (r"^(projector|pred_proj)\.fc2\.", r"\1.net.3."),
        (r"^action_encoder\.mix\.bias$", "action_encoder.patch_embed.bias"),
        (r"^action_encoder\.fc1\.", "action_encoder.embed.0."),
        (r"^action_encoder\.fc2\.", "action_encoder.embed.2."),
        (r"^predictor\.pos$", "predictor.pos_embedding"),
        (r"^predictor\.norm\.", "predictor.transformer.norm."),
        (r"^predictor\.blocks\.(\d+)\.ada\.", r"predictor.transformer.layers.\1.adaLN_modulation.1."),
        (r"^predictor\.blocks\.(\d+)\.attn_norm\.", r"predictor.transformer.layers.\1.attn.norm."),
        (r"^predictor\.blocks\.(\d+)\.qkv\.", r"predictor.transformer.layers.\1.attn.to_qkv."),
        (r"^predictor\.blocks\.(\d+)\.attn_out\.", r"predictor.transformer.layers.\1.attn.to_out.0."),
        (r"^predictor\.blocks\.(\d+)\.mlp_norm\.", r"predictor.transformer.layers.\1.mlp.net.0."),
        (r"^predictor\.blocks\.(\d+)\.fc1\.", r"predictor.transformer.layers.\1.mlp.net.1."),
        (r"^predictor\.blocks\.(\d+)\.fc2\.", r"predictor.transformer.layers.\1.mlp.net.4."),
    ]
    for pat, rep in rules:
        if re.search(pat, k):
            return re.sub(pat, rep, k)
    raise KeyError(f"no official name for {k}")
