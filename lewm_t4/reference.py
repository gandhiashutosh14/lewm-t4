"""Build the official LeWM from its published parts, for parity checks only.

The encoder is a Hugging Face ViTModel configured exactly as stable-pretraining's ``vit_hf("tiny",
patch_size=14, image_size=224, use_mask_token=False)`` builds it; the predictor, action embedder and
projectors are the MIT-licensed modules shipped in stable-worldmodel (``stable_worldmodel.wm.lewm``).
Requires ``pip install transformers stable-worldmodel`` (the ``reference`` extra).
"""
from __future__ import annotations

import torch
from torch import nn


def build_reference(action_dim: int = 10) -> nn.Module:
    from stable_worldmodel.wm.lewm import LeWM as RefLeWM
    from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor
    from transformers import ViTConfig, ViTModel

    vit = ViTModel(ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768,
                             image_size=224, patch_size=14), add_pooling_layer=False, use_mask_token=False)
    predictor = Predictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16,
                            mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)
    return RefLeWM(encoder=vit, predictor=predictor, action_encoder=Embedder(input_dim=action_dim, emb_dim=192),
                   projector=MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d),
                   pred_proj=MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d))


def load_official(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)
