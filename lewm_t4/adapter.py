"""Run this project's LeWM inside the official planning code path.

stable-worldmodel's planners call ``model.get_cost(info_dict, action_candidates)``. The reference
LeWM implements that on top of ``encode``, ``predict`` and ``action_encoder``. Subclassing it and
swapping in this project's modules means the rollout, the goal cost and the CEM solver are the
official code, and only the model differs: any difference in planning success comes from the model.
"""
from __future__ import annotations

from torch import nn

from .model import LeWM


def swm_cost_model(model: LeWM) -> nn.Module:
    from stable_worldmodel.wm.lewm import LeWM as RefLeWM

    class Adapter(RefLeWM):
        def __init__(self, m: LeWM):
            nn.Module.__init__(self)
            self.m = m
            self.action_encoder = m.action_encoder
            self.predictor = m.predictor
            self.predictor.num_frames = m.cfg.history      # read by the reference rollout

        def encode(self, info):
            info["emb"] = self.m.encode(info["pixels"].to(next(self.m.parameters()).dtype))
            if "action" in info:
                info["act_emb"] = self.m.action_encoder(info["action"])
            return info

        def predict(self, emb, act_emb):
            return self.m.predict(emb, act_emb)

    return Adapter(model)
