"""Goal-reaching evaluation on TwoRoom, following LeWorldModel's protocol (eval.py and
config/eval/tworoom.yaml in github.com/lucas-maes/le-wm, MIT), without Hydra.

Protocol: 50 start states sampled from the dataset (seed 42); the goal is the state 25 steps later
in the same episode; the agent has 50 environment steps. Planning: CEM with 300 samples, 30
iterations, top 30, horizon 5 planning steps of 5 frames each (action block 5), re-planning every 5.
Actions and proprioception are z-scored with dataset statistics; images get ImageNet normalisation
at 224 x 224. Success is the environment's own goal test.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

PROTOCOL = {"num_eval": 50, "goal_offset_steps": 25, "eval_budget": 50, "img_size": 224, "seed": 42,
            "horizon": 5, "receding_horizon": 5, "action_block": 5,
            "cem": {"num_samples": 300, "var_scale": 1.0, "n_steps": 30, "topk": 30}}


def image_transform(size: int = 224):
    from torchvision.transforms import v2 as T
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), T.Resize(size=size)])


def evaluate_tworoom(cost_model: torch.nn.Module, dataset_path: str, device: str = "cuda", protocol: Optional[Dict] = None,
                     video_dir: Optional[str] = None) -> Dict:
    import stable_worldmodel as swm
    from sklearn import preprocessing
    p = {**PROTOCOL, **(protocol or {})}
    world = swm.World(env_name="swm/TwoRoom-v1", num_envs=p["num_eval"], max_episode_steps=2 * p["eval_budget"],
                      image_shape=(224, 224))
    dataset = swm.data.HDF5Dataset(path=dataset_path, keys_to_cache=["action", "proprio"])
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    process = {}
    for c in ("action", "proprio"):
        scaler = preprocessing.StandardScaler()
        data = dataset.get_col_data(c)
        scaler.fit(data[~np.isnan(data).any(axis=1)])
        process[c] = scaler
        if c != "action":
            process[f"goal_{c}"] = scaler
    cost_model = cost_model.to(device).eval().requires_grad_(False)
    solver = swm.solver.CEMSolver(model=cost_model, batch_size=1, device=device, seed=p["seed"], **p["cem"])
    policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(horizon=p["horizon"], receding_horizon=p["receding_horizon"],
                                                                              action_block=p["action_block"]),
                                         process=process, transform={"pixels": image_transform(p["img_size"]), "goal": image_transform(p["img_size"])})
    ep_ids = dataset.get_col_data(col)
    steps = dataset.get_col_data("step_idx")
    episodes = np.unique(ep_ids)
    lengths = np.array([steps[ep_ids == e].max() + 1 for e in episodes])
    max_start = dict(zip(episodes, lengths - p["goal_offset_steps"] - 1))
    valid = np.nonzero(steps <= np.array([max_start[e] for e in ep_ids]))[0]
    g = np.random.default_rng(p["seed"])
    rows = np.sort(valid[g.choice(len(valid) - 1, size=p["num_eval"], replace=False)])
    info = dataset.get_row_data(rows)
    world.set_policy(policy)
    t0 = time.time()
    metrics = world.evaluate(dataset=dataset, start_steps=info["step_idx"].tolist(), goal_offset=p["goal_offset_steps"],
                             eval_budget=p["eval_budget"], episodes_idx=info[col].tolist(),
                             callables=[{"method": "_set_state", "args": {"state": {"value": "proprio"}}},
                                        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}}],
                             video=Path(video_dir) if video_dir else None)
    out = {k: (float(v) if np.isscalar(v) else (np.asarray(v).tolist() if hasattr(v, "__len__") else v)) for k, v in metrics.items()}
    out["seconds"] = round(time.time() - t0, 1)
    out["protocol"] = p
    return out
