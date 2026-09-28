"""Kaggle T4 kernel 2: train LeWM from scratch on TwoRoom, then plan with it.

Two seeds train at once, one per T4, on the same stream of batches (the data is decoded once; Kaggle
bills a T4 x2 session at twice wall-clock whether or not the second GPU works). The schedule is
fitted to a 9-hour budget by steps, so training ends on schedule inside the 12-hour session. Each
trained model then runs the official 50-episode TwoRoom planning evaluation.

Outputs in /kaggle/working: s<seed>/{last.pt, last_official_layout.pt, history.json}, train.json.
"""
import glob
import json
import os
import subprocess
import sys
import tarfile
import time

REPO_SHA = "REPLACE_WITH_SHA"
TRAIN_HOURS = 9.0
SEEDS = [3072, 3073]
T0 = time.time()
OUT = "/kaggle/working"
W = "/tmp/lewm"
results = {"repo_sha": REPO_SHA, "train_hours_budget": TRAIN_HOURS, "seeds": SEEDS}


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")


def stamp(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def save():
    with open(f"{OUT}/train.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


import torch  # noqa: E402

assert torch.cuda.is_available(), "No CUDA: enable_gpu was not set"
names = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
print("GPUs:", names, flush=True)
assert all("T4" in n for n in names), f"Wrong GPU {names}: set machine_shape=NvidiaTeslaT4"
results["gpus"] = names
sh('pip install -q "stable-worldmodel==0.1.1" "transformers>=4.40,<5" h5py hdf5plugin opencv-python-headless '
   '"imageio[ffmpeg]" pygame pymunk shapely zstandard einops scikit-learn')
sh(f"git clone -q https://github.com/gandhiashutosh14/lewm-t4 /tmp/lewm-t4 && git -C /tmp/lewm-t4 checkout -q {REPO_SHA}")
sys.path.insert(0, "/tmp/lewm-t4")
os.makedirs(W, exist_ok=True)
stamp("downloading dataset")
sh(f"wget -q -O {W}/tworoom.tar.zst https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst")
import zstandard  # noqa: E402

with open(f"{W}/tworoom.tar.zst", "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        tar.extractall(f"{W}/data", filter="data")
os.remove(f"{W}/tworoom.tar.zst")
H5 = sorted(glob.glob(f"{W}/data/**/*.h5", recursive=True))[0]

from lewm_t4.train import TrainConfig, train  # noqa: E402

seeds = SEEDS[: len(names)]
stamp(f"training seeds {seeds} on {len(seeds)} GPU(s) for {TRAIN_HOURS} h")
res = train(TrainConfig(dataset_path=H5, out_dir=OUT, budget_hours=TRAIN_HOURS, workers=4), log=print,
            seeds=seeds, devices=[f"cuda:{i}" for i in range(len(seeds))])
results["training"] = {k: v for k, v in res.items() if k != "history"}
results["history"] = res["history"]
save()
stamp(f"training done: {res['steps']} steps, {res['steps'] / res['steps_per_epoch']:.2f} epochs, {res['hours']:.2f} h")

from lewm_t4.adapter import swm_cost_model  # noqa: E402
from lewm_t4.evaluate import evaluate_tworoom  # noqa: E402
from lewm_t4.model import LeWM  # noqa: E402

for seed in seeds:
    m = LeWM()
    m.load_state_dict(torch.load(f"{OUT}/s{seed}/last.pt", map_location="cpu", weights_only=True))
    stamp(f"planning evaluation, seed {seed}")
    results[f"plan_s{seed}"] = evaluate_tworoom(swm_cost_model(m), H5, device="cuda:0")
    save()
    stamp(f"seed {seed}: success {results[f'plan_s{seed}'].get('success_rate')}")
results["total_hours"] = (time.time() - T0) / 3600
save()
stamp("done")
