"""Kaggle T4 kernel 2: train LeWM from scratch on TwoRoom, two seeds at once (one per T4), then
evaluate each trained model with the official 50-episode planning protocol.

Kaggle bills a T4 x2 session at twice wall-clock whether or not the second GPU is used, so the second
GPU trains a second seed: the reproduction gets a seed-to-seed spread at no extra quota.

Outputs in /kaggle/working: run_s<seed>/{last.pt, last_official_layout.pt, history.json},
train.json (curves, timings, planning results) and the log.
"""
import glob
import json
import os
import subprocess
import sys
import tarfile
import time

REPO_SHA = "REPLACE_WITH_SHA"
TRAIN_HOURS = 8.0          # per seed, both in parallel; leaves time for setup and evaluation in 12 h
SEEDS = (3072, 3073)
T0 = time.time()
OUT = "/kaggle/working"
W = "/tmp/lewm"
results = {"repo_sha": REPO_SHA, "train_hours_budget": TRAIN_HOURS, "seeds": list(SEEDS)}


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


WORKER = r'''
import json, sys
sys.path.insert(0, "/tmp/lewm-t4")
from lewm_t4.train import TrainConfig, train
seed, h5, hours, out = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
res = train(TrainConfig(dataset_path=h5, out_dir=out, seed=seed, budget_hours=hours, workers=2), log=lambda m: print(f"[s{seed}] {m}", flush=True))
json.dump({k: v for k, v in res.items()}, open(out + "/result.json", "w"), indent=2, default=str)
'''

if __name__ == "__main__":
    import torch

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
    import zstandard

    with open(f"{W}/tworoom.tar.zst", "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(f"{W}/data", filter="data")
    os.remove(f"{W}/tworoom.tar.zst")
    H5 = sorted(glob.glob(f"{W}/data/**/*.h5", recursive=True))[0]
    with open("/tmp/worker.py", "w") as f:
        f.write(WORKER)
    procs = []
    for gpu, seed in enumerate(SEEDS[: len(names)]):
        out = f"{OUT}/run_s{seed}"
        os.makedirs(out, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        log = open(f"{OUT}/train_s{seed}.log", "w")
        procs.append((seed, subprocess.Popen([sys.executable, "/tmp/worker.py", str(seed), H5, str(TRAIN_HOURS), out],
                                             env=env, stdout=log, stderr=subprocess.STDOUT), log))
        stamp(f"training seed {seed} on GPU {gpu}")
    for seed, p, log in procs:
        p.wait()
        log.close()
        stamp(f"seed {seed} exited with {p.returncode}")
        with open(f"{OUT}/train_s{seed}.log") as f:
            tail = f.read()[-3000:]
        print(tail, flush=True)
        results[f"exit_s{seed}"] = p.returncode
    save()

    from lewm_t4.adapter import swm_cost_model
    from lewm_t4.evaluate import evaluate_tworoom
    from lewm_t4.model import LeWM

    for seed in SEEDS[: len(names)]:
        run = f"{OUT}/run_s{seed}"
        if not os.path.exists(f"{run}/last.pt"):
            continue
        hist = json.load(open(f"{run}/history.json"))
        results[f"history_s{seed}"] = hist["history"]
        m = LeWM()
        m.load_state_dict(torch.load(f"{run}/last.pt", map_location="cpu", weights_only=True))
        stamp(f"planning evaluation, seed {seed}")
        results[f"plan_s{seed}"] = evaluate_tworoom(swm_cost_model(m), H5, device="cuda")
        save()
        stamp(f"seed {seed}: {json.dumps({k: v for k, v in results[f'plan_s{seed}'].items() if k != 'protocol'})[:300]}")
    results["total_hours"] = (time.time() - T0) / 3600
    save()
    stamp("done")
