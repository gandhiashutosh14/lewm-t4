"""Kaggle T4 kernel 1: verify the reimplementation where it matters, in the official planning loop.

1. GPU check (T4 or fail in seconds).
2. Install the reference stack, clone this repository at a pinned commit.
3. Download the official TwoRoom checkpoint and dataset from Hugging Face; record the dataset layout.
4. Parity on the GPU (fp32) between the official model and this reimplementation.
5. The official 50-episode TwoRoom planning evaluation, run twice through the same CEM solver and
   environment: once with the official model, once with this reimplementation carrying the same weights.
6. Training throughput of the reimplementation (fp16 autocast), to size the from-scratch run.

Everything is written to /kaggle/working/verify.json and the log.
"""
import glob
import json
import os
import subprocess
import sys
import tarfile
import time

REPO_SHA = "REPLACE_WITH_SHA"
T0 = time.time()
OUT = "/kaggle/working"
W = "/tmp/lewm"
results = {"repo_sha": REPO_SHA}


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")


def stamp(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def save():
    with open(f"{OUT}/verify.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


import torch  # noqa: E402

assert torch.cuda.is_available(), "No CUDA: enable_gpu was not set"
names = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
print("GPUs:", names, flush=True)
assert all("T4" in n for n in names), f"Wrong GPU {names}: set machine_shape=NvidiaTeslaT4"
results["gpus"] = names
results["torch"] = torch.__version__

sh('pip install -q "stable-worldmodel==0.1.1" "transformers>=4.40,<5" h5py hdf5plugin opencv-python-headless '
   '"imageio[ffmpeg]" pygame pymunk shapely zstandard einops scikit-learn')
sh(f"git clone -q https://github.com/gandhiashutosh14/lewm-t4 /tmp/lewm-t4 && git -C /tmp/lewm-t4 checkout -q {REPO_SHA}")
sys.path.insert(0, "/tmp/lewm-t4")
os.makedirs(W, exist_ok=True)
stamp("downloading checkpoint and dataset")
sh(f"wget -q -O {W}/weights.pt https://huggingface.co/quentinll/lewm-tworooms/resolve/main/weights.pt")
sh(f"wget -q -O {W}/tworoom.tar.zst https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst")
import zstandard  # noqa: E402

with open(f"{W}/tworoom.tar.zst", "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        tar.extractall(f"{W}/data")
os.remove(f"{W}/tworoom.tar.zst")
h5_files = sorted(glob.glob(f"{W}/data/**/*.h5", recursive=True))
stamp(f"dataset files: {h5_files}")
H5 = h5_files[0]

import h5py  # noqa: E402
import numpy as np  # noqa: E402

with h5py.File(H5, "r") as f:
    layout = {k: {"shape": list(f[k].shape), "dtype": str(f[k].dtype)} for k in f.keys()}
    ep_len = f["ep_len"][:]
    results["dataset"] = {"file": os.path.basename(H5), "bytes": os.path.getsize(H5), "layout": layout,
                          "episodes": int(len(ep_len)), "frames": int(ep_len.sum()),
                          "episode_length": {"min": int(ep_len.min()), "median": float(np.median(ep_len)), "max": int(ep_len.max())}}
    px = f["pixels"][:4]
    from PIL import Image
    Image.fromarray(np.concatenate(list(px), axis=1)).save(f"{OUT}/sample_frames.png")
save()
stamp(f"dataset: {results['dataset']['episodes']} episodes, {results['dataset']['frames']} frames")

from lewm_t4.adapter import swm_cost_model  # noqa: E402
from lewm_t4.convert import from_official  # noqa: E402
from lewm_t4.evaluate import evaluate_tworoom  # noqa: E402
from lewm_t4.model import LeWM  # noqa: E402
from lewm_t4.reference import build_reference  # noqa: E402

sd = torch.load(f"{W}/weights.pt", map_location="cpu", weights_only=True)
ref = build_reference()
ref.load_state_dict(sd, strict=True)
ref = ref.cuda().eval()
mine = LeWM()
mine.load_state_dict(from_official(sd), strict=True)
mine = mine.cuda().eval()
with torch.no_grad():
    torch.manual_seed(0)
    pix = (torch.rand(4, 3, 3, 224, 224, device="cuda") * 4 - 2)
    act = torch.randn(4, 3, 10, device="cuda")
    r = ref.encode({"pixels": pix.clone(), "action": act.clone()})
    e_me = mine.encode(pix)
    p_ref = ref.predict(r["emb"], r["act_emb"])
    p_me = mine.predict(e_me, mine.action_encoder(act))
    results["parity_gpu_fp32"] = {"encode_max_abs": float((r["emb"] - e_me).abs().max()),
                                  "predict_max_abs": float((p_ref - p_me).abs().max()),
                                  "embedding_mean_abs": float(r["emb"].abs().mean())}
save()
stamp(f"parity: {results['parity_gpu_fp32']}")

for name, model in (("official", ref), ("reimplementation", swm_cost_model(mine))):
    stamp(f"planning evaluation: {name}")
    res = evaluate_tworoom(model, H5, device="cuda")
    results[f"plan_{name}"] = res
    save()
    stamp(f"{name}: {json.dumps({k: v for k, v in res.items() if k != 'protocol'})[:400]}")

from lewm_t4.train import TrainConfig, train  # noqa: E402

del ref, mine
torch.cuda.empty_cache()
stamp("training throughput benchmark")
bench = train(TrainConfig(dataset_path=H5, out_dir="/tmp/bench", max_steps=200, max_epochs=1, workers=4), log=print)
steps_hours = bench["hours"]
results["train_benchmark"] = {"steps": bench["steps"], "hours": steps_hours, "sec_per_step": steps_hours * 3600 / bench["steps"],
                              "history": bench["history"], "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
save()
stamp(f"benchmark: {results['train_benchmark']['sec_per_step']:.3f} s/step")
results["total_minutes"] = (time.time() - T0) / 60
save()
stamp("done")
