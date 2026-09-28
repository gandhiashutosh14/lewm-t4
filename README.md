# lewm-t4

**LeWorldModel, rebuilt from the paper in plain PyTorch, proven equal to the official model on its own weights, and retrained on a free Kaggle T4.**

[![parity](https://github.com/gandhiashutosh14/lewm-t4/actions/workflows/ci.yml/badge.svg)](https://github.com/gandhiashutosh14/lewm-t4/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Kaggle%20T4%20(free)-blue)
![Status](https://img.shields.io/badge/status-v0.1%20in%20progress-yellow)

> **In plain English:** LeWorldModel (Maes, Le Lidec, Scieur, LeCun and Balestriero, 2026) is a
> JEPA world model: it learns to predict the *internal representation* of the next camera frame,
> not the pixels, and plans by searching for actions whose imagined future lands on a goal. It
> trains with only two loss terms, thanks to SIGReg from LeJEPA. This repository rebuilds it from
> the paper as one readable PyTorch file, then proves the rebuild is exact: loaded with the
> authors' released weights it produces the same numbers as their model, checked by CI on every
> push. The next steps run it inside the authors' own planning evaluation and retrain it from
> scratch on the free GPUs Kaggle gives every account.

**Reading guide:** [status](#status) is the evidence so far; [how it works](#how-it-works) is the
model in one screen; [verify it yourself](#verify-it-yourself) takes two commands.

## Status

| Step | Evidence | State |
|---|---|---|
| Architecture from the paper | [`lewm_t4/model.py`](lewm_t4/model.py): ViT-tiny encoder, BatchNorm projectors, action embedder, 6-layer AdaLN-zero causal predictor, SIGReg; 18,034,478 parameters, the official count | done |
| Load the official TwoRoom checkpoint | [`lewm_t4/convert.py`](lewm_t4/convert.py) maps it onto this code; mapping back is exact | done |
| Numerical parity on CPU | same inputs through both models: encoder **bit-exact** (max difference 0.0), action encoder 6e-7, predictor 8.6e-7, planning cost 1.2e-7 relative; [CI](.github/workflows/ci.yml) re-checks it on every push against the checkpoint downloaded fresh | done |
| Parity inside the official planning loop, on a T4 | [`kaggle/verify`](kaggle/verify): the authors' 50-episode TwoRoom protocol, their CEM solver, run once with their model and once with this one | running |
| Retrain from scratch on a T4 (two seeds, fp16) and compare planning success | [`kaggle/train`](kaggle/train) | next |

## How it works

```
 frame t-2, t-1, t (224 x 224)      actions (5 frames x 2-D, z-scored)
        |                                   |
  ViT-tiny, patch 14 -> CLS (192)     Linear mix -> MLP (SiLU) -> 192
        |                                   |
  Linear -> BatchNorm -> GELU -> Linear     |
        |                                   |
     z_{t-2}, z_{t-1}, z_t  ---->  6 causal transformer blocks, each sub-layer shifted, scaled and
                                   gated by the action embedding (AdaLN-zero)  ->  BatchNorm MLP
                                                  |
                                   z_hat_{t+1}  (compared with the encoding of frame t+1)

 loss = MSE(z_hat, z) + 0.09 * SIGReg(z)          no EMA teacher, no stop-gradient, no decoder
 plan = CEM over 5 steps x 5 frames of actions, cost = ||z_hat_final - z_goal||^2
```

SIGReg draws 1,024 random directions, projects each time step's batch of embeddings onto them,
and measures (Epps-Pulley) how far each 1-D projection is from a standard normal. Pushing every
projection toward N(0, 1) pushes the whole embedding toward an isotropic Gaussian, which rules out
the collapsed solution where every frame maps to the same point.

## Verify it yourself

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[reference,dev]"
pytest -q            # 7 tests; downloads the official 72 MB checkpoint once
```

## A note for anyone loading the official checkpoint

The released weights use the parameter names of Hugging Face `transformers` 4.x
(`encoder.layer.N.attention.attention.query`). `transformers` 5 renamed the ViT modules
(`encoder.layers.N.attention.q_proj`), so the checkpoint no longer loads into a current `ViTModel`.
Pin `transformers<5`, or use `lewm_t4.convert.from_official`, which reads the checkpoint's own names.

## What this does not claim

- It is a reimplementation and reproduction, not a new method. Credit for LeWorldModel belongs to
  its authors; see [`NOTICE.md`](NOTICE.md).
- Parity is shown for the TwoRoom checkpoint. The other environments use the same architecture,
  but only TwoRoom was checked.
- Training here uses fp16 on a T4 rather than the authors' bf16; results from scratch are reported
  as they come, with the number of epochs that fit a free session.

## License

MIT. See [`NOTICE.md`](NOTICE.md) for the paper, code and data this reimplements or downloads.
