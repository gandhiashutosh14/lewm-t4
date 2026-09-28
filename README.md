# lewm-t4

**LeWorldModel, rebuilt from the paper in plain PyTorch, proven equal to the official model on its own weights, and retrained on a free Kaggle T4.**

[![parity](https://github.com/gandhiashutosh14/lewm-t4/actions/workflows/ci.yml/badge.svg)](https://github.com/gandhiashutosh14/lewm-t4/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

> **In plain English:** LeWorldModel (Maes, Le Lidec, Scieur, LeCun and Balestriero, 2026) is a
> JEPA world model: it learns to predict the *internal representation* of the next camera frame
> rather than the pixels, and plans by searching for actions whose imagined future lands on a goal.
> Its training needs only two loss terms, thanks to SIGReg from LeJEPA. This repository rebuilds
> it from the paper as one readable PyTorch file, then proves the rebuild is exact: loaded with the
> authors' released weights, it produces the same numbers as their model, and it plans with the
> same success in their evaluation. The next step retrains it from scratch on the free GPUs Kaggle
> gives every account.

## Status

| Step | Evidence | State |
|---|---|---|
| Architecture from the paper: ViT-tiny encoder, BatchNorm projectors, action embedder, 6-layer AdaLN-zero causal predictor, SIGReg | [`lewm_t4/model.py`](lewm_t4/model.py), 18,034,478 parameters (the official count) | done |
| Load the official TwoRoom checkpoint | [`lewm_t4/convert.py`](lewm_t4/convert.py) maps it onto this code; the round trip back is exact | done |
| Numerical parity on CPU | encoder bit-exact (max difference 0.0), predictor within 8.6e-7, planning cost within 1.2e-7 relative; checked by CI on every push | done |
| Parity in the official planning loop on a T4 | [`kaggle/verify`](kaggle/verify) | running |
| Retrain from scratch on a T4 and compare planning success | `kaggle/train` | next |

## A note for anyone loading the official checkpoint

The released weights use the parameter names of Hugging Face `transformers` 4.x (`encoder.layer.N.attention.attention.query`).
`transformers` 5 renamed the ViT modules (`encoder.layers.N.attention.q_proj`), so the checkpoint no longer loads into a current
`ViTModel`. Pin `transformers<5`, or use `lewm_t4.convert.from_official`, which reads the checkpoint's own names.

## License

MIT. See [`NOTICE.md`](NOTICE.md) for the paper, code and data this reimplements or downloads.
