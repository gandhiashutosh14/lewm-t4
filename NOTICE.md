# Notices

`lewm-t4` is MIT-licensed (see `LICENSE`). It reimplements a published model; this file records
what it depends on and what it does not contain.

| Item | Source | Relationship |
|---|---|---|
| The LeWorldModel method and architecture | Maes, Le Lidec, Scieur, LeCun, Balestriero, *LeWorldModel*, arXiv 2603.19312; code github.com/lucas-maes/le-wm (MIT, Copyright (c) 2026 Lucas Maes) | Reimplemented in `lewm_t4/model.py` with this project's own module structure. Hyperparameters, the loss and the evaluation protocol follow the paper's released configuration, which is cited where it is used. |
| SIGReg | Balestriero and LeCun, *LeJEPA*, arXiv 2511.08544 | Implemented from the equations, with LeWM's numerical choices. The LeJEPA reference repository (CC BY-NC) was not used. |
| Reference model parts used only for parity checks and the planning harness | stable-worldmodel 0.1.1 (MIT) and Hugging Face transformers (Apache 2.0) | An optional dependency (`pip install -e .[reference]`). No code is copied into this repository. |
| Official checkpoint `quentinll/lewm-tworooms` and the TwoRoom dataset | Hugging Face Hub, published by the LeWorldModel authors | Downloaded at test or run time, never redistributed here. |
