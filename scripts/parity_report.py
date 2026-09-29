"""Measure, rather than just assert, the agreement between this code and the official LeWM.

Loads the official TwoRoom checkpoint into both implementations, feeds them the same seeded inputs
(the ones tests/test_model.py uses) and records the largest differences. Writes
reports/parity-cpu.json and prints it; CI runs this on every push so the numbers are in every log.

    python scripts/parity_report.py [--weights PATH] [--out reports/parity-cpu.json]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from lewm_t4.convert import from_official  # noqa: E402
from lewm_t4.model import LeWM  # noqa: E402
from lewm_t4.reference import build_reference  # noqa: E402

URL = "https://huggingface.co/quentinll/lewm-tworooms/resolve/main/weights.pt"


@torch.no_grad()
def measure(weights: Path) -> dict:
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    ref = build_reference()
    ref.load_state_dict(sd, strict=True)
    ref.eval()
    mine = LeWM()
    mine.load_state_dict(from_official(sd), strict=True)
    mine.eval()
    torch.manual_seed(0)
    px = torch.rand(2, 3, 3, 224, 224) * 4 - 2
    act = torch.randn(2, 3, 10)
    r = ref.encode({"pixels": px.clone(), "action": act.clone()})
    e_me, a_me = mine.encode(px), mine.action_encoder(act)
    p_ref, p_me = ref.predict(r["emb"], r["act_emb"]), mine.predict(e_me, a_me)
    torch.manual_seed(1)
    S = 16
    px2 = torch.rand(1, 2, 3, 224, 224) * 4 - 2
    cand = torch.randn(1, S, 5, 10)
    info = {"pixels": px2[:, None, :1].expand(1, S, 1, 3, 224, 224).clone(),
            "goal": px2[:, None, 1:2].expand(1, S, 1, 3, 224, 224).clone(), "action": cand[:, :, :1].clone()}
    c_ref = ref.get_cost(info, cand.clone()).flatten()
    c_me = mine.plan_cost(mine.encode(px2[:, :1]).expand(S, 1, -1), mine.encode(px2[:, 1:2])[:, 0].expand(S, -1), cand[0])
    import transformers
    return {
        "encoder_max_abs_diff": float((r["emb"] - e_me).abs().max()),
        "encoder_bit_exact": bool(torch.equal(r["emb"], e_me)),
        "action_encoder_max_abs_diff": float((r["act_emb"] - a_me).abs().max()),
        "predictor_max_abs_diff": float((p_ref - p_me).abs().max()),
        "predictor_output_mean_abs": float(p_ref.abs().mean()),
        "plan_cost_max_rel_diff": float(((c_ref - c_me).abs() / c_ref.abs()).max()),
        "plan_cost_mean": float(c_ref.mean()),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "machine": platform.machine(), "processor": platform.processor() or "unknown",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("LEWM_WEIGHTS", str(Path.home() / ".cache" / "lewm-t4" / "tworooms-weights.pt")))
    ap.add_argument("--out", default=str(ROOT / "reports" / "parity-cpu.json"))
    a = ap.parse_args()
    w = Path(a.weights)
    if not w.exists():
        w.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, w)
    res = measure(w)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
