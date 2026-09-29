import os
import urllib.request
from pathlib import Path

import pytest
import torch

from lewm_t4.convert import from_official, to_official
from lewm_t4.model import LeWM, LeWMConfig, count_params, init_like_reference, sigreg

WEIGHTS_URL = "https://huggingface.co/quentinll/lewm-tworooms/resolve/main/weights.pt"


def test_parameter_count_matches_the_official_checkpoint():
    assert count_params(LeWM()) == 18_034_478


def test_loss_and_rollout_shapes_on_cpu():
    torch.manual_seed(0)
    m = init_like_reference(LeWM(LeWMConfig(image_size=112)))
    px = torch.randn(2, 4, 3, 112, 112)
    act = torch.randn(2, 4, 10)
    loss, parts = m.loss(px, act)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"pred", "sigreg"}
    m.eval()
    emb0 = m.encode(px[:, :1])
    out = m.rollout(emb0, torch.randn(2, 5, 10))
    assert out.shape == (2, 6, 192)
    cost = m.plan_cost(emb0, emb0[:, 0], torch.randn(2, 5, 10))
    assert cost.shape == (2,) and (cost >= 0).all()


def test_sigreg_is_near_its_null_value_on_gaussians_and_large_on_collapse():
    g = torch.Generator().manual_seed(0)
    iso = torch.randn(4, 512, 64, generator=g)
    s_iso = float(sigreg(iso, slices=256, generator=torch.Generator().manual_seed(1)))
    s_col = float(sigreg(torch.zeros(4, 512, 64) + 0.5, slices=256, generator=torch.Generator().manual_seed(1)))
    assert 0.3 < s_iso < 2.0 and s_col > 50 * s_iso


def test_sigreg_runs_in_fp32_under_autocast():
    z = torch.randn(4, 256, 64, generator=torch.Generator().manual_seed(0)).bfloat16()
    plain = sigreg(z, slices=128, generator=torch.Generator().manual_seed(1))
    with torch.autocast("cpu", dtype=torch.bfloat16):              # matmuls would otherwise run in bf16
        mixed = sigreg(z, slices=128, generator=torch.Generator().manual_seed(1))
    assert torch.equal(plain, mixed)


def test_converter_round_trips_a_state_dict():
    sd = LeWM().state_dict()
    back = from_official(to_official(sd))
    assert set(back) == set(sd)
    assert all(torch.equal(back[k], sd[k]) for k in sd)


def test_predictor_is_causal():
    m = LeWM().eval()
    for blk in m.predictor.blocks:                   # non-zero gates so attention matters
        torch.nn.init.normal_(blk.ada.weight, std=0.05)
    emb, act = torch.randn(1, 3, 192), torch.randn(1, 3, 192)
    a = m.predict(emb, act)
    emb2 = emb.clone()
    emb2[:, 2] += 1.0
    b = m.predict(emb2, act)
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6) and not torch.allclose(a[:, 2], b[:, 2])


def _weights() -> Path:
    p = Path(os.environ.get("LEWM_WEIGHTS", Path.home() / ".cache" / "lewm-t4" / "tworooms-weights.pt"))
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, p)
    return p


@pytest.fixture(scope="module")
def pair():
    pytest.importorskip("stable_worldmodel")
    transformers = pytest.importorskip("transformers")
    if int(transformers.__version__.split(".")[0]) >= 5:
        pytest.skip("the official checkpoint uses transformers 4 ViT parameter names")
    from lewm_t4.reference import build_reference
    sd = torch.load(_weights(), map_location="cpu", weights_only=True)
    ref = build_reference()
    ref.load_state_dict(sd, strict=True)
    mine = LeWM()
    mine.load_state_dict(from_official(sd), strict=True)
    return ref.eval(), mine.eval()


@torch.no_grad()
def test_parity_with_the_official_model(pair):
    ref, mine = pair
    torch.manual_seed(0)
    px = torch.rand(2, 3, 3, 224, 224) * 4 - 2
    act = torch.randn(2, 3, 10)
    r = ref.encode({"pixels": px.clone(), "action": act.clone()})
    assert torch.allclose(r["emb"], mine.encode(px), atol=1e-5)
    assert torch.allclose(r["act_emb"], mine.action_encoder(act), atol=1e-5)
    assert torch.allclose(ref.predict(r["emb"], r["act_emb"]), mine.predict(mine.encode(px), mine.action_encoder(act)), atol=1e-5)


@torch.no_grad()
def test_planning_cost_matches_the_official_model(pair):
    ref, mine = pair
    torch.manual_seed(1)
    S = 16
    px = torch.rand(1, 2, 3, 224, 224) * 4 - 2
    cand = torch.randn(1, S, 5, 10)

    def fresh():                                     # get_cost caches embeddings in the dict it is given
        return {"pixels": px[:, None, :1].expand(1, S, 1, 3, 224, 224).clone(),
                "goal": px[:, None, 1:2].expand(1, S, 1, 3, 224, 224).clone(), "action": cand[:, :, :1].clone()}

    c_ref = ref.get_cost(fresh(), cand.clone()).flatten()
    emb0 = mine.encode(px[:, :1]).expand(S, 1, -1)
    c_me = mine.plan_cost(emb0, mine.encode(px[:, 1:2])[:, 0].expand(S, -1), cand[0])
    assert torch.allclose(c_ref, c_me, rtol=1e-5)
    from lewm_t4.adapter import swm_cost_model
    info = fresh()
    c_adapter = swm_cost_model(mine).get_cost(info, cand.clone()).flatten()
    assert torch.allclose(c_ref, c_adapter, rtol=1e-5)
    assert torch.equal(info["goal_emb"], mine.encode(px[:, 1:2]))   # the adapter encoded the goal itself
