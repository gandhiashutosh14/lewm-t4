"""The from-scratch initialisation matches the reference model's, built from its published parts with the same seed."""
import math

import pytest
import torch

from lewm_t4.convert import to_official
from lewm_t4.model import LeWM, init_like_reference


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("stable_worldmodel")
    pytest.importorskip("transformers")
    from lewm_t4.reference import build_reference
    torch.manual_seed(0)
    ref = build_reference()
    torch.manual_seed(0)
    return ref, init_like_reference(LeWM())


@torch.no_grad()
def test_every_parameter_has_the_reference_spread(models):
    ref, mine = models
    theirs = dict(ref.named_parameters())
    ours = to_official(dict(mine.named_parameters()))
    assert set(ours) == set(theirs)
    for k, r in theirs.items():
        m = ours[k]
        if r.std() == 0:                     # constants: LayerNorm and BatchNorm affines, zero biases, AdaLN gates
            assert torch.equal(m, r), k
            continue
        # independent draws: the log-ratio of two sample standard deviations has a standard error below 1/sqrt(n)
        ratio = float(m.std() / r.std())
        assert ratio > 0 and abs(math.log(ratio)) < 5 / math.sqrt(r.numel() - 1), (k, ratio)


@torch.no_grad()
def test_action_embedding_has_the_reference_scale_at_init(models):
    ref, mine = models
    a = torch.randn(256, 4, 10, generator=torch.Generator().manual_seed(1))
    ratio = float(mine.action_encoder(a).std() / ref.action_encoder(a).std())
    assert 1 / 1.5 < ratio < 1.5, ratio
