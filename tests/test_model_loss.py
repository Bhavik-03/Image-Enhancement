import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

from losses import Loss
from model import AUTHORS_SHA256, UWnet, load_pretrained

PRETRAINED = Path("data/downloads/shallow_uwnet.ckpt")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def tiny_features():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())


def test_output_has_input_shape_for_any_size():
    model = UWnet().eval()
    x = torch.rand(2, 3, 33, 47)
    assert model(x).shape == x.shape


def test_parameter_count_is_the_paper_size():
    assert sum(p.numel() for p in UWnet().parameters()) == 219_456  # ~220K


def test_load_pretrained_roundtrips_a_pickled_model(tmp_path):
    model = UWnet().eval()
    torch.save(model, tmp_path / "m.ckpt")  # same format as the authors' file
    loaded = load_pretrained(tmp_path / "m.ckpt").eval()
    x = torch.rand(1, 3, 32, 32)
    assert torch.equal(model(x), loaded(x))


def test_load_pretrained_rejects_foreign_classes(tmp_path):
    torch.save(nn.Linear(2, 2), tmp_path / "evil.ckpt")
    with pytest.raises(pickle.UnpicklingError):
        load_pretrained(tmp_path / "evil.ckpt")


@pytest.mark.skipif(
    not PRETRAINED.exists(), reason="authors' checkpoint not downloaded"
)
def test_authors_checkpoint_loads():
    model = load_pretrained(PRETRAINED, sha256=AUTHORS_SHA256).eval()
    assert sum(p.numel() for p in model.parameters()) == 219_456
    assert model(torch.rand(1, 3, 40, 40)).shape == (1, 3, 40, 40)


def test_loss_is_zero_for_identical_images_and_finite_otherwise():
    loss = Loss(tiny_features())
    a, b = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    assert loss(a, a) == 0
    assert torch.isfinite(loss(a, b)) and loss(a, b) > 0


def test_loss_backprops_to_prediction_but_not_to_features():
    loss = Loss(tiny_features())
    pred = torch.rand(1, 3, 16, 16, requires_grad=True)
    loss(pred, torch.rand(1, 3, 16, 16)).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert all(p.grad is None for p in loss.features.parameters())


def test_features_stay_in_eval_mode_when_loss_is_set_to_train():
    loss = Loss(nn.Sequential(nn.Conv2d(3, 4, 3), nn.Dropout(0.5)))
    loss.train()
    assert not loss.features.training


def test_model_can_overfit_one_batch():
    model, loss = UWnet(), Loss(tiny_features())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    raw, ref = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    first = loss(model(raw), ref).item()
    for _ in range(60):
        opt.zero_grad()
        loss(model(raw), ref).backward()
        opt.step()
    assert loss(model(raw), ref).item() < 0.6 * first
