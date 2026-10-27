import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import PairedDataset
from evaluate import score
from model import UWnet, load_finetuned


def make_root(tmp_path, raw_value, ref_value, n=10):
    for d, value in (("raw", raw_value), ("ref", ref_value)):
        (tmp_path / d).mkdir()
        for i in range(n):
            cv2.imwrite(
                str(tmp_path / d / f"{i}.png"), np.full((32, 40, 3), value, np.uint8)
            )
    return tmp_path


class Fixed(nn.Module):
    """Model that ignores its input and returns fn(x)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def loader(root):
    return DataLoader(PairedDataset(root, "val"), batch_size=1)


def test_psnr_matches_the_formula_for_a_known_error(tmp_path):
    # every pixel is off by 50/255, so MSE = (50/255)^2 and PSNR = -20 log10(50/255)
    psnr, ssim = score(nn.Identity(), loader(make_root(tmp_path, 100, 150)))
    assert psnr == pytest.approx(-20 * np.log10(50 / 255), abs=1e-3)
    assert 0 < ssim < 1


def test_a_model_that_returns_the_reference_beats_the_raw_input(tmp_path):
    root = make_root(tmp_path, 100, 150)
    ref_value = torch.full((1, 3, 32, 40), 150 / 255)
    raw_psnr, _ = score(nn.Identity(), loader(root))
    good_psnr, good_ssim = score(Fixed(lambda x: ref_value), loader(root))
    assert good_psnr > raw_psnr and good_ssim == pytest.approx(1.0)


def test_output_is_clamped_before_scoring(tmp_path):
    # a model that overshoots to 5.0 must score the same as one that returns 1.0
    root = make_root(tmp_path, 100, 255)
    assert (
        score(Fixed(lambda x: torch.full_like(x, 5.0)), loader(root))[0]
        == score(Fixed(torch.ones_like), loader(root))[0]
    )


def test_side_by_side_images_are_saved(tmp_path):
    (tmp_path / "d").mkdir()
    root = make_root(tmp_path / "d", 100, 150)
    score(nn.Identity(), loader(root), save_dir=tmp_path / "out", n_save=1)
    strip = cv2.imread(str(tmp_path / "out" / "000.png"))
    assert strip.shape == (32, 3 * 40, 3)  # raw | enhanced | reference


def test_load_finetuned_roundtrips_a_state_dict(tmp_path):
    model = UWnet().eval()
    torch.save(model.state_dict(), tmp_path / "ft.pth")
    x = torch.rand(1, 3, 32, 32)
    assert torch.equal(model(x), load_finetuned(tmp_path / "ft.pth").eval()(x))
