from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import PairedDataset
from evaluate import score
from losses import Loss
from model import UWnet, load_finetuned
from train import fit, seed_everything


def make_root(path, n=20, size=32):
    rng = np.random.default_rng(0)
    for d in ("raw", "ref"):
        (path / d).mkdir(parents=True)
    for i in range(n):
        img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(path / "raw" / f"{i}.png"), img)
        cv2.imwrite(str(path / "ref" / f"{i}.png"), 255 - img)  # a learnable mapping
    return path


@pytest.fixture
def loaders(tmp_path, monkeypatch):
    root = make_root(tmp_path / "d")
    monkeypatch.chdir(
        tmp_path
    )  # fit() writes checkpoints/ and results/ relative to cwd
    train = DataLoader(
        PairedDataset(root, "train", 32), batch_size=8, shuffle=True, drop_last=True
    )
    return train, DataLoader(PairedDataset(root, "val"), batch_size=1)


def run_fit(loaders, run="t", epochs=6, seed=0):
    seed_everything(seed)
    tiny = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
    return fit(
        UWnet(), Loss(tiny), *loaders, epochs=epochs, lr=1e-3, run=run, device="cpu"
    )


def test_training_loss_falls(loaders):
    history = run_fit(loaders)
    assert len(history) == 6
    assert history[-1][0] < history[0][0]


def test_best_checkpoint_is_a_loadable_state_dict_with_the_best_val_psnr(loaders):
    history = run_fit(loaders)
    best = max(psnr for _, psnr, _ in history)
    model = load_finetuned("checkpoints/t_best.pth")
    assert score(model, loaders[1])[0] == pytest.approx(best, abs=1e-3)


def test_each_epoch_is_logged(loaders):
    run_fit(loaders, epochs=3)
    lines = Path("results/t/log.txt").read_text().splitlines()
    assert (
        len(lines) == 3
        and "val_psnr" in lines[0]
        and any("(best)" in line for line in lines)
    )


def test_same_seed_gives_the_same_run(loaders):
    a, b = run_fit(loaders, "a"), run_fit(loaders, "b")
    assert a == b


def test_seed_everything_fixes_all_generators():
    seed_everything(3)
    first = (torch.rand(2), np.random.rand(2))
    seed_everything(3)
    assert torch.equal(first[0], torch.rand(2)) and np.array_equal(
        first[1], np.random.rand(2)
    )
