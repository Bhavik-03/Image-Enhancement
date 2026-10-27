import os

import cv2
import gradio as gr
import numpy as np
import pytest
import torch

import app
from model import UWnet


@pytest.fixture(autouse=True)
def tiny_model_and_cwd(tmp_path, monkeypatch):
    """Use a random model instead of the downloaded weights, and an empty checkpoints/ folder."""
    torch.manual_seed(0)
    monkeypatch.chdir(tmp_path)
    app.load_cached.cache_clear()
    monkeypatch.setattr(app, "load_model", lambda checkpoint=None: UWnet())
    monkeypatch.setattr(app, "DEVICE", "cpu")


def test_model_choices_list_pretrained_and_trained_checkpoints(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "ft1_best.pth").write_bytes(b"x")
    (tmp_path / "checkpoints" / "notes.txt").write_bytes(b"x")
    assert app.model_choices() == [app.PRETRAINED, "ft1_best.pth"]


def test_photo_keeps_its_size_and_type():
    rgb = np.random.default_rng(0).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    out = app.enhance_photo(rgb, app.PRETRAINED)
    assert out.shape == rgb.shape and out.dtype == np.uint8


def test_video_comes_back_as_a_playable_file(tmp_path):
    writer = cv2.VideoWriter(
        str(tmp_path / "in.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24)
    )
    for _ in range(4):
        writer.write(np.full((24, 32, 3), 90, np.uint8))
    writer.release()
    out = app.enhance_clip(str(tmp_path / "in.avi"), app.PRETRAINED, 0.3)
    cap = cv2.VideoCapture(out)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode().lower()
    assert out.endswith(".mp4") and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert fourcc in ("avc1", "h264")  # H.264, which browsers can play


def test_missing_uploads_give_a_friendly_error():
    with pytest.raises(gr.Error, match="photo"):
        app.enhance_photo(None, app.PRETRAINED)
    with pytest.raises(gr.Error, match="video"):
        app.enhance_clip(None, app.PRETRAINED, 0.3)


def test_the_interface_builds():
    assert isinstance(app.build_demo(), gr.Blocks)


def test_model_is_cached_and_reloaded_when_the_checkpoint_changes(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    ckpt = tmp_path / "checkpoints" / "ft_best.pth"
    ckpt.write_bytes(b"x")
    first = app.get_model("ft_best.pth")
    assert app.get_model("ft_best.pth") is first
    assert app.get_model(app.PRETRAINED) is app.get_model(app.PRETRAINED)
    os.utime(
        ckpt, (ckpt.stat().st_atime, ckpt.stat().st_mtime + 10)
    )  # train.py wrote better weights
    assert app.get_model("ft_best.pth") is not first
