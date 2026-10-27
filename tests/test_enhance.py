import cv2
import numpy as np
import pytest
import torch
from torch import nn

import enhance
from enhance import enhance_frame, enhance_image, enhance_video, load_model
from model import UWnet

FACTOR = 2.4  # gives an output std of ~77 on the test frame


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def lively_model():
    """Random init gives a nearly flat output; scale the weights so the output spans the range."""
    model = UWnet().eval()
    for p in model.parameters():
        p.data *= FACTOR
    return model


def write_video(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def read_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def test_frame_keeps_size_and_uint8_and_identity_model_is_a_noop():
    rgb = np.random.default_rng(0).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    out = enhance_frame(nn.Identity(), rgb)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_overshooting_output_is_clamped_to_the_valid_range():
    class Overshoot(nn.Module):
        def forward(self, x):
            return x * 10 - 2

    out = enhance_frame(Overshoot(), np.full((8, 8, 3), 128, np.uint8))
    assert out.min() >= 0 and out.max() <= 255


def test_image_file_roundtrip_with_a_real_model_keeps_the_size(tmp_path):
    cv2.imwrite(str(tmp_path / "in.png"), np.full((41, 67, 3), 90, np.uint8))
    enhance_image(UWnet(), tmp_path / "in.png", tmp_path / "out.png")
    assert cv2.imread(str(tmp_path / "out.png")).shape == (41, 67, 3)


def test_model_is_put_in_eval_mode_so_dropout_is_off(tmp_path):
    cv2.imwrite(
        str(tmp_path / "in.png"),
        np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8),
    )
    model = UWnet().train()
    enhance_image(model, tmp_path / "in.png", tmp_path / "a.png")
    enhance_image(model, tmp_path / "in.png", tmp_path / "b.png")
    assert not model.training
    assert np.array_equal(
        cv2.imread(str(tmp_path / "a.png")), cv2.imread(str(tmp_path / "b.png"))
    )


def test_video_keeps_frame_count_and_size(tmp_path):
    write_video(
        tmp_path / "in.avi", [np.full((24, 32, 3), 40 * i, np.uint8) for i in range(5)]
    )
    assert enhance_video(nn.Identity(), tmp_path / "in.avi", tmp_path / "out.avi") == 5
    frames = read_video(tmp_path / "out.avi")
    assert len(frames) == 5 and frames[0].shape == (24, 32, 3)


def test_smoothing_reduces_flicker_between_frames(tmp_path):
    flicker = [np.full((24, 32, 3), 20 if i % 2 else 230, np.uint8) for i in range(8)]
    write_video(tmp_path / "in.avi", flicker)

    def mean_jump(smooth):
        enhance_video(
            nn.Identity(), tmp_path / "in.avi", tmp_path / "out.avi", smooth=smooth
        )
        f = np.array(read_video(tmp_path / "out.avi"), dtype=np.float32)
        return np.abs(np.diff(f, axis=0)).mean()

    assert mean_jump(0.5) < 0.6 * mean_jump(0.0)


def test_bad_inputs_fail_early(tmp_path):
    with pytest.raises(FileNotFoundError):
        enhance_image(nn.Identity(), tmp_path / "missing.png", tmp_path / "o.png")
    with pytest.raises(FileNotFoundError):
        enhance_video(nn.Identity(), tmp_path / "missing.mp4", tmp_path / "o.mp4")
    with pytest.raises(ValueError, match="smooth"):
        enhance_video(nn.Identity(), tmp_path / "x.avi", tmp_path / "o.avi", smooth=1.0)


def test_load_model_reads_a_finetuned_state_dict(tmp_path):
    model = UWnet().eval()
    torch.save(model.state_dict(), tmp_path / "ft.pth")
    x = torch.rand(1, 3, 16, 16)
    assert torch.equal(model(x), load_model(tmp_path / "ft.pth").eval()(x))


def test_banded_processing_is_exact(monkeypatch):
    # 50 rows split into bands of 13: must equal one pass over the whole frame, bit for bit
    rgb = np.random.default_rng(2).integers(0, 256, (50, 20, 3), dtype=np.uint8)
    model = lively_model()
    whole = enhance_frame(model, rgb, max_pixels=10**9).astype(int)
    assert (
        whole.std() > 30
    )  # the comparison below is only meaningful if the output has signal
    banded = enhance_frame(model, rgb, max_pixels=20 * 13).astype(int)
    assert np.array_equal(whole, banded)
    # and the test must notice an overlap one pixel short of the net's 11 px reach
    monkeypatch.setattr(enhance, "HALO", 10)  # one short of the reach
    assert not np.array_equal(whole, enhance_frame(model, rgb, max_pixels=20 * 13))


def test_odd_sized_video_keeps_whatever_size_the_source_decodes_to(tmp_path):
    # the MJPG writer may round odd sizes, so compare with the source as read back
    write_video(tmp_path / "in.avi", [np.full((25, 33, 3), 80, np.uint8)] * 3)
    assert enhance_video(nn.Identity(), tmp_path / "in.avi", tmp_path / "out.avi") == 3
    assert (
        read_video(tmp_path / "out.avi")[0].shape
        == read_video(tmp_path / "in.avi")[0].shape
    )


def test_video_without_frames_fails_and_writes_no_output(tmp_path):
    (tmp_path / "empty.avi").write_bytes(b"")
    with pytest.raises((FileNotFoundError, ValueError)):
        enhance_video(nn.Identity(), tmp_path / "empty.avi", tmp_path / "o.avi")
    assert not (tmp_path / "o.avi").exists()


def test_unsupported_output_types_are_refused(tmp_path):
    write_video(tmp_path / "in.avi", [np.full((24, 32, 3), 80, np.uint8)] * 2)
    with pytest.raises(ValueError, match="unsupported"):
        enhance_video(nn.Identity(), tmp_path / "in.avi", tmp_path / "o.gif")
    cv2.imwrite(str(tmp_path / "in.png"), np.full((8, 8, 3), 80, np.uint8))
    with pytest.raises(OSError, match="cannot write"):
        enhance_image(nn.Identity(), tmp_path / "in.png", tmp_path / "o.unknown")
