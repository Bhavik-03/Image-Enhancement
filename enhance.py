"""Enhance an underwater photo or video at full resolution.

    python enhance.py photo.jpg                                  # authors' pretrained model
    python enhance.py clip.mp4 --checkpoint checkpoints/ft_best.pth --smooth 0.3
Results go to results/enhanced/ unless --out is given.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import AUTHORS_SHA256, load_finetuned, load_pretrained

PRETRAINED = "data/downloads/shallow_uwnet.ckpt"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_model(checkpoint=None):
    """Fine-tuned state_dict if given, else the authors' pretrained model."""
    return (
        load_finetuned(checkpoint)
        if checkpoint
        else load_pretrained(PRETRAINED, AUTHORS_SHA256)
    )


HALO = 16  # the net sees 11 px around each pixel (11 stacked 3x3 convs), so 16 px of overlap is exact
MAX_PIXELS = 4_000_000  # per band; a full 4K frame (8.3M px) needs 7 GB of VRAM, so big frames are split


@torch.no_grad()
def enhance_frame(model, rgb, device="cpu", max_pixels=MAX_PIXELS):
    """HxWx3 uint8 RGB -> enhanced HxWx3 uint8 RGB, same size (fully convolutional net).

    The model must already be in eval mode (enhance_image/enhance_video do this). Frames over
    max_pixels run in horizontal bands with HALO rows of overlap: same result, less memory.
    """
    h, w = rgb.shape[:2]
    rows = max(1, max_pixels // w)
    if rows >= h:
        return run_net(model, rgb, device)
    bands = []
    for y in range(0, h, rows):
        top, bottom = max(0, y - HALO), min(h, y + rows + HALO)
        out = run_net(model, rgb[top:bottom], device)
        bands.append(out[y - top : y - top + min(rows, h - y)])
    return np.concatenate(bands)


def run_net(model, rgb, device):
    x = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)[None]
        .float()
        .div(255)
        .to(device)
    )
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        y = model(x)
    return (
        (y.squeeze(0).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )


def enhance_image(model, src, dst, device="cpu"):
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {src}")
    out = enhance_frame(
        model.to(device).eval(), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), device
    )
    try:
        written = cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    except (
        cv2.error
    ) as e:  # OpenCV raises for an unknown extension, returns False for a bad path
        raise OSError(f"cannot write image {dst}: {e}") from e
    if not written:
        raise OSError(f"cannot write image {dst}")


def enhance_video(model, src, dst, device="cpu", smooth=0.0, on_frame=None):
    """Frame by frame. smooth in [0,1) blends each result with the previous output (less flicker).

    on_frame(n, total), if given, is called after each frame is written (n = frames done so far;
    total = cv2's frame count estimate, or None when the codec doesn't report one).
    """
    if not 0 <= smooth < 1:
        raise ValueError(f"smooth must be in [0, 1), got {smooth}")
    fourcc = {".avi": "MJPG", ".mp4": "mp4v", ".mov": "mp4v", ".mkv": "mp4v"}.get(
        Path(dst).suffix.lower()
    )
    if fourcc is None:
        raise ValueError(f"unsupported video output type '{Path(dst).suffix}'")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot read video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0  # also catches NaN
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None  # 0/NaN for some codecs -> unknown
    model, writer, prev, n = model.to(device).eval(), None, None, 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = enhance_frame(model, rgb, device).astype(np.float32)
            prev = out if prev is None else smooth * prev + (1 - smooth) * out
            if writer is None:
                # the size comes from the first decoded frame, so no file exists before then
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(
                    str(dst), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                )
                if not writer.isOpened():
                    raise OSError(f"cannot write video (codec or path problem?): {dst}")
            writer.write(cv2.cvtColor(prev.round().astype(np.uint8), cv2.COLOR_RGB2BGR))
            n += 1
            if on_frame is not None:
                on_frame(n, total)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if n == 0:
        raise ValueError(f"no frames read from {src}")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="image or video file")
    p.add_argument(
        "--checkpoint", help="state_dict from train.py (default: authors' pretrained)"
    )
    p.add_argument("--out", help="output path (default: results/enhanced/<name>)")
    p.add_argument(
        "--smooth", type=float, default=0.3, help="video only: 0 = off, 0.3 = light"
    )
    args = p.parse_args()

    src = Path(args.input)
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise ValueError(f"unsupported file type '{ext}'")
    dst = Path(args.out) if args.out else Path("results/enhanced") / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint)
    if ext in IMAGE_EXTS:
        enhance_image(model, src, dst, device)
    else:
        print(f"{enhance_video(model, src, dst, device, args.smooth)} frames")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
