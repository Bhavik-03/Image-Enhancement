"""Gradio demo: upload an underwater photo or video, get the enhanced version.

    python app.py        # then open http://127.0.0.1:7860 (local only)
Models: the authors' pretrained one, plus every checkpoints/*_best.pth made by train.py.
The model list is read at start-up, so restart the app after training a new model.
"""

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
import torch

from enhance import enhance_frame, enhance_video, load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED = "pretrained (authors)"


def model_choices():
    return [PRETRAINED] + sorted(p.name for p in Path("checkpoints").glob("*_best.pth"))


def get_model(name):
    path = None if name == PRETRAINED else Path("checkpoints") / name
    return load_cached(
        name, path.stat().st_mtime if path else 0
    )  # new weights on disk -> reload


@lru_cache(maxsize=2)
def load_cached(name, mtime):
    checkpoint = None if name == PRETRAINED else Path("checkpoints") / name
    return load_model(checkpoint).to(DEVICE).eval()


def to_browser_mp4(src, dst):
    """Re-encode to H.264 + yuv420p (even sides), which browsers play; OpenCV's mp4v they mostly do not."""
    scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src)]
    cmd += [
        "-vf",
        scale,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise gr.Error(f"video encoding failed: {result.stderr[:200]}")


def enhance_photo(rgb, name):
    if rgb is None:
        raise gr.Error("Upload a photo first.")
    return enhance_frame(get_model(name), rgb, DEVICE)


def enhance_clip(path, name, smooth):
    if path is None:
        raise gr.Error("Upload a video first.")
    out = (
        Path(tempfile.mkdtemp()) / "enhanced.mp4"
    )  # Gradio copies it into its own cache; /tmp is cleared by the OS
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        enhance_video(get_model(name), path, raw, DEVICE, smooth)
        to_browser_mp4(raw, out)
    return str(out)


def build_demo():
    with gr.Blocks(
        title="Underwater image enhancement", analytics_enabled=False
    ) as demo:
        gr.Markdown(
            "# Underwater image enhancement\nRestores colour and contrast with Shallow-UWnet."
        )
        model = gr.Dropdown(model_choices(), value=PRETRAINED, label="Model")
        with gr.Tab("Photo"):
            with gr.Row():
                photo_in = gr.Image(type="numpy", label="Original")
                photo_out = gr.Image(label="Enhanced")
            gr.Button("Enhance", variant="primary").click(
                enhance_photo, [photo_in, model], photo_out
            )
        with gr.Tab("Video"):
            with gr.Row():
                video_in = gr.Video(label="Original")
                video_out = gr.Video(label="Enhanced")
            smooth = gr.Slider(0, 0.9, value=0.3, step=0.1, label="Flicker smoothing")
            gr.Button("Enhance", variant="primary").click(
                enhance_clip, [video_in, model, smooth], video_out
            )
    return demo


if __name__ == "__main__":
    build_demo().launch()
