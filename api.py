"""FastAPI backend for the React frontend: upload a photo or video, get the enhanced version.

    uvicorn api:app --reload        # dev, paired with `npm run dev` in frontend/ (proxied, see vite.config.ts)
    python api.py                   # prod-ish: also serves frontend/dist/ if it has been built
Models: the authors' pretrained one, plus every checkpoints/*_best.pth made by train.py.
Video is processed as a background job (a clip can run well past an HTTP request's patience),
polled by job id; a photo is small enough to enhance inline within one request.
"""

import logging
import subprocess
import tempfile
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from enhance import enhance_frame, enhance_video, load_model

logger = logging.getLogger("image_enhancement.api")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED = "pretrained (authors)"
JOBS_DIR = Path(tempfile.gettempdir()) / "image-enhancement-jobs"

app = FastAPI(title="Image Enhancement API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def model_choices():
    return [PRETRAINED] + sorted(p.name for p in Path("checkpoints").glob("*_best.pth"))


def get_model(name):
    path = None if name == PRETRAINED else Path("checkpoints") / name
    if path is not None and not path.is_file():
        raise HTTPException(404, f"unknown model '{name}'")
    return load_cached(name, path.stat().st_mtime if path else 0)


@lru_cache(maxsize=2)
def load_cached(name, mtime):  # mtime busts the cache when a checkpoint is rewritten
    checkpoint = None if name == PRETRAINED else Path("checkpoints") / name
    return load_model(checkpoint).to(DEVICE).eval()


def to_browser_mp4(src, dst):
    """Re-encode to H.264 + yuv420p (even sides), which browsers play; OpenCV's mp4v they mostly do not."""
    scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src)]
    cmd += ["-vf", scale, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"video encoding failed: {result.stderr[:200]}")


@app.get("/api/models")
def list_models():
    return {"models": model_choices(), "default": PRETRAINED}


@app.post("/api/enhance/photo")
async def enhance_photo(file: UploadFile = File(...), model: str = Form(PRETRAINED)):  # noqa: B008 - FastAPI's own idiom for request parts
    raw = await file.read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = enhance_frame(get_model(model), rgb, DEVICE)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "could not encode result")
    return Response(bytes(buf), media_type="image/png")


# ---- Video: background job + polling, since a clip can take much longer than one request ----

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _run_video_job(job_id, src_path, model_name, smooth):
    try:
        model = get_model(model_name)
        raw_dst = JOBS_DIR / job_id / "raw.mp4"
        final_dst = JOBS_DIR / job_id / "enhanced.mp4"

        def on_frame(n, total):
            with _jobs_lock:
                _jobs[job_id]["progress"] = n / total if total else None
                _jobs[job_id]["frames_done"] = n

        n = enhance_video(model, src_path, raw_dst, DEVICE, smooth, on_frame=on_frame)
        to_browser_mp4(raw_dst, final_dst)
        with _jobs_lock:
            _jobs[job_id].update(status="done", progress=1.0, frames_done=n, result=str(final_dst))
    except Exception as exc:  # the job runs off-request, so this is the only place that can report failure
        logger.exception("video job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=str(exc))


@app.post("/api/enhance/video")
async def start_video_job(
    file: UploadFile = File(...), model: str = Form(PRETRAINED), smooth: float = Form(0.3)  # noqa: B008 - FastAPI's own idiom for request parts
):
    if not 0 <= smooth < 1:
        raise HTTPException(400, "smooth must be in [0, 1)")
    get_model(model)  # fail fast on an unknown model, before accepting the upload
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or ".mp4"
    src_path = job_dir / f"source{suffix}"
    src_path.write_bytes(await file.read())
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "progress": 0.0, "frames_done": 0}
    threading.Thread(
        target=_run_video_job, args=(job_id, src_path, model, smooth), daemon=True
    ).start()
    return {"job_id": job_id}


@app.get("/api/enhance/video/{job_id}")
def video_job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "result"}


@app.get("/api/enhance/video/{job_id}/result")
def video_job_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}, not done")
    return FileResponse(job["result"], media_type="video/mp4")


# ---- Serve the built frontend, if present (`python api.py` after `npm run build`) ----

_dist = Path(__file__).parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
