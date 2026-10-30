import time

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import api
from model import UWnet


@pytest.fixture(autouse=True)
def tiny_model_and_cwd(tmp_path, monkeypatch):
    """Use a random model instead of the downloaded weights, and an empty checkpoints/ folder."""
    torch.manual_seed(0)
    monkeypatch.chdir(tmp_path)
    api.load_cached.cache_clear()
    monkeypatch.setattr(api, "load_model", lambda checkpoint=None: UWnet())
    monkeypatch.setattr(api, "DEVICE", "cpu")
    monkeypatch.setattr(api, "JOBS_DIR", tmp_path / "jobs")
    api._jobs.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


def _png_bytes(h=37, w=53):
    rgb = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _avi_bytes(n_frames=4, h=24, w=32, path=None):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for _ in range(n_frames):
        writer.write(np.full((h, w, 3), 90, np.uint8))
    writer.release()


def test_model_list_has_pretrained_and_trained_checkpoints(client, tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "ft1_best.pth").write_bytes(b"x")
    (tmp_path / "checkpoints" / "notes.txt").write_bytes(b"x")
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"] == [api.PRETRAINED, "ft1_best.pth"]
    assert body["default"] == api.PRETRAINED


def test_enhance_photo_returns_a_same_shape_png(client):
    files = {"file": ("photo.png", _png_bytes(37, 53), "image/png")}
    resp = client.post("/api/enhance/photo", files=files, data={"model": api.PRETRAINED})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    out = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (37, 53, 3)


def test_enhance_photo_rejects_garbage_bytes(client):
    files = {"file": ("not-a-photo.png", b"definitely not an image", "image/png")}
    resp = client.post("/api/enhance/photo", files=files)
    assert resp.status_code == 400


def test_enhance_photo_rejects_unknown_model(client):
    files = {"file": ("photo.png", _png_bytes(), "image/png")}
    resp = client.post("/api/enhance/photo", files=files, data={"model": "nope.pth"})
    assert resp.status_code == 404


def test_video_job_runs_to_completion_and_serves_a_playable_result(client, tmp_path):
    src = tmp_path / "in.avi"
    _avi_bytes(4, 24, 32, src)
    with open(src, "rb") as f:
        resp = client.post(
            "/api/enhance/video",
            files={"file": ("in.avi", f, "video/x-msvideo")},
            data={"model": api.PRETRAINED, "smooth": "0.3"},
        )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    for _ in range(50):
        status = client.get(f"/api/enhance/video/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert status["status"] == "done", status
    assert status["progress"] == 1.0
    assert status["frames_done"] == 4

    result = client.get(f"/api/enhance/video/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["content-type"] == "video/mp4"


def test_unknown_job_id_is_404(client):
    assert client.get("/api/enhance/video/does-not-exist").status_code == 404
    assert client.get("/api/enhance/video/does-not-exist/result").status_code == 404


def test_video_job_rejects_bad_smooth_up_front(client, tmp_path):
    src = tmp_path / "in.avi"
    _avi_bytes(1, 16, 16, src)
    with open(src, "rb") as f:
        resp = client.post(
            "/api/enhance/video",
            files={"file": ("in.avi", f, "video/x-msvideo")},
            data={"model": api.PRETRAINED, "smooth": "1.5"},
        )
    assert resp.status_code == 400
