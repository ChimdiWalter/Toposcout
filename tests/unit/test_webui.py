"""M6 Phase 3/4/6 tests: Cloud Run-safe async UI, canonical display strings,
durable artifacts. Hermetic: stubbed inference engine, local storage/runstate,
no Gemini, no Cloud Tasks client."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from toposcout_core import runstate

WEBUI_DIR = Path(__file__).resolve().parents[2] / "webui"


# ---------- fixtures ----------

@pytest.fixture()
def ui(monkeypatch, tmp_path: Path):
    """UI TestClient in dev mode with a real (stub-engine) worker behind it."""
    # local durable backends
    monkeypatch.delenv("TOPOSCOUT_GCS_BUCKET", raising=False)
    monkeypatch.setenv("TOPOSCOUT_SW_ALLOW_LOCAL_IO", "1")
    monkeypatch.setenv("TOPOSCOUT_SW_LOCAL_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.delenv("TOPOSCOUT_RUNSTATE", raising=False)
    monkeypatch.setenv("TOPOSCOUT_RUNSTATE_DIR", str(tmp_path / "runstate"))
    monkeypatch.setenv("TOPOSCOUT_OUTPUT_DIR", str(tmp_path / "outputs"))
    # dev processing: direct /internal/process calls, deterministic tool sequence
    monkeypatch.delenv("TOPOSCOUT_TASKS_QUEUE", raising=False)
    monkeypatch.setenv("TOPOSCOUT_TASKS_MODE", "local")
    monkeypatch.setenv("TOPOSCOUT_UI_ORCHESTRATOR", "tools")
    monkeypatch.setenv("TOPOSCOUT_WORKER_URL", "http://worker.test")
    monkeypatch.setenv("TOPOSCOUT_WORKER_AUTH", "none")

    # real worker app with stubbed engine, bridged into cloud_tools
    import scientific_worker.app as worker_app
    from tests.unit.test_cloud_worker import StubEngine
    stub = StubEngine()
    monkeypatch.setattr(worker_app, "_engine", lambda: stub)
    worker_client = TestClient(worker_app.app)

    import app.cloud_tools as ct

    class _Resp:
        def __init__(self, r): self._r = r
        @property
        def status_code(self): return self._r.status_code
        def json(self): return self._r.json()

    monkeypatch.setattr(ct.requests, "post",
                        lambda url, json=None, headers=None, timeout=None:
                        _Resp(worker_client.post("/segment", json=json)))

    import webui.app as webui_app
    return TestClient(webui_app.app), webui_app, tmp_path


def _png_bytes(size=(96, 128)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((size[0], size[1], 3), 120, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, data: bytes, name="leaf.png"):
    return client.post("/api/runs", files={"image": (name, data, "image/png")})


# ---------- upload validation ----------

def test_upload_returns_202_and_stages_durably(ui):
    client, webui_app, tmp = ui
    r = _upload(client, _png_bytes())
    assert r.status_code == 202
    body = r.json()
    assert body["queued"] == "dev-manual"
    run = runstate.get_run(body["run_id"])
    assert run["state"] == "RECEIVED"
    assert run["image_uri"]
    # staged straight into canonical durable storage
    assert Path(run["image_uri"]).parent.name == "input"


def test_upload_byte_limit(ui, monkeypatch):
    client, webui_app, _ = ui
    monkeypatch.setattr(webui_app, "MAX_UPLOAD_BYTES", 1000)
    r = _upload(client, _png_bytes((512, 512)))
    assert r.status_code == 413


def test_upload_rejects_undecodable_bytes(ui):
    client, _, _ = ui
    r = _upload(client, b"MZ\x90\x00 this is not an image")
    assert r.status_code == 400
    assert "decode" in r.json()["detail"]


def test_upload_rejects_bad_suffix(ui):
    client, _, _ = ui
    r = _upload(client, _png_bytes(), name="x.exe")
    assert r.status_code == 400


# ---------- internal processing ----------

def _process(client, body: dict):
    return client.post("/internal/process", json=body)


def test_process_full_run_and_canonical_display(ui):
    client, _, tmp = ui
    run_id = _upload(client, _png_bytes()).json()["run_id"]
    uri = runstate.get_run(run_id)["image_uri"]

    p = _process(client, {"run_id": run_id, "image_uri": uri})
    assert p.status_code == 200
    assert p.json()["state"] in {"COMPLETE", "HUMAN_REVIEW"}

    status = client.get(f"/api/runs/{run_id}").json()
    topo = status["evidence"]["attempt1"]["topology"]
    stored = runstate.get_evidence(run_id, "topology", attempt=1)
    # display strings are EXACTLY the canonical json serialization
    for key in ("fragmentation_score", "largest_component_fraction", "beta_0"):
        assert topo["display"][key] == json.dumps(stored[key])
    seg = status["evidence"]["attempt1"]["segmentation"]
    seg_stored = runstate.get_evidence(run_id, "segmentation", attempt=1)
    assert seg["display"]["foreground_fraction"] == json.dumps(seg_stored["foreground_fraction"])
    # no local paths leak to the browser
    assert not any(k.endswith("_path") for k in seg)

    # artifacts all served from durable storage
    for name in ("input", "mask", "overlay", "report"):
        assert client.get(f"/api/runs/{run_id}/artifact/{name}").status_code == 200
    # report on-disk values equal what the status API serves as display strings
    report = json.loads(client.get(f"/api/runs/{run_id}/artifact/report").content)
    assert json.dumps(report["runs"][0]["topology"]["fragmentation_score"]) == \
        topo["display"]["fragmentation_score"]


def test_process_is_idempotent_when_complete(ui):
    client, _, _ = ui
    run_id = _upload(client, _png_bytes()).json()["run_id"]
    uri = runstate.get_run(run_id)["image_uri"]
    assert _process(client, {"run_id": run_id, "image_uri": uri}).status_code == 200
    again = _process(client, {"run_id": run_id, "image_uri": uri})
    assert again.status_code == 200
    assert again.json()["status"] == "already_complete"


def test_process_rejects_duplicate_active_run(ui):
    client, _, _ = ui
    run_id = _upload(client, _png_bytes()).json()["run_id"]
    uri = runstate.get_run(run_id)["image_uri"]
    runstate.set_state(run_id, "SEGMENTING")  # simulate an in-flight execution
    assert _process(client, {"run_id": run_id, "image_uri": uri}).status_code == 409


def test_process_payload_is_closed(ui):
    client, _, _ = ui
    run_id = _upload(client, _png_bytes()).json()["run_id"]
    uri = runstate.get_run(run_id)["image_uri"]
    for extra in ({"model": "x"}, {"threshold": 0.1}, {"checkpoint": "c"},
                  {"bucket": "b"}, {"executable": "/bin/sh"}, {"output_dir": "/e"}):
        r = _process(client, {"run_id": run_id, "image_uri": uri, **extra})
        assert r.status_code == 422


def test_process_unknown_and_untrusted(ui):
    client, _, _ = ui
    assert _process(client, {"run_id": "nope-123", "image_uri": "x.png"}).status_code == 400
    r = _upload(client, _png_bytes())
    uri = runstate.get_run(r.json()["run_id"])["image_uri"]
    # valid uri shape but nonexistent run doc
    other = uri.replace(r.json()["run_id"], "ghost-run")
    assert _process(client, {"run_id": "ghost-run", "image_uri": other}).status_code in (400, 404)


def test_process_disabled_without_dev_optin(ui, monkeypatch):
    client, _, _ = ui
    monkeypatch.delenv("TOPOSCOUT_TASKS_MODE", raising=False)
    assert _process(client, {"run_id": "r", "image_uri": "u"}).status_code == 403


def test_enqueue_payload_only_run_id_and_image_uri(ui, monkeypatch):
    client, webui_app, _ = ui
    captured = {}
    monkeypatch.setattr(webui_app, "_enqueue_process_task",
                        lambda run_id, image_uri, **kw: captured.update(
                            run_id=run_id, image_uri=image_uri, extra=kw) or "captured")
    r = _upload(client, _png_bytes())
    assert r.json()["queued"] == "captured"
    assert set(captured) == {"run_id", "image_uri", "extra"} and captured["extra"] == {}


# ---------- durability: original survives instance restart ----------

def test_input_artifact_survives_staging_loss(ui):
    client, _, tmp = ui
    data = _png_bytes()
    run_id = _upload(client, data).json()["run_id"]
    # the upload staging tempdir is already deleted by the endpoint itself;
    # a fresh "instance" (nothing in memory) must still serve the original
    r = client.get(f"/api/runs/{run_id}/artifact/input")
    assert r.status_code == 200
    assert r.content == data


# ---------- static guarantees ----------

def test_no_daemon_threads_or_background_tasks_in_webui():
    src = (WEBUI_DIR / "app.py").read_text()
    assert "import threading" not in src
    assert "threading.Thread" not in src
    assert "daemon=True" not in src
    assert "BackgroundTasks" not in src


def test_index_html_never_reformats_numbers():
    html = (WEBUI_DIR / "index.html").read_text()
    assert "toPrecision" not in html
    assert "toFixed" not in html
