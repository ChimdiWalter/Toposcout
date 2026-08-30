#!/usr/bin/env python3
"""TopoScout image-first web UI (M6: Cloud Run-safe asynchronous execution).

Public surface:
    GET  /                       the scientific interface
    POST /api/runs               upload → validate → stage → enqueue → 202
    GET  /api/runs/{run_id}      run state + exact evidence (+ display strings)
    GET  /api/runs/{run_id}/artifact/{input|mask|overlay|report}

Internal surface:
    POST /internal/process       Cloud Tasks-invoked; drives the approved
                                 deployed Gemini agent workflow for one run.

Design rules (M6 Phase 3/4):
- No daemon threads, no framework background-task helpers, no in-memory run ownership;
  Firestore/GCS are the only durable state. Uploads are staged straight into
  the canonical GCS input prefix and processing happens in a Cloud Task
  request, so request-based billing is safe.
- Every number shown to a person is a server-built display string serialized
  exactly like the canonical report JSON (json.dumps); the browser never
  reformats values.
- The internal endpoint accepts ONLY {run_id, image_uri} (closed schema) and
  verifies the Cloud Tasks OIDC identity in cloud mode.
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests as _requests
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict

import app.cloud_tools as ct
from scientific_worker import storage
from scientific_worker.storage import StorageError
from toposcout_core import runstate

INDEX_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

MAX_UPLOAD_BYTES = int(os.environ.get("TOPOSCOUT_UI_MAX_UPLOAD_BYTES", str(24 * 1024 * 1024)))

# Cloud Tasks wiring (unset ⇒ dev mode: tests/dev call /internal/process directly)
TASKS_QUEUE_ENV = "TOPOSCOUT_TASKS_QUEUE"          # projects/.../locations/.../queues/...
TASKS_SA_ENV = "TOPOSCOUT_TASKS_SA"                # OIDC identity Cloud Tasks uses
UI_BASE_URL_ENV = "TOPOSCOUT_UI_BASE_URL"          # public URL of this service
AGENT_URL_ENV = "TOPOSCOUT_AGENT_URL"              # deployed ADK agent service
AGENT_APP = os.environ.get("TOPOSCOUT_AGENT_APP", "app")
ORCHESTRATOR_ENV = "TOPOSCOUT_UI_ORCHESTRATOR"     # "agent" (default) | "tools" (dev)

TERMINAL_STATES = {"COMPLETE", "HUMAN_REVIEW"}

app = FastAPI(title="TopoScout", docs_url=None, redoc_url=None)


def _tasks_queue() -> str | None:
    return os.environ.get(TASKS_QUEUE_ENV, "").strip() or None


def _ui_base_url() -> str:
    return os.environ.get(UI_BASE_URL_ENV, "").strip().rstrip("/")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    # never leak stack traces or filesystem detail to the public surface
    return JSONResponse(status_code=500, content={"status": "failed", "reason": "internal_error"})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


# ── cross-domain portability pilots (frozen artifacts bundled at build) ──────

PILOTS_DIR = Path(os.environ.get("TOPOSCOUT_PILOTS_DIR", "artifacts/pilots"))
PILOTS_HTML_PATH = Path(__file__).parent / "pilots.html"


@app.get("/pilots", response_class=HTMLResponse)
def pilots_page() -> str:
    if not PILOTS_HTML_PATH.is_file():
        raise HTTPException(404, "pilots page not available")
    return PILOTS_HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/pilots")
def pilots_evidence():
    out = {}
    for f in sorted(PILOTS_DIR.glob("*/evidence.json")):
        try:
            out[f.parent.name] = json.loads(f.read_text())
        except ValueError:
            continue
    return out


_PILOT_ASSETS = {"input": ("input", None), "mask": ("mask.png", "image/png"),
                 "overlay": ("overlay.jpg", "image/jpeg")}


def _pilot_imagery_public(domain: str) -> bool:
    try:
        ev = json.loads((PILOTS_DIR / domain / "evidence.json").read_text())
        return bool(ev.get("public_imagery", False))
    except (OSError, ValueError):
        return False


@app.get("/pilots/assets/{domain}/{name}")
def pilot_asset(domain: str, name: str):
    if _SAFE_NAME_RE.search(domain) or name not in _PILOT_ASSETS:
        raise HTTPException(404, "unknown pilot asset")
    if not _pilot_imagery_public(domain):
        # third-party licensing (e.g. CC BY-NC-SA datasets): numbers stay public,
        # imagery does not
        raise HTTPException(404, "imagery omitted from public redistribution")
    fname, ctype = _PILOT_ASSETS[name]
    if name == "input":
        preview = PILOTS_DIR / domain / "input_preview.jpg"
        if preview.is_file():  # browser-friendly (e.g. TIFF originals)
            return Response(content=preview.read_bytes(), media_type="image/jpeg")
        for p in sorted((PILOTS_DIR / domain).glob("input.*")):
            return Response(content=p.read_bytes(),
                            media_type=_CONTENT_TYPES.get(p.suffix.lower(), "application/octet-stream"))
        raise HTTPException(404, "input not found")
    p = PILOTS_DIR / domain / fname
    if not p.is_file():
        raise HTTPException(404, "asset not found")
    return Response(content=p.read_bytes(), media_type=ctype)


# ── upload ───────────────────────────────────────────────────────────────────

def _decode_ok(data: bytes) -> bool:
    from PIL import Image
    try:
        Image.open(io.BytesIO(data)).verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001
        return False


def _enqueue_process_task(run_id: str, image_uri: str) -> str:
    """Enqueue exactly {run_id, image_uri} for /internal/process. Returns mode."""
    queue = _tasks_queue()
    if not queue:
        return "dev-manual"  # dev/tests invoke /internal/process themselves
    from google.cloud import tasks_v2
    client = tasks_v2.CloudTasksClient()
    payload = {"run_id": run_id, "image_uri": image_uri}
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{_ui_base_url()}/internal/process",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload).encode(),
            "oidc_token": {
                "service_account_email": os.environ.get(TASKS_SA_ENV, "").strip(),
                "audience": _ui_base_url(),
            },
        },
        "dispatch_deadline": {"seconds": 1500},
    }
    client.create_task(parent=queue, task=task)
    return "cloud-tasks"


@app.post("/api/runs", status_code=202)
async def create_run(image: UploadFile):
    suffix = Path(image.filename or "upload.png").suffix.lower()
    if suffix not in storage.ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(400, f"unsupported image type {suffix!r}")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
    if not data or not _decode_ok(data):
        raise HTTPException(400, "file does not decode as a supported image")

    stem = _SAFE_NAME_RE.sub("_", Path(image.filename or "upload").stem)[:40] or "upload"
    run_id = f"{stem}-{uuid.uuid4().hex[:8]}"

    # stage the original straight into canonical durable storage
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / f"{stem}{suffix}"
        staged.write_bytes(data)
        image_uri = storage.upload_input(run_id, staged)
    storage.validate_run_input(run_id, image_uri)
    runstate.create_run(run_id, stem, image_uri=image_uri)

    mode = _enqueue_process_task(run_id, image_uri)
    return {"run_id": run_id, "sample_id": stem, "queued": mode}


# ── internal processing (Cloud Tasks target) ─────────────────────────────────

class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # nothing but run_id/image_uri

    run_id: str
    image_uri: str


def _verify_tasks_identity(request: Request) -> None:
    """Cloud mode: the caller must present the queue's OIDC identity."""
    if not _tasks_queue():
        if os.environ.get("TOPOSCOUT_TASKS_MODE", "").strip() == "local":
            return  # explicit dev opt-in
        raise HTTPException(403, "processing endpoint disabled")
    authz = request.headers.get("authorization", "")
    if not authz.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    import google.auth.transport.requests
    import google.oauth2.id_token
    try:
        claims = google.oauth2.id_token.verify_oauth2_token(
            authz.removeprefix("Bearer ").strip(),
            google.auth.transport.requests.Request(),
            audience=_ui_base_url())
    except Exception:  # noqa: BLE001
        raise HTTPException(401, "invalid token")
    expected = os.environ.get(TASKS_SA_ENV, "").strip()
    if not expected or claims.get("email") != expected or not claims.get("email_verified"):
        raise HTTPException(403, "unauthorized identity")


def _invoke_deployed_agent(run_id: str, image_uri: str) -> None:
    """Drive the approved deployed Gemini/ADK agent for this run (blocking)."""
    base = os.environ.get(AGENT_URL_ENV, "").strip().rstrip("/")
    if not base:
        raise RuntimeError("agent_not_configured")
    import google.auth.transport.requests
    import google.oauth2.id_token
    token = google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), base)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    user_id = "toposcout-ui"
    s = _requests.post(f"{base}/apps/{AGENT_APP}/users/{user_id}/sessions",
                       headers=headers, json={}, timeout=60)
    s.raise_for_status()
    session_id = s.json()["id"]
    message = (f"Process this scientific imaging run. run_id: {run_id} "
               f"image_uri: {image_uri}")
    r = _requests.post(f"{base}/run", headers=headers, timeout=1500, json={
        "app_name": AGENT_APP, "user_id": user_id, "session_id": session_id,
        "new_message": {"role": "user", "parts": [{"text": message}]},
    })
    r.raise_for_status()


def _run_tool_sequence(run_id: str, image_uri: str) -> None:
    """Dev/test orchestrator: the same approved deterministic tool sequence."""
    qc = ct.inspect_image(run_id, image_uri)
    if not qc.get("qc_pass"):
        ct.bounded_policy(run_id, attempt=1)
        ct.create_report(run_id)
        return
    for attempt in (1, 2):
        seg = ct.run_scientific_segmentation(run_id, image_uri, attempt=attempt)
        if seg.get("status") != "ok":
            break
        ct.audit_topology(run_id, attempt=attempt)
        if ct.bounded_policy(run_id, attempt=attempt)["action"] != "RETRY":
            break
    ct.create_report(run_id)


@app.post("/internal/process")
def internal_process(payload: ProcessRequest, request: Request):
    _verify_tasks_identity(request)
    try:
        storage.validate_run_input(payload.run_id, payload.image_uri)
    except StorageError as exc:
        raise HTTPException(400, exc.reason)

    run = runstate.get_run(payload.run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")
    if run["state"] in TERMINAL_STATES:
        return {"status": "already_complete", "run_id": payload.run_id,
                "state": run["state"]}  # idempotent re-delivery
    if run["state"] != "RECEIVED":
        raise HTTPException(409, "run already processing")

    orchestrator = os.environ.get(ORCHESTRATOR_ENV, "agent").strip() or "agent"
    try:
        if orchestrator == "tools":
            _run_tool_sequence(payload.run_id, payload.image_uri)
        else:
            _invoke_deployed_agent(payload.run_id, payload.image_uri)
    finally:
        # terminal state must always be durable, even on orchestrator failure
        final = runstate.get_run(payload.run_id)
        if final is not None and final["state"] not in TERMINAL_STATES:
            runstate.set_state(payload.run_id, "HUMAN_REVIEW",
                               decision={"action": "HUMAN_REVIEW",
                                         "reason": "pipeline_error"})

    final = runstate.get_run(payload.run_id)
    return {"status": "processed", "run_id": payload.run_id, "state": final["state"]}


# ── run status + exact display strings ───────────────────────────────────────

_DISPLAY_KEYS = {
    "segmentation": ["foreground_fraction", "prob_threshold", "min_area_px",
                     "n_components_significant", "runtime_seconds"],
    "topology": ["beta_0", "beta_1", "foreground_pixels", "tiny_components",
                 "fragmentation_score", "largest_component_fraction"],
    "qc": ["mean_intensity", "contrast_std", "sharpness_score"],
}


def _with_display(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach canonical display strings: exactly how the report JSON serializes
    each value (json.dumps), so the browser shows verbatim evidence."""
    out = {k: v for k, v in payload.items() if not k.endswith("_path")}
    out["display"] = {k: json.dumps(payload[k]) for k in _DISPLAY_KEYS.get(kind, [])
                      if k in payload}
    return out


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    if not storage.RUN_ID_RE.match(run_id):
        raise HTTPException(400, "invalid run_id")
    run = runstate.get_run(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")

    evidence: dict[str, Any] = {}
    for attempt in (1, 2):
        entry = {}
        for kind in ("segmentation", "topology", "decision"):
            payload = runstate.get_evidence(run_id, kind, attempt=attempt)
            if payload is not None:
                entry[kind] = _with_display(kind, payload)
        if entry:
            evidence[f"attempt{attempt}"] = entry
    qc = runstate.get_evidence(run_id, "qc")
    if qc:
        evidence["qc"] = _with_display("qc", qc)
    return {"run": run, "evidence": evidence}


# ── artifacts (durable storage only — survives restarts/instances) ───────────

def _serve(data: bytes | None, media_type: str, what: str) -> Response:
    if data is None:
        raise HTTPException(404, f"artifact not available: {what}")
    return Response(content=data, media_type=media_type)


_CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".tif": "image/tiff", ".tiff": "image/tiff"}


@app.get("/api/runs/{run_id}/artifact/{name}")
def artifact(run_id: str, name: str, attempt: int = 1):
    if not storage.RUN_ID_RE.match(run_id) or attempt not in (1, 2):
        raise HTTPException(404, "unknown artifact")
    prefix = storage.run_prefix(run_id)

    if name == "input":
        found = storage.find_input(run_id)
        if not found:
            raise HTTPException(404, "input not found")
        key, data = found
        return _serve(data, _CONTENT_TYPES.get(Path(key).suffix.lower(), "application/octet-stream"), name)
    if name == "mask":
        return _serve(storage._get_bytes(f"{prefix}/attempt{attempt}/mask.png"), "image/png", name)
    if name == "overlay":
        return _serve(storage._get_bytes(f"{prefix}/attempt{attempt}/overlay.jpg"), "image/jpeg", name)
    if name == "report":
        return _serve(storage._get_bytes(f"{prefix}/report/report.json"), "application/json", name)
    raise HTTPException(404, "unknown artifact")
