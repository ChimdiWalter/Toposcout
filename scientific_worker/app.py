#!/usr/bin/env python3
"""Scientific worker HTTP API (Cloud Run).

GET  /health    liveness + model/checkpoint identity (loads nothing heavy)
POST /segment   {run_id, image_uri, attempt} -> deterministic evidence JSON

The request cannot choose executables, checkpoints, thresholds, filters,
architectures, or output locations (schemas.SegmentRequest is a closed
schema; all of those are fixed server-side). Attempt semantics are frozen:
attempt 1 = raw T0 mask at 0.5, attempt 2 = the SAME probability map + the
validated 500 px significance filter (fetched from the run's attempt-1 cache
in GCS when available).

Run locally:
    TOPOSCOUT_SW_ALLOW_LOCAL_IO=1 uvicorn scientific_worker.app:app --port 8081
"""
from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .model_loader import ADAPTER_NAME, CHECKPOINT_PATH, MODEL_VERSION
from .schemas import SegmentRequest, SegmentResponse, WorkerError
from .storage import (
    StorageError, load_cached_prob, load_image_bgr, save_attempt_artifacts,
)

app = FastAPI(title="TopoScout scientific worker", version="0.1.0", docs_url=None, redoc_url=None)

_started = time.time()
_segment_lock = threading.Lock()  # concurrency=1 service; serialize defensively


def _engine():
    import torch
    torch.set_num_threads(int(os.environ.get("TOPOSCOUT_REAL_THREADS", "2")))
    from .inference import InferenceEngine
    return InferenceEngine.get()


@app.get("/health")
def health() -> dict:
    import sys
    mod = sys.modules.get("scientific_worker.inference") or sys.modules.get(f"{__package__}.inference")
    loaded = bool(mod and mod.InferenceEngine._instance is not None)
    return {
        "status": "ok",
        "service": "toposcout-scientific-worker",
        "adapter": ADAPTER_NAME,
        "model_version": MODEL_VERSION,
        "checkpoint_present": CHECKPOINT_PATH.is_file(),
        "model_loaded": loaded,
        "uptime_seconds": round(time.time() - _started, 1),
    }


def _error(status_code: int, reason: str, detail: str = "") -> JSONResponse:
    return JSONResponse(status_code=status_code,
                        content=WorkerError(reason=reason, detail=detail or None).model_dump())


@app.post("/segment")
def segment(req: SegmentRequest):
    try:
        img = load_image_bgr(req.image_uri)
    except StorageError as exc:
        code = 404 if exc.reason == "input_not_found" else 400
        return _error(code, exc.reason, exc.detail)

    try:
        engine = _engine()
    except Exception as exc:  # noqa: BLE001 — model must never half-load silently
        return _error(500, "model_load_failed", str(exc))

    cached = None
    if req.attempt == 2:
        cached = load_cached_prob(req.run_id, img.shape[:2])

    with _segment_lock:
        result = engine.segment(img, req.attempt, cached_prob=cached)

    try:
        uris = save_attempt_artifacts(
            req.run_id, req.attempt, result["mask"], result["overlay"],
            result["prob"] if req.attempt == 1 else None, img.shape[:2])
    except StorageError as exc:
        return _error(500, "artifact_store_failed", f"{exc.reason} {exc.detail}")

    return SegmentResponse(
        status="ok",
        adapter=result["adapter"],
        run_id=req.run_id,
        attempt=req.attempt,
        mask_uri=uris["mask_uri"],
        overlay_uri=uris["overlay_uri"],
        prob_uri=uris["prob_uri"],
        prob_reused=result["prob_reused"],
        foreground_fraction=result["foreground_fraction"],
        prob_threshold=result["prob_threshold"],
        min_area_px=result["min_area_px"],
        n_components_significant=result["n_components_significant"],
        image_height=result["image_height"],
        image_width=result["image_width"],
        model_version=result["model_version"],
        checkpoint_sha256=result["checkpoint_sha256"],
        runtime_seconds=result["runtime_seconds"],
    )
