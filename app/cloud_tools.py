"""Cloud-mode ADK tools (Milestone 5 Phase 5).

The LLM decides WHEN to call these approved tools; it can never choose the
model, checkpoint, threshold, filter, executable, bucket path, or worker URL —
all fixed by trusted configuration:

    TOPOSCOUT_WORKER_URL    scientific worker base URL (Cloud Run, private)
    TOPOSCOUT_WORKER_AUTH   "id_token" (default; SA identity token) | "none" (local dev)
    TOPOSCOUT_GCS_BUCKET    trusted artifact bucket
    TOPOSCOUT_RUNSTATE      firestore | local

Every deterministic tool output is persisted verbatim via toposcout_core.runstate
keyed by run_id; create_report restores the exact stored values, so Gemini
never reconstructs scientific numbers from conversation history. Decisions
remain the exclusive authority of toposcout_core.tools.bounded_policy.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import requests

from scientific_worker import storage
from scientific_worker.storage import StorageError
from toposcout_core import runstate, tools as core

WORKER_URL_ENV = "TOPOSCOUT_WORKER_URL"
WORKER_AUTH_ENV = "TOPOSCOUT_WORKER_AUTH"
WORKER_TIMEOUT_ENV = "TOPOSCOUT_WORKER_TIMEOUT_S"
DEFAULT_WORKER_TIMEOUT_S = 900.0


def _worker_url() -> str:
    return os.environ.get(WORKER_URL_ENV, "").strip().rstrip("/")


def _auth_headers(audience: str) -> dict[str, str]:
    if os.environ.get(WORKER_AUTH_ENV, "").strip() == "none":
        return {}
    import google.auth.transport.requests
    import google.oauth2.id_token
    req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(req, audience)
    return {"Authorization": f"Bearer {token}"}


def _download_to_local(uri: str) -> str:
    """Fetch a small artifact (mask) so local topology audit can read it."""
    if not uri.startswith("gs://"):
        return uri  # dev mode: already a local path
    from google.cloud import storage as gcs
    bucket_name, key = uri[5:].split("/", 1)
    cache = Path(tempfile.gettempdir()) / "toposcout_artifacts" / key
    cache.parent.mkdir(parents=True, exist_ok=True)
    gcs.Client().bucket(bucket_name).blob(key).download_to_filename(str(cache))
    return str(cache)


def inspect_image(run_id: str, image_uri: str) -> dict[str, Any]:
    """Inspect the run's canonical input image for acquisition/quality problems.

    image_uri must be this run's canonical trusted-storage input
    (gs://<trusted bucket>/runs/<run_id>/input/<file>); any other bucket,
    prefix, or run's object is rejected. The image is fetched into a
    run-specific temporary directory server-side — no caller-controlled paths.
    """
    try:
        local = storage.download_run_input(run_id, image_uri)
    except StorageError as exc:
        return {"status": "failed", "reason": exc.reason, "detail": exc.detail or None}
    if runstate.get_run(run_id) is None:
        runstate.create_run(run_id, Path(image_uri).stem, image_uri=image_uri)
    runstate.set_state(run_id, "QC")
    result = core.inspect_image(str(local))
    # evidence identifies the input by its canonical URI, never a container path
    result["image_path"] = image_uri
    result["image_uri"] = image_uri
    runstate.record_evidence(run_id, "qc", result)
    return result


def run_scientific_segmentation(run_id: str, image_uri: str, attempt: int = 1) -> dict[str, Any]:
    """Run the validated real lesion model on the private scientific worker.

    attempt 1 = raw validated mask; attempt 2 = deterministic scientifically
    grounded recovery (same probability map + significance filter). The worker,
    model, checkpoint, and thresholds are fixed server-side.
    """
    url = _worker_url()
    if not url:
        return {"status": "failed", "reason": "worker_not_configured", "adapter": runstate.ADAPTER_NAME}
    if attempt not in (1, 2):
        return {"status": "failed", "reason": "invalid_attempt", "attempt": attempt}

    try:
        storage.validate_run_input(run_id, image_uri)
    except StorageError as exc:
        return {"status": "failed", "reason": exc.reason, "detail": exc.detail or None}

    runstate.set_state(run_id, "RETRYING" if attempt == 2 else "SEGMENTING", attempt=attempt)
    try:
        timeout_s = float(os.environ.get(WORKER_TIMEOUT_ENV, "").strip() or DEFAULT_WORKER_TIMEOUT_S)
    except ValueError:
        timeout_s = DEFAULT_WORKER_TIMEOUT_S
    try:
        resp = requests.post(
            f"{url}/segment",
            json={"run_id": run_id, "image_uri": image_uri, "attempt": attempt},
            headers=_auth_headers(url), timeout=timeout_s)
    except requests.RequestException as exc:
        return {"status": "failed", "reason": "worker_unreachable", "error": str(exc)}
    try:
        result = resp.json()
    except ValueError:
        return {"status": "failed", "reason": "malformed_worker_output", "http_status": resp.status_code}
    if resp.status_code != 200 or result.get("status") != "ok":
        result.setdefault("status", "failed")
        result.setdefault("reason", f"worker_http_{resp.status_code}")
        return result

    # No container paths leave the tool layer; the audit fetches by mask_uri.
    result.pop("mask_path", None)
    runstate.record_evidence(run_id, "segmentation", result, attempt=attempt)
    return result


def audit_topology(run_id: str, attempt: int = 1) -> dict[str, Any]:
    """Compute structural diagnostics (beta_0, beta_1, fragmentation) for this
    run's stored attempt mask. The mask location comes from the persisted
    segmentation evidence — never from the caller."""
    seg = runstate.get_evidence(run_id, "segmentation", attempt=attempt)
    if not seg or not seg.get("mask_uri"):
        return {"status": "failed", "reason": "missing_segmentation_evidence", "attempt": attempt}
    runstate.set_state(run_id, "TOPOLOGY_AUDIT")
    try:
        mask_path = _download_to_local(seg["mask_uri"])
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": "artifact_fetch_failed", "error": str(exc)}
    result = core.audit_topology(mask_path)
    runstate.record_evidence(run_id, "topology", result, attempt=attempt)
    return result


def bounded_policy(run_id: str, attempt: int) -> dict[str, Any]:
    """Deterministic bounded next action from the EXACT stored evidence for this run/attempt."""
    qc = runstate.get_evidence(run_id, "qc")
    seg = runstate.get_evidence(run_id, "segmentation", attempt=attempt)
    topo = runstate.get_evidence(run_id, "topology", attempt=attempt)
    decision = core.bounded_policy(qc or {}, seg, topo, attempt)
    runstate.record_evidence(run_id, "decision", decision, attempt=attempt)
    return decision


def create_report(run_id: str) -> dict[str, Any]:
    """Assemble and persist the run report from the exact stored evidence."""
    runstate.set_state(run_id, "REPORTING")
    run = runstate.get_run(run_id)
    if run is None:
        return {"status": "failed", "reason": "unknown_run_id"}
    qc = runstate.get_evidence(run_id, "qc") or {}

    runs: list[dict[str, Any]] = []
    final_decision: dict[str, Any] | None = None
    for attempt in (1, 2):
        seg = runstate.get_evidence(run_id, "segmentation", attempt=attempt)
        topo = runstate.get_evidence(run_id, "topology", attempt=attempt)
        decision = runstate.get_evidence(run_id, "decision", attempt=attempt)
        if seg is None and topo is None and decision is None:
            continue
        runs.append({"attempt": attempt, "segmentation": seg, "topology": topo, "decision": decision})
        if decision:
            final_decision = decision
    if final_decision is None:
        final_decision = runstate.get_evidence(run_id, "decision") or core.bounded_policy(qc, None, None, 1)
    if final_decision.get("action") == "RETRY":  # a final RETRY is never allowed
        final_decision = {"action": "HUMAN_REVIEW", "reason": "retry_limit_reached"}

    sample_id = run.get("sample_id") or run_id
    created = core.create_report(sample_id, qc, runs, final_decision)
    report = created["report"]
    try:
        from scientific_worker.storage import save_report
        report_uri = save_report(run_id, report)
    except Exception:  # noqa: BLE001 — local fallback already wrote report_path
        report_uri = created["report_path"]

    final_state = "HUMAN_REVIEW" if final_decision.get("action") == "HUMAN_REVIEW" else "COMPLETE"
    runstate.set_state(run_id, final_state, decision=final_decision)
    return {"status": "ok", "report_path": created["report_path"],
            "report_uri": report_uri, "report": report}


__all__ = [
    "inspect_image",
    "run_scientific_segmentation",
    "audit_topology",
    "bounded_policy",
    "create_report",
]
