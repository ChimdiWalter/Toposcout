"""Run-state + exact-evidence persistence (Milestone 5 Phase 4).

Replaces the in-process evidence cache for cloud execution. Two backends
behind one API, selected by trusted configuration only:

    TOPOSCOUT_RUNSTATE=firestore   Google Cloud Firestore (collection
                                   ``toposcout_runs``, evidence subcollection)
    TOPOSCOUT_RUNSTATE=local       JSON files under TOPOSCOUT_RUNSTATE_DIR
                                   (default demo_outputs/runstate) — hermetic
                                   tests and local dev. This is the default.

Design rules:
- Evidence payloads are the EXACT deterministic tool outputs, stored verbatim
  and restored verbatim; Gemini never reconstructs scientific values from
  conversation history (create_report reads back from here).
- States are a closed set; invalid names raise. Transitions are recorded with
  timestamps (created_at/updated_at, ISO-8601 UTC).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ADAPTER_NAME = "real_lesion_model_v1"

STATES = (
    "RECEIVED", "QC", "SEGMENTING", "TOPOLOGY_AUDIT", "RETRYING",
    "REPORTING", "COMPLETE", "HUMAN_REVIEW",
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

BACKEND_ENV = "TOPOSCOUT_RUNSTATE"
DIR_ENV = "TOPOSCOUT_RUNSTATE_DIR"
DEFAULT_DIR = "demo_outputs/runstate"
FIRESTORE_COLLECTION = "toposcout_runs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return run_id


def _check_state(state: str) -> str:
    if state not in STATES:
        raise ValueError(f"invalid state {state!r}; allowed: {STATES}")
    return state


def _backend() -> str:
    return os.environ.get(BACKEND_ENV, "").strip() or "local"


# ── local JSON backend ───────────────────────────────────────────────────────

def _local_dir() -> Path:
    return Path(os.environ.get(DIR_ENV, "").strip() or DEFAULT_DIR)


def _local_run_path(run_id: str) -> Path:
    return _local_dir() / f"{run_id}.json"


def _local_read(run_id: str) -> dict[str, Any] | None:
    p = _local_run_path(run_id)
    return json.loads(p.read_text()) if p.is_file() else None


def _local_write(run_id: str, doc: dict[str, Any]) -> None:
    p = _local_run_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2))


# ── Firestore backend ────────────────────────────────────────────────────────

def _fs():
    from google.cloud import firestore
    return firestore.Client()


def _fs_doc(run_id: str):
    return _fs().collection(FIRESTORE_COLLECTION).document(run_id)


# ── public API ───────────────────────────────────────────────────────────────

def create_run(run_id: str, sample_id: str, image_uri: str | None = None) -> dict[str, Any]:
    _check_run_id(run_id)
    doc = {
        "run_id": run_id,
        "sample_id": sample_id,
        "state": "RECEIVED",
        "adapter": ADAPTER_NAME,
        "attempt": 0,
        "decision": None,
        "image_uri": image_uri,
        "created_at": _now(),
        "updated_at": _now(),
    }
    if _backend() == "firestore":
        _fs_doc(run_id).set(doc)
    else:
        existing = _local_read(run_id) or {}
        existing.update(doc)
        existing.setdefault("evidence", {})
        _local_write(run_id, existing)
    return doc


def set_state(run_id: str, state: str, *, attempt: int | None = None,
              decision: dict[str, Any] | None = None) -> dict[str, Any]:
    _check_run_id(run_id)
    _check_state(state)
    patch: dict[str, Any] = {"state": state, "updated_at": _now()}
    if attempt is not None:
        patch["attempt"] = int(attempt)
    if decision is not None:
        patch["decision"] = decision
    if _backend() == "firestore":
        _fs_doc(run_id).update(patch)
        return patch
    doc = _local_read(run_id)
    if doc is None:
        raise KeyError(f"unknown run_id {run_id!r}")
    doc.update(patch)
    _local_write(run_id, doc)
    return patch


def record_evidence(run_id: str, kind: str, payload: dict[str, Any],
                    attempt: int | None = None) -> str:
    """Persist one exact tool output. kind in {qc, segmentation, topology, decision}."""
    _check_run_id(run_id)
    if kind not in {"qc", "segmentation", "topology", "decision"}:
        raise ValueError(f"invalid evidence kind {kind!r}")
    key = f"{kind}_a{attempt}" if attempt is not None else kind
    entry = {"kind": kind, "attempt": attempt, "recorded_at": _now(), "payload": payload}
    if _backend() == "firestore":
        _fs_doc(run_id).collection("evidence").document(key).set(entry)
        return key
    doc = _local_read(run_id)
    if doc is None:
        raise KeyError(f"unknown run_id {run_id!r}")
    doc.setdefault("evidence", {})[key] = entry
    doc["updated_at"] = _now()
    _local_write(run_id, doc)
    return key


def get_run(run_id: str) -> dict[str, Any] | None:
    _check_run_id(run_id)
    if _backend() == "firestore":
        snap = _fs_doc(run_id).get()
        return snap.to_dict() if snap.exists else None
    doc = _local_read(run_id)
    if doc is not None:
        doc = {k: v for k, v in doc.items() if k != "evidence"}
    return doc


def get_evidence(run_id: str, kind: str, attempt: int | None = None) -> dict[str, Any] | None:
    """Return the exact stored payload (verbatim floats) or None."""
    _check_run_id(run_id)
    key = f"{kind}_a{attempt}" if attempt is not None else kind
    if _backend() == "firestore":
        snap = _fs_doc(run_id).collection("evidence").document(key).get()
        return snap.to_dict().get("payload") if snap.exists else None
    doc = _local_read(run_id)
    entry = (doc or {}).get("evidence", {}).get(key)
    return entry.get("payload") if entry else None
