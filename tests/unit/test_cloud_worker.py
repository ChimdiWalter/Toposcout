"""Milestone 5 Phase 8 tests: scientific worker API, storage naming, run-state
evidence preservation, and the agent→worker contract.

All hermetic: no Gemini, no real model (the inference engine is stubbed), no
GCS/Firestore (local backends). The real-model paths are covered by the
equivalence gates (scripts/check_*_equivalence.py) and the opt-in smoke test.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import scientific_worker.app as worker_app
from scientific_worker.model_loader import ADAPTER_NAME, MODEL_VERSION
from scientific_worker.schemas import SegmentRequest
from toposcout_core import fixtures, runstate

FAKE_SHA = "f" * 64


# ---------- shared fixtures ----------

@pytest.fixture()
def local_cloud_env(monkeypatch, tmp_path: Path) -> Path:
    """Local dev mode: no bucket, local artifacts + runstate under tmp."""
    monkeypatch.delenv("TOPOSCOUT_GCS_BUCKET", raising=False)
    monkeypatch.setenv("TOPOSCOUT_SW_ALLOW_LOCAL_IO", "1")
    monkeypatch.setenv("TOPOSCOUT_SW_LOCAL_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.delenv("TOPOSCOUT_RUNSTATE", raising=False)  # local backend
    monkeypatch.setenv("TOPOSCOUT_RUNSTATE_DIR", str(tmp_path / "runstate"))
    monkeypatch.setenv("TOPOSCOUT_OUTPUT_DIR", str(tmp_path / "outputs"))
    return tmp_path


class StubEngine:
    """Deterministic stand-in for InferenceEngine with frozen attempt semantics."""

    def __init__(self):
        self.calls: list[dict] = []

    def segment(self, img_bgr, attempt, cached_prob=None):
        assert attempt in (1, 2)
        h, w = img_bgr.shape[:2]
        prob_reused = cached_prob is not None and cached_prob.shape == (h, w)
        prob = cached_prob if prob_reused else np.full((h, w), 0.75, np.float32)
        self.calls.append({"attempt": attempt, "cached": prob_reused})
        mask = prob > 0.5
        min_area = 500 if attempt >= 2 else 0
        return {
            "status": "ok", "adapter": ADAPTER_NAME, "attempt": attempt,
            "prob": prob, "mask": mask,
            "overlay": np.zeros((h, w, 3), np.uint8),
            "prob_reused": bool(prob_reused),
            "foreground_fraction": float(mask.mean()),
            "prob_threshold": 0.5, "min_area_px": min_area,
            "n_components_significant": 1,
            "image_height": h, "image_width": w,
            "model_version": MODEL_VERSION, "checkpoint_sha256": FAKE_SHA,
            "runtime_seconds": 0.01,
        }


@pytest.fixture()
def worker(monkeypatch, local_cloud_env: Path):
    """TestClient over the real FastAPI app with a stubbed engine."""
    stub = StubEngine()
    monkeypatch.setattr(worker_app, "_engine", lambda: stub)
    client = TestClient(worker_app.app)
    return client, stub, local_cloud_env


@pytest.fixture()
def demo_image(tmp_path: Path) -> Path:
    p = tmp_path / "leaf.png"
    Image.fromarray(np.full((96, 128, 3), 120, np.uint8)).save(p)
    return p


# ---------- worker /health ----------

def test_health_reports_identity_without_loading_model(worker):
    client, stub, _ = worker
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["adapter"] == ADAPTER_NAME
    assert body["model_version"] == MODEL_VERSION
    assert body["model_loaded"] is False  # health must not trigger a load
    assert stub.calls == []


# ---------- request surface is closed ----------

@pytest.mark.parametrize("extra", [
    {"checkpoint_path": "/tmp/evil.pth"},
    {"checkpoint": "other"},
    {"prob_threshold": 0.1},
    {"threshold": 0.9},
    {"min_area_px": 0},
    {"executable": "/bin/sh"},
    {"command": "rm -rf /"},
    {"python": "/usr/bin/python"},
    {"architecture": "unet"},
    {"output_dir": "/etc"},
])
def test_request_cannot_choose_model_threshold_or_executable(worker, demo_image, extra):
    client, stub, _ = worker
    payload = {"run_id": "r1", "image_uri": str(demo_image), "attempt": 1, **extra}
    resp = client.post("/segment", json=payload)
    assert resp.status_code == 422  # closed schema: extra="forbid"
    assert stub.calls == []


def test_schema_rejects_extra_fields_directly():
    with pytest.raises(Exception):
        SegmentRequest(run_id="r1", image_uri="x.png", attempt=1, checkpoint="c")


@pytest.mark.parametrize("attempt", [0, 3, -1])
def test_invalid_attempt_rejected(worker, demo_image, attempt):
    client, stub, _ = worker
    resp = client.post("/segment", json={"run_id": "r1", "image_uri": str(demo_image), "attempt": attempt})
    assert resp.status_code == 422
    assert stub.calls == []


def test_invalid_run_id_rejected(worker, demo_image):
    client, _, _ = worker
    resp = client.post("/segment", json={"run_id": "../escape", "image_uri": str(demo_image), "attempt": 1})
    assert resp.status_code == 422


def test_missing_input_is_404(worker, tmp_path):
    client, _, _ = worker
    resp = client.post("/segment", json={"run_id": "r1", "image_uri": str(tmp_path / "nope.png"), "attempt": 1})
    assert resp.status_code == 404
    assert resp.json()["reason"] == "input_not_found"


def test_local_paths_rejected_outside_dev_mode(monkeypatch, demo_image):
    monkeypatch.delenv("TOPOSCOUT_GCS_BUCKET", raising=False)
    monkeypatch.delenv("TOPOSCOUT_SW_ALLOW_LOCAL_IO", raising=False)
    client = TestClient(worker_app.app)
    resp = client.post("/segment", json={"run_id": "r1", "image_uri": str(demo_image), "attempt": 1})
    assert resp.status_code == 400
    assert resp.json()["reason"] == "local_paths_disabled"


def test_untrusted_bucket_uri_rejected(monkeypatch):
    monkeypatch.setenv("TOPOSCOUT_GCS_BUCKET", "toposcout-trusted")
    monkeypatch.delenv("TOPOSCOUT_SW_ALLOW_LOCAL_IO", raising=False)
    client = TestClient(worker_app.app)
    resp = client.post("/segment", json={"run_id": "r1", "image_uri": "gs://other-bucket/x.png", "attempt": 1})
    assert resp.status_code == 400
    assert resp.json()["reason"] == "untrusted_image_uri"


# ---------- attempt semantics + GCS artifact naming (local mirror) ----------

def test_attempt1_behavior_and_artifact_naming(worker, demo_image):
    client, stub, tmp = worker
    resp = client.post("/segment", json={"run_id": "run-A", "image_uri": str(demo_image), "attempt": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["adapter"] == ADAPTER_NAME
    assert body["attempt"] == 1
    assert body["min_area_px"] == 0
    assert body["prob_threshold"] == 0.5
    assert body["prob_reused"] is False
    assert body["checkpoint_sha256"] == FAKE_SHA
    # canonical layout runs/<run_id>/attempt1/...
    root = tmp / "artifacts" / "runs" / "run-A" / "attempt1"
    assert (root / "mask.png").is_file()
    assert (root / "overlay.jpg").is_file()
    assert (root / "prob.npy").is_file()      # prob map cached on attempt 1
    assert (root / "prob.json").is_file()
    assert body["mask_uri"].endswith("runs/run-A/attempt1/mask.png")
    assert body["overlay_uri"].endswith("runs/run-A/attempt1/overlay.jpg")


def test_attempt2_reuses_exact_prob_and_applies_filter(worker, demo_image):
    client, stub, tmp = worker
    r1 = client.post("/segment", json={"run_id": "run-B", "image_uri": str(demo_image), "attempt": 1})
    r2 = client.post("/segment", json={"run_id": "run-B", "image_uri": str(demo_image), "attempt": 2})
    assert r2.status_code == 200
    body = r2.json()
    assert body["min_area_px"] == 500
    assert body["prob_reused"] is True                 # SAME probability map
    assert stub.calls[1]["cached"] is True
    root = tmp / "artifacts" / "runs" / "run-B" / "attempt2"
    assert (root / "mask.png").is_file()
    assert not (root / "prob.npy").exists()            # cached only on attempt 1
    assert r1.json()["checkpoint_sha256"] == body["checkpoint_sha256"]


# ---------- run-state: exact evidence preservation ----------

TOPO_EXACT = {
    "status": "ok", "beta_0": 65, "beta_1": 2, "tiny_components": 55,
    "fragmentation_score": 0.8461538461538461,
    "largest_component_fraction": 0.493804068272153,
    "foreground_pixels": 8554,
}


def test_runstate_preserves_exact_evidence(local_cloud_env):
    runstate.create_run("run-C", "DSC_0059")
    runstate.record_evidence("run-C", "topology", TOPO_EXACT, attempt=1)
    assert runstate.get_evidence("run-C", "topology", attempt=1) == TOPO_EXACT
    # verbatim on disk too (no float mangling through the round-trip)
    raw = json.loads((local_cloud_env / "runstate" / "run-C.json").read_text())
    assert raw["evidence"]["topology_a1"]["payload"] == TOPO_EXACT


def test_runstate_state_machine_is_closed(local_cloud_env):
    runstate.create_run("run-D", "s")
    doc = runstate.get_run("run-D")
    assert doc["state"] == "RECEIVED"
    assert doc["adapter"] == ADAPTER_NAME
    assert "evidence" not in doc
    runstate.set_state("run-D", "SEGMENTING", attempt=1)
    assert runstate.get_run("run-D")["state"] == "SEGMENTING"
    with pytest.raises(ValueError):
        runstate.set_state("run-D", "NOT_A_STATE")
    with pytest.raises(ValueError):
        runstate.record_evidence("run-D", "vibes", {})
    with pytest.raises(ValueError):
        runstate.create_run("../evil", "s")
    with pytest.raises(KeyError):
        runstate.set_state("run-unknown", "QC")


# ---------- agent → worker contract ----------

@pytest.fixture()
def cloud_agent(monkeypatch, worker):
    """cloud_tools wired to the real worker app through the TestClient."""
    client, stub, tmp = worker
    import app.cloud_tools as ct

    class _Resp:
        def __init__(self, r): self._r = r
        @property
        def status_code(self): return self._r.status_code
        def json(self): return self._r.json()

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url.endswith("/segment")
        assert set(json) == {"run_id", "image_uri", "attempt"}  # closed contract
        return _Resp(client.post("/segment", json=json))

    monkeypatch.setenv("TOPOSCOUT_WORKER_URL", "http://worker.test")
    monkeypatch.setenv("TOPOSCOUT_WORKER_AUTH", "none")
    monkeypatch.setattr(ct.requests, "post", fake_post)
    return ct, stub, tmp


def test_cloud_tools_expose_no_scientific_choices():
    import app.cloud_tools as ct
    params = inspect.signature(ct.run_scientific_segmentation).parameters
    assert set(params) == {"run_id", "image_uri", "attempt"}
    for forbidden in ("checkpoint", "threshold", "model", "executable", "output_dir", "bucket"):
        assert all(forbidden not in p for p in params)
    # M6: no local paths anywhere in the deployed tool schemas
    assert set(inspect.signature(ct.inspect_image).parameters) == {"run_id", "image_uri"}
    assert set(inspect.signature(ct.audit_topology).parameters) == {"run_id", "attempt"}
    assert set(inspect.signature(ct.bounded_policy).parameters) == {"run_id", "attempt"}
    assert set(inspect.signature(ct.create_report).parameters) == {"run_id"}
    for fn in (ct.inspect_image, ct.audit_topology, ct.bounded_policy, ct.create_report):
        for p in inspect.signature(fn).parameters:
            assert "path" not in p


# ---------- M6: cloud-native canonical-input QC ----------

@pytest.fixture()
def staged_input(local_cloud_env, tmp_path):
    from scientific_worker import storage
    image = fixtures.write_fragmented_recoverable_image(tmp_path / "in" / "leafy.png")
    uri = storage.upload_input("run-QC", image)
    return uri


def test_cloud_qc_accepts_canonical_trusted_input(staged_input):
    import app.cloud_tools as ct
    qc = ct.inspect_image("run-QC", staged_input)
    assert qc["status"] == "ok" and qc["qc_pass"] is True
    assert qc["image_path"] == staged_input  # canonical URI, no /tmp path
    # exact evidence persisted verbatim
    assert runstate.get_evidence("run-QC", "qc") == qc
    assert runstate.get_run("run-QC")["state"] == "QC"


def test_cloud_qc_rejects_other_bucket(monkeypatch, local_cloud_env):
    import app.cloud_tools as ct
    monkeypatch.setenv("TOPOSCOUT_GCS_BUCKET", "toposcout-trusted")
    monkeypatch.delenv("TOPOSCOUT_SW_ALLOW_LOCAL_IO", raising=False)
    out = ct.inspect_image("run-QC", "gs://other-bucket/runs/run-QC/input/x.png")
    assert out == {"status": "failed", "reason": "untrusted_image_uri",
                   "detail": out["detail"]}
    assert "toposcout-trusted" in out["detail"]


def test_cloud_qc_rejects_other_runs_prefix(monkeypatch, local_cloud_env, staged_input):
    import app.cloud_tools as ct
    # canonical object of run-QC presented under a different run_id
    out = ct.inspect_image("run-OTHER", staged_input)
    assert out["status"] == "failed"
    assert out["reason"] == "untrusted_image_uri"
    # gs:// variant: run_id mismatch inside a trusted bucket
    monkeypatch.setenv("TOPOSCOUT_GCS_BUCKET", "toposcout-trusted")
    out = ct.inspect_image("run-A", "gs://toposcout-trusted/runs/run-B/input/x.png")
    assert out["reason"] == "untrusted_image_uri"


def test_cloud_qc_rejects_nested_or_bad_suffix(monkeypatch, local_cloud_env):
    import app.cloud_tools as ct
    monkeypatch.setenv("TOPOSCOUT_GCS_BUCKET", "tb")
    assert ct.inspect_image("r1", "gs://tb/runs/r1/input/a/b.png")["reason"] == "untrusted_image_uri"
    assert ct.inspect_image("r1", "gs://tb/runs/r1/input/x.exe")["reason"] == "unsupported_image_type"
    assert ct.inspect_image("bad/../id", "gs://tb/runs/bad/../id/input/x.png")["reason"] == "invalid_run_id"


def test_segmentation_rejects_untrusted_uri_before_worker(monkeypatch, local_cloud_env):
    import app.cloud_tools as ct
    monkeypatch.setenv("TOPOSCOUT_WORKER_URL", "http://worker.test")
    monkeypatch.setenv("TOPOSCOUT_GCS_BUCKET", "tb")
    monkeypatch.delenv("TOPOSCOUT_SW_ALLOW_LOCAL_IO", raising=False)
    called = []
    monkeypatch.setattr(ct.requests, "post", lambda *a, **k: called.append(1))
    runstate.create_run("run-U", "s")
    out = ct.run_scientific_segmentation("run-U", "gs://evil/runs/run-U/input/x.png", attempt=1)
    assert out["reason"] == "untrusted_image_uri"
    assert called == []  # never reached the worker


def test_agent_worker_vertical_contract(cloud_agent, tmp_path):
    """QC → worker segment → topology → policy → report, all evidence exact."""
    ct, stub, _ = cloud_agent
    from scientific_worker import storage
    image = fixtures.write_fragmented_recoverable_image(tmp_path / "in" / "DSC_test.png")
    image_uri = storage.upload_input("run-E", image)  # canonical staging

    qc = ct.inspect_image("run-E", image_uri)
    assert qc["qc_pass"] is True

    seg = ct.run_scientific_segmentation("run-E", image_uri, attempt=1)
    assert seg["status"] == "ok"
    assert seg["adapter"] == ADAPTER_NAME
    assert "mask_path" not in seg  # no container paths in evidence
    assert Path(seg["mask_uri"]).is_file()  # local dev mode: URI is the artifact path

    topo = ct.audit_topology("run-E", attempt=1)
    assert topo["status"] == "ok"

    decision = ct.bounded_policy("run-E", attempt=1)
    assert decision["action"] in {"ACCEPT", "RETRY", "HUMAN_REVIEW", "REQUEST_REACQUISITION"}

    report = ct.create_report("run-E")
    assert report["status"] == "ok"
    body = report["report"]
    # exact stored evidence, not conversation copies
    assert body["qc"] == qc
    assert body["runs"][0]["segmentation"] == seg
    assert body["runs"][0]["topology"] == topo
    assert body["runs"][0]["decision"] == decision
    assert report["report_uri"].endswith("runs/run-E/report/report.json")
    assert runstate.get_run("run-E")["state"] in {"COMPLETE", "HUMAN_REVIEW"}


def test_agent_policy_uses_stored_not_narrated_evidence(cloud_agent, tmp_path):
    """bounded_policy must read runstate; fragmented stored topology → RETRY."""
    ct, _, _ = cloud_agent
    runstate.create_run("run-F", "s")
    runstate.record_evidence("run-F", "qc", {"status": "ok", "qc_pass": True})
    runstate.record_evidence("run-F", "segmentation", {"foreground_fraction": 0.3}, attempt=1)
    runstate.record_evidence("run-F", "topology", TOPO_EXACT, attempt=1)  # frag 0.846
    assert ct.bounded_policy("run-F", attempt=1)["action"] == "RETRY"
    # same anomaly at the retry limit escalates, never loops
    runstate.record_evidence("run-F", "segmentation", {"foreground_fraction": 0.3}, attempt=2)
    runstate.record_evidence("run-F", "topology", TOPO_EXACT, attempt=2)
    assert ct.bounded_policy("run-F", attempt=2)["action"] == "HUMAN_REVIEW"


def test_final_retry_is_never_reported(cloud_agent, tmp_path):
    ct, _, _ = cloud_agent
    runstate.create_run("run-G", "s")
    runstate.record_evidence("run-G", "qc", {"status": "ok", "qc_pass": True})
    runstate.record_evidence("run-G", "decision", {"action": "RETRY", "reason": "x"}, attempt=1)
    report = ct.create_report("run-G")
    assert report["report"]["final_decision"]["action"] == "HUMAN_REVIEW"
    assert report["report"]["final_decision"]["reason"] == "retry_limit_reached"


def test_worker_not_configured_fails_closed(monkeypatch, local_cloud_env):
    import app.cloud_tools as ct
    monkeypatch.delenv("TOPOSCOUT_WORKER_URL", raising=False)
    runstate.create_run("run-H", "s")
    out = ct.run_scientific_segmentation("run-H", "x.png", attempt=1)
    assert out == {"status": "failed", "reason": "worker_not_configured", "adapter": ADAPTER_NAME}
    monkeypatch.setenv("TOPOSCOUT_WORKER_URL", "http://worker.test")
    assert ct.run_scientific_segmentation("run-H", "x.png", attempt=3)["reason"] == "invalid_attempt"
