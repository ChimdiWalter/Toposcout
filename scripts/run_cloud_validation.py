#!/usr/bin/env python3
"""FINAL CLOUD VALIDATION (M5): real leaves through the DEPLOYED worker.

Same deterministic tool sequence as scripts/run_cloud_vertical_slice.py, but
fully cloud-backed: gs:// inputs in the trusted bucket, the private Cloud Run
worker (id-token auth), Firestore run state, GCS artifacts. No Gemini calls.

Auth: production agent code uses SA identity tokens on Cloud Run; for this
operator-driven validation we mint the caller's identity token via
`gcloud auth print-identity-token` and patch cloud_tools' auth header only
(the deployed contract — private service, Bearer id-token — is unchanged).

Expected:
    DSC_0059  RETRY -> ACCEPT        DSC_0100  RETRY -> HUMAN_REVIEW

Usage:
    TOPOSCOUT_WORKER_URL=https://... python scripts/run_cloud_validation.py

Writes outputs/cloud_validation.json; exits nonzero on any mismatch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("TOPOSCOUT_GCS_BUCKET", "toposcout-agent-runs")
os.environ.setdefault("TOPOSCOUT_RUNSTATE", "firestore")
os.environ.setdefault("TOPOSCOUT_OUTPUT_DIR", "demo_outputs/cloud_validation/reports")
os.environ.pop("TOPOSCOUT_SW_ALLOW_LOCAL_IO", None)  # cloud mode: gs:// only

WORKER_URL = os.environ.get("TOPOSCOUT_WORKER_URL", "").rstrip("/")
if not WORKER_URL:
    raise SystemExit("set TOPOSCOUT_WORKER_URL to the deployed worker URL")

import requests  # noqa: E402

import app.cloud_tools as ct  # noqa: E402
from scientific_worker import storage  # noqa: E402
from toposcout_core import runstate  # noqa: E402

_TOKEN = subprocess.run(["gcloud", "auth", "print-identity-token"],
                        capture_output=True, text=True, check=True).stdout.strip()
ct._auth_headers = lambda audience: {"Authorization": f"Bearer {_TOKEN}"}  # operator auth

NORMALIZED = os.environ.get("TOPOSCOUT_DEMO_LEAVES_DIR", "")  # private demo leaves
STAMP = time.strftime("%Y%m%d-%H%M%S")
CASES = {
    f"cloudval-{STAMP}-DSC_0059": {
        "image": f"{NORMALIZED}/DSC_0059_segment_1_segmented_smoothed.png",
        "expect_actions": ["RETRY", "ACCEPT"], "expect_final": "ACCEPT",
        "expect_state": "COMPLETE",
    },
    f"cloudval-{STAMP}-DSC_0100": {
        "image": f"{NORMALIZED}/DSC_0100_segment_1_segmented_smoothed.png",
        "expect_actions": ["RETRY", "HUMAN_REVIEW"], "expect_final": "HUMAN_REVIEW",
        "expect_state": "HUMAN_REVIEW",
    },
}


def drive(run_id: str, image: str) -> dict:
    t0 = time.time()
    runstate.create_run(run_id, Path(image).stem)
    image_uri = storage.upload_input(run_id, image)
    print(f"  input -> {image_uri}", flush=True)

    qc = ct.inspect_image(run_id, image_uri)  # M6: QC fetches the canonical GCS input
    assert qc.get("qc_pass"), f"unexpected QC failure: {qc}"

    actions, attempts = [], []
    for attempt in (1, 2):
        t_a = time.time()
        seg = ct.run_scientific_segmentation(run_id, image_uri, attempt=attempt)
        if seg.get("status") != "ok":
            raise SystemExit(f"{run_id} attempt {attempt}: worker failed {seg}")
        topo = ct.audit_topology(run_id, attempt=attempt)
        decision = ct.bounded_policy(run_id, attempt=attempt)
        actions.append(decision["action"])
        attempts.append({
            "attempt": attempt, "mask_uri": seg["mask_uri"],
            "worker_seconds": seg["runtime_seconds"],
            "roundtrip_seconds": round(time.time() - t_a, 2),
            "prob_reused": seg["prob_reused"], "min_area_px": seg["min_area_px"],
            "beta_0": topo["beta_0"],
            "fragmentation_score": topo["fragmentation_score"],
            "foreground_fraction": seg["foreground_fraction"],
            "checkpoint_sha256": seg["checkpoint_sha256"],
            "action": decision["action"],
        })
        print(f"  attempt {attempt}: min_area={seg['min_area_px']} "
              f"prob_reused={seg['prob_reused']} beta_0={topo['beta_0']} "
              f"frag={topo['fragmentation_score']:.3f} -> {decision['action']} "
              f"({attempts[-1]['roundtrip_seconds']}s)", flush=True)
        if decision["action"] != "RETRY":
            break

    report = ct.create_report(run_id)
    assert report.get("status") == "ok", report
    doc = runstate.get_run(run_id)
    print(f"  FINAL: {doc['decision']['action']} (state={doc['state']}, "
          f"report={report['report_uri']}, total {time.time() - t0:.1f}s)", flush=True)
    return {"actions": actions, "final": doc["decision"]["action"], "state": doc["state"],
            "attempts": attempts, "report_uri": report["report_uri"],
            "firestore_doc": doc}


def main() -> None:
    r = requests.get(f"{WORKER_URL}/health",
                     headers={"Authorization": f"Bearer {_TOKEN}"}, timeout=30)
    r.raise_for_status()
    print("worker healthy:", json.dumps(r.json()), flush=True)

    results: dict = {"worker_url": WORKER_URL, "worker_health": r.json()}
    ok = True
    for run_id, case in CASES.items():
        print(f"\n=== {run_id} (expect {' -> '.join(case['expect_actions'])}) ===", flush=True)
        out = drive(run_id, case["image"])
        out["pass"] = (out["actions"] == case["expect_actions"]
                       and out["final"] == case["expect_final"]
                       and out["state"] == case["expect_state"])
        ok = ok and out["pass"]
        results[run_id] = out

    results["verdict"] = "PASS" if ok else "FAIL"
    out_path = Path("outputs/cloud_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nCLOUD VALIDATION: {results['verdict']} -> {out_path}", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
