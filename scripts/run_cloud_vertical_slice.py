#!/usr/bin/env python3
"""Local agent→worker vertical slice with the REAL model (M5 Phase 9 gate).

Drives app.cloud_tools (the exact tools the cloud agent uses) against a
locally running scientific worker over real HTTP, in local dev mode:

    TOPOSCOUT_SW_ALLOW_LOCAL_IO=1 uvicorn scientific_worker.app:app --port 8081

No Gemini calls — the tool sequence is executed deterministically, mirroring
the agent instruction (QC → segment 1 → audit → policy → [segment 2 → audit →
policy] → report). Expected on the two held-out gold leaves (read in place):

    DSC_0059  RETRY -> ACCEPT
    DSC_0100  RETRY -> HUMAN_REVIEW

Writes outputs/cloud_vertical_slice.json and exits nonzero on any mismatch.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORKER_URL = os.environ.setdefault("TOPOSCOUT_WORKER_URL", "http://127.0.0.1:8081")
os.environ.setdefault("TOPOSCOUT_WORKER_AUTH", "none")
os.environ.setdefault("TOPOSCOUT_RUNSTATE", "local")
os.environ.setdefault("TOPOSCOUT_RUNSTATE_DIR", "demo_outputs/vertical_slice/runstate")
os.environ.setdefault("TOPOSCOUT_SW_ALLOW_LOCAL_IO", "1")
os.environ.setdefault("TOPOSCOUT_SW_LOCAL_ARTIFACTS", "demo_outputs/vertical_slice/artifacts")
os.environ.setdefault("TOPOSCOUT_OUTPUT_DIR", "demo_outputs/vertical_slice/reports")

import requests  # noqa: E402

import app.cloud_tools as ct  # noqa: E402
from scientific_worker import storage  # noqa: E402
from toposcout_core import runstate  # noqa: E402

NORMALIZED = os.environ.get("TOPOSCOUT_DEMO_LEAVES_DIR", "")  # private demo leaves
CASES = {
    "DSC_0059_segment_1_segmented_smoothed": {
        "image": f"{NORMALIZED}/DSC_0059_segment_1_segmented_smoothed.png",
        "expect_actions": ["RETRY", "ACCEPT"],
        "expect_final": "ACCEPT",
    },
    "DSC_0100_segment_1_segmented_smoothed": {
        "image": f"{NORMALIZED}/DSC_0100_segment_1_segmented_smoothed.png",
        "expect_actions": ["RETRY", "HUMAN_REVIEW"],
        "expect_final": "HUMAN_REVIEW",
    },
}


def wait_for_health(timeout_s: float = 60.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{WORKER_URL}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise SystemExit(f"worker at {WORKER_URL} not healthy within {timeout_s}s")


def drive_run(run_id: str, image: str) -> dict:
    t0 = time.time()
    image_uri = storage.upload_input(run_id, image)  # canonical staging (M6 contract)
    qc = ct.inspect_image(run_id, image_uri)
    if not qc.get("qc_pass"):
        raise SystemExit(f"{run_id}: unexpected QC failure {qc}")

    actions: list[str] = []
    attempts: list[dict] = []
    for attempt in (1, 2):
        t_seg = time.time()
        seg = ct.run_scientific_segmentation(run_id, image_uri, attempt=attempt)
        if seg.get("status") != "ok":
            raise SystemExit(f"{run_id} attempt {attempt}: worker failed {seg}")
        topo = ct.audit_topology(run_id, attempt=attempt)
        decision = ct.bounded_policy(run_id, attempt=attempt)
        actions.append(decision["action"])
        attempts.append({
            "attempt": attempt,
            "worker_seconds": seg["runtime_seconds"],
            "roundtrip_seconds": round(time.time() - t_seg, 2),
            "prob_reused": seg["prob_reused"],
            "min_area_px": seg["min_area_px"],
            "n_components_significant": seg["n_components_significant"],
            "foreground_fraction": seg["foreground_fraction"],
            "beta_0": topo["beta_0"],
            "fragmentation_score": topo["fragmentation_score"],
            "action": decision["action"],
        })
        print(f"  attempt {attempt}: min_area={seg['min_area_px']} "
              f"prob_reused={seg['prob_reused']} beta_0={topo['beta_0']} "
              f"frag={topo['fragmentation_score']:.3f} -> {decision['action']}", flush=True)
        if decision["action"] != "RETRY":
            break

    report = ct.create_report(run_id)
    if report.get("status") != "ok":
        raise SystemExit(f"{run_id}: create_report failed {report}")
    final = report["report"]["final_decision"]["action"]
    state = runstate.get_run(run_id)["state"]
    print(f"  FINAL: {final} (state={state}, total {time.time() - t0:.1f}s)", flush=True)
    return {"actions": actions, "final": final, "state": state,
            "attempts": attempts, "report_uri": report["report_uri"],
            "checkpoint_sha256": attempts[0].get("checkpoint_sha256")}


def main() -> None:
    health = wait_for_health()
    print(f"worker healthy: {json.dumps(health)}", flush=True)

    results: dict[str, dict] = {"worker_health": health}
    ok = True
    for run_id, case in CASES.items():
        print(f"\n=== {run_id} (expect {' -> '.join(case['expect_actions'])}) ===", flush=True)
        out = drive_run(run_id, case["image"])
        out["expected_actions"] = case["expect_actions"]
        out["expected_final"] = case["expect_final"]
        out["pass"] = (out["actions"] == case["expect_actions"]
                       and out["final"] == case["expect_final"])
        ok = ok and out["pass"]
        results[run_id] = out

    results["verdict"] = "PASS" if ok else "FAIL"
    out_path = Path("outputs/cloud_vertical_slice.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nVERTICAL SLICE: {results['verdict']} -> {out_path}", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
