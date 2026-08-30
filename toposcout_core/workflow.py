from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import run_segmentation
from .tools import audit_topology, bounded_policy, create_report, inspect_image


def run_local_workflow(image_path: str, output_dir: str | None = None) -> dict[str, Any]:
    """Reliable local vertical slice used for tests and fallback demos.

    output_dir=None resolves the canonical artifact directory (TOPOSCOUT_OUTPUT_DIR).
    """
    sample_id = Path(image_path).stem
    qc = inspect_image(image_path)
    if qc.get("status") != "ok" or not qc.get("qc_pass", False):
        decision = bounded_policy(qc, None, None, 1)
        return create_report(sample_id, qc, [], decision, output_dir)["report"]

    runs: list[dict[str, Any]] = []
    for attempt in (1, 2):
        segmentation = run_segmentation(image_path, output_dir, attempt=attempt)
        if segmentation.get("status") != "ok":
            runs.append({"attempt": attempt, "segmentation": segmentation, "topology": None,
                         "decision": {"action": "HUMAN_REVIEW", "reason": "segmentation_failed"}})
            decision = {"action": "HUMAN_REVIEW", "reason": "segmentation_failed"}
            return create_report(sample_id, qc, runs, decision, output_dir)["report"]
        topology = audit_topology(segmentation["mask_path"])
        decision = bounded_policy(qc, segmentation, topology, attempt)
        runs.append({
            "attempt": attempt,
            "segmentation": segmentation,
            "topology": topology,
            "decision": decision,
        })
        if decision["action"] != "RETRY":
            return create_report(sample_id, qc, runs, decision, output_dir)["report"]

    # Defensive fallback; the loop should always return on attempt 2.
    decision = {"action": "HUMAN_REVIEW", "reason": "retry_limit_reached"}
    return create_report(sample_id, qc, runs, decision, output_dir)["report"]
