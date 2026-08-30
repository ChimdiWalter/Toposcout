"""ADK-facing tool wrappers.

ADK tools are plain Python functions; their signatures become the LLM-facing
schemas. These wrappers deliberately expose NO filesystem artifact choices:
the canonical output directory is resolved in Python (TOPOSCOUT_OUTPUT_DIR,
default "outputs") and the report sample_id is derived from the QC evidence.
The language model decides which approved tool to call next, never where
artifacts are written or what the measurements are.

Measurement tools additionally record their exact outputs in an in-process
evidence cache. create_report restores the cached values, so the canonical
report keeps full float precision even if numbers were truncated while
round-tripping through the model transport (observed in the first agentic
trace: 0.4938040682721534 arrived back as 0.493804068272153).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from toposcout_core import adapters, tools as core

_EVIDENCE: dict[str, dict[str, dict[str, Any]]] = {"qc": {}, "seg": {}, "topo": {}}


def inspect_image(image_path: str) -> dict[str, Any]:
    """Inspect a scientific image for basic acquisition and quality problems."""
    result = core.inspect_image(image_path)
    _EVIDENCE["qc"][result.get("image_path", image_path)] = result
    return result


def run_segmentation(image_path: str, attempt: int = 1) -> dict[str, Any]:
    """Run the configured deterministic segmentation adapter (attempt 1 or 2).

    Which adapter runs (hackathon demo adapter or the validated real lesion
    model) is fixed by trusted Python configuration, never by the caller.
    Masks and overlays are saved automatically to the canonical artifact
    directory; the returned mask_path reports where the mask was written.
    """
    result = adapters.run_segmentation(image_path, attempt=attempt)
    if result.get("status") == "ok" and result.get("mask_path"):
        _EVIDENCE["seg"][result["mask_path"]] = result
    return result


def audit_topology(mask_path: str) -> dict[str, Any]:
    """Compute structural diagnostics (beta_0, beta_1, fragmentation) from a binary mask."""
    result = core.audit_topology(mask_path)
    _EVIDENCE["topo"][mask_path] = result
    return result


def bounded_policy(qc: dict[str, Any], segmentation: dict[str, Any] | None, topology: dict[str, Any] | None, attempt: int) -> dict[str, Any]:
    """Return the deterministic bounded next action for the exact tool evidence provided."""
    return core.bounded_policy(qc, segmentation, topology, attempt)


def create_report(qc: dict[str, Any], runs: list[dict[str, Any]], final_decision: dict[str, Any]) -> dict[str, Any]:
    """Persist the experiment evidence report to the canonical artifact directory.

    Pass the exact, unmodified qc/segmentation/topology tool outputs. The
    report includes a deterministic display_summary to relay verbatim.
    """
    action = final_decision.get("action") if isinstance(final_decision, dict) else None
    if action not in core.ACTIONS:
        return {
            "status": "failed",
            "reason": "invalid_final_action",
            "allowed_actions": sorted(core.ACTIONS),
        }

    exact_qc = _EVIDENCE["qc"].get(str(qc.get("image_path", "")), qc)
    exact_runs: list[dict[str, Any]] = []
    for run in runs:
        run = dict(run)
        seg = run.get("segmentation") or {}
        mask_path = str(seg.get("mask_path", ""))
        if mask_path in _EVIDENCE["seg"]:
            run["segmentation"] = _EVIDENCE["seg"][mask_path]
        if mask_path in _EVIDENCE["topo"]:
            run["topology"] = _EVIDENCE["topo"][mask_path]
        exact_runs.append(run)

    sample_id = Path(str(exact_qc.get("image_path", "sample"))).stem or "sample"
    return core.create_report(sample_id, exact_qc, exact_runs, final_decision)


__all__ = [
    "inspect_image",
    "run_segmentation",
    "audit_topology",
    "bounded_policy",
    "create_report",
]
