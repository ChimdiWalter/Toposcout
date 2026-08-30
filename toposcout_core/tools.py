from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .components import connected_components, hole_count, save_mask
from .config import artifact_output_dir

ACTIONS = {"ACCEPT", "RETRY", "HUMAN_REVIEW", "REQUEST_REACQUISITION"}
MAX_SEGMENTATION_ATTEMPTS = 2


def inspect_image(image_path: str) -> dict[str, Any]:
    """Inspect a scientific image for basic acquisition and quality problems."""
    p = Path(image_path)
    result: dict[str, Any] = {"image_path": str(p), "exists": p.exists()}
    if not p.exists():
        return {**result, "status": "failed", "reason": "file_not_found"}

    try:
        with Image.open(p) as im:
            im.load()
            gray = np.asarray(im.convert("L"), dtype=np.float32)
            result.update({
                "status": "ok",
                "format": im.format,
                "width": int(im.width),
                "height": int(im.height),
                "mode": im.mode,
                "mean_intensity": float(gray.mean()),
                "contrast_std": float(gray.std()),
            })
            # Gradient-energy proxy for sharpness; deterministic and dependency-light.
            if gray.shape[0] > 2 and gray.shape[1] > 2:
                gy = np.diff(gray, axis=0)
                gx = np.diff(gray, axis=1)
                sharpness = float((gx.var() + gy.var()) / 2.0)
            else:
                sharpness = 0.0
            result["sharpness_score"] = sharpness

            issues: list[str] = []
            if min(im.width, im.height) < 64:
                issues.append("very_small_image")
            if gray.mean() < 12:
                issues.append("severely_underexposed")
            if gray.mean() > 245:
                issues.append("severely_overexposed")
            if gray.std() < 8:
                issues.append("very_low_contrast")
            if sharpness < 2:
                issues.append("possible_blur_or_flat_image")
            result["issues"] = issues
            result["qc_pass"] = not any(
                x in issues for x in {
                    "very_small_image", "severely_underexposed", "severely_overexposed"
                }
            )
            return result
    except Exception as exc:  # noqa: BLE001
        return {**result, "status": "failed", "reason": "unreadable_image", "error": str(exc)}


def run_demo_segmentation(image_path: str, output_dir: str | None = None, attempt: int = 1) -> dict[str, Any]:
    """Run the hackathon demo segmentation adapter and save a binary mask.

    This is deliberately a simple deterministic adapter, not the scientific claim of
    the project. Attempt 2 applies contrast enhancement and a less aggressive cutoff.
    output_dir=None resolves the canonical artifact directory (TOPOSCOUT_OUTPUT_DIR).
    """
    p = Path(image_path)
    out_dir = artifact_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(p) as im:
        gray_im = im.convert("L")
        if attempt >= 2:
            gray_im = ImageEnhance.Contrast(gray_im).enhance(1.5).filter(ImageFilter.MedianFilter(3))
        gray = np.asarray(gray_im, dtype=np.float32)

    # Generic foreground demo: isolate relatively dark structures.
    percentile = 28 if attempt == 1 else 35
    threshold = float(np.percentile(gray, percentile))
    mask = gray < threshold

    # Remove tiny components to make topology diagnostics interpretable.
    stats = connected_components(mask)
    min_area = max(8, int(mask.size * 0.0002))
    if stats.count:
        # Re-label using a lightweight flood fill so tiny islands can be removed.
        from collections import deque
        h, w = mask.shape
        cleaned = np.zeros_like(mask, dtype=bool)
        seen = np.zeros_like(mask, dtype=bool)
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or seen[y, x]:
                    continue
                q = deque([(y, x)]); seen[y, x] = True; pixels = []
                while q:
                    cy, cx = q.popleft(); pixels.append((cy, cx))
                    for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                if len(pixels) >= min_area:
                    for py, px in pixels:
                        cleaned[py, px] = True
        mask = cleaned

    mask_path = out_dir / f"{p.stem}.attempt{attempt}.mask.png"
    save_mask(mask, mask_path)
    foreground_fraction = float(mask.mean())
    return {
        "status": "ok",
        "adapter": "demo_dark_structure_v1",
        "attempt": attempt,
        "threshold": threshold,
        "mask_path": str(mask_path),
        "foreground_fraction": foreground_fraction,
    }


def audit_topology(mask_path: str) -> dict[str, Any]:
    """Compute structural diagnostics from a binary mask."""
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L")) > 127
    stats = connected_components(mask)
    beta_0 = stats.count
    beta_1 = hole_count(mask)
    total_fg = int(mask.sum())
    largest = max(stats.areas, default=0)
    tiny_cutoff = max(8, int(mask.size * 0.0005))
    tiny_components = sum(a < tiny_cutoff for a in stats.areas)
    fragmentation_score = float(tiny_components / max(1, beta_0))
    largest_component_fraction = float(largest / max(1, total_fg))
    return {
        "status": "ok",
        "beta_0": beta_0,
        "beta_1": beta_1,
        "tiny_components": int(tiny_components),
        "fragmentation_score": fragmentation_score,
        "largest_component_fraction": largest_component_fraction,
        "foreground_pixels": total_fg,
    }


def bounded_policy(qc: dict[str, Any], segmentation: dict[str, Any] | None, topology: dict[str, Any] | None, attempt: int) -> dict[str, Any]:
    """Return a safe bounded next action. Gemini may explain or select among these same actions."""
    if qc.get("status") != "ok" or not qc.get("qc_pass", False):
        return {"action": "REQUEST_REACQUISITION", "reason": "input_failed_basic_qc"}
    if segmentation is None or topology is None:
        return {"action": "HUMAN_REVIEW", "reason": "missing_analysis_evidence"}

    fg = float(segmentation.get("foreground_fraction", 0.0))
    frag = float(topology.get("fragmentation_score", 1.0))
    beta0 = int(topology.get("beta_0", 0))

    # Deliberately conservative generic thresholds for the demo adapter.
    suspicious = fg < 0.002 or fg > 0.65 or beta0 == 0 or frag > 0.55
    if suspicious and attempt < MAX_SEGMENTATION_ATTEMPTS:
        return {"action": "RETRY", "reason": "structural_or_coverage_anomaly"}
    if suspicious:
        return {"action": "HUMAN_REVIEW", "reason": "anomaly_persisted_after_retry"}
    return {"action": "ACCEPT", "reason": "qc_and_structural_checks_passed"}


def _fmt(value: Any) -> str:
    """Render a value exactly as it appears in the JSON report (no rounding)."""
    return json.dumps(value)


def build_display_summary(report: dict[str, Any]) -> str:
    """Deterministic human-readable summary with exact tool-output values.

    Generated entirely in Python so the language model can relay it verbatim
    instead of re-narrating (and accidentally rounding) scientific numbers.
    """
    lines: list[str] = []
    lines.append(f"TopoScout report for sample {_fmt(report['sample_id'])}")
    final = report.get("final_decision", {})
    lines.append(f"Final action: {final.get('action')} (reason: {final.get('reason')})")

    qc = report.get("qc", {})
    if qc.get("status") == "ok":
        lines.append(
            "QC: pass=" + _fmt(qc.get("qc_pass"))
            + f", size={qc.get('width')}x{qc.get('height')}, mode={qc.get('mode')}"
            + ", mean_intensity=" + _fmt(qc.get("mean_intensity"))
            + ", contrast_std=" + _fmt(qc.get("contrast_std"))
            + ", sharpness_score=" + _fmt(qc.get("sharpness_score"))
            + ", issues=" + _fmt(qc.get("issues", []))
        )
    else:
        lines.append("QC: status=" + _fmt(qc.get("status")) + ", reason=" + _fmt(qc.get("reason")))

    for run in report.get("runs", []):
        seg = run.get("segmentation") or {}
        topo = run.get("topology") or {}
        attempt = run.get("attempt", seg.get("attempt"))
        threshold = seg.get("threshold", seg.get("prob_threshold"))
        seg_line = (
            f"Attempt {attempt} segmentation: adapter={seg.get('adapter')}"
            + ", threshold=" + _fmt(threshold)
            + ", foreground_fraction=" + _fmt(seg.get("foreground_fraction"))
        )
        if "min_area_px" in seg:
            seg_line += ", min_area_px=" + _fmt(seg.get("min_area_px"))
        if "n_components_significant" in seg:
            seg_line += ", n_components_significant=" + _fmt(seg.get("n_components_significant"))
        if seg.get("model_version"):
            seg_line += f", model={seg.get('model_version')}"
        seg_line += f", mask_path={seg.get('mask_path')}"
        lines.append(seg_line)
        lines.append(
            f"Attempt {attempt} topology: beta_0=" + _fmt(topo.get("beta_0"))
            + ", beta_1=" + _fmt(topo.get("beta_1"))
            + ", foreground_pixels=" + _fmt(topo.get("foreground_pixels"))
            + ", tiny_components=" + _fmt(topo.get("tiny_components"))
            + ", fragmentation_score=" + _fmt(topo.get("fragmentation_score"))
            + ", largest_component_fraction=" + _fmt(topo.get("largest_component_fraction"))
        )
        decision = run.get("decision")
        if decision:
            lines.append(
                f"Attempt {attempt} policy: {decision.get('action')} (reason: {decision.get('reason')})"
            )
    return "\n".join(lines)


def create_report(sample_id: str, qc: dict[str, Any], runs: list[dict[str, Any]], final_decision: dict[str, Any], output_dir: str | None = None) -> dict[str, Any]:
    """Create a machine-readable experiment evidence report.

    output_dir=None resolves the canonical artifact directory (TOPOSCOUT_OUTPUT_DIR).
    """
    out_dir = artifact_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "sample_id": sample_id,
        "workflow": "TopoScout MVP v0.1",
        "qc": qc,
        "runs": runs,
        "final_decision": final_decision,
        "scientific_guardrail": "Measurements are deterministic tool outputs; the language model must not invent values.",
    }
    report["display_summary"] = build_display_summary(report)
    report_path = out_dir / f"{sample_id}.report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"status": "ok", "report_path": str(report_path), "report": report}
