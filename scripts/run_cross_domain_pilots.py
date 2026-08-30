#!/usr/bin/env python3
"""Run cross-domain portability pilots (M6A) and freeze their evidence.

Each pilot: fixed registered adapter -> mask/overlay -> the SAME TopoScout
structural audit layer (domain-profile-selected metrics) -> evidence.json +
README under artifacts/pilots/<domain>/, plus a portability matrix row.

Pilots are DESCRIPTIVE: no maize thresholds are reused, and no pilot claims a
validated ACCEPT rule. Missing dependencies are reported, never faked.

Usage:
    python scripts/run_cross_domain_pilots.py [--pilot satellite_road ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pilots import registry  # noqa: E402
from pilots.base import ARTIFACT_ROOT, PilotUnavailable, audit_pilot_mask, write_evidence  # noqa: E402

INPUTS = {
    "satellite": "input.tiff",
    "microscopy": "input.png",
    "pathology": "input.png",
    "materials": "input.png",
    "industrial": "input.png",
}

PILOT_LABEL = {
    "microscopy": "MICROSCOPY PORTABILITY PILOT — NOT BIOLOGICALLY VALIDATED",
    "pathology": "PATHOLOGY PORTABILITY PILOT — NOT FOR CLINICAL USE",
    "satellite": "SATELLITE PORTABILITY PILOT — NOT FOR OPERATIONAL GIS USE",
    "materials": "MATERIALS PORTABILITY PILOT — NOT A SAFETY CERTIFICATION",
    "industrial": "INDUSTRIAL PORTABILITY PILOT — NO ACCEPTANCE CLAIMS",
}


def run_pilot(name: str) -> dict:
    import cv2

    adapter = registry.get_adapter(name)
    domain = adapter.domain
    input_path = ARTIFACT_ROOT / domain / INPUTS[domain]
    if not input_path.is_file():
        return {"pilot_name": name, "status": "failed", "reason": "input_missing",
                "input_path": str(input_path)}

    result = adapter.predict(str(input_path))
    if result.get("status") != "ok":
        return {"pilot_name": name, **result}

    mask = cv2.imread(result["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
    audit = audit_pilot_mask(mask, adapter.profile)

    evidence = {
        "pilot": True,
        "pilot_label": PILOT_LABEL[domain],
        "domain": domain,
        "adapter": adapter.adapter_name,
        "model": adapter.model_name,
        "model_source": adapter.model_source,
        "license": adapter.model_license,
        "input": str(input_path),
        "mask": result["mask_path"],
        "overlay": result["overlay_path"],
        "runtime_seconds": result["runtime_seconds"],
        "structural_audit": audit,
        "profile_primary_metrics": {k: audit[k] for k in adapter.profile.primary_metrics
                                    if k in audit},
        "decision": None,  # descriptive pilot: no validated accept rule
        "limitations": adapter.limitations,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_evidence(domain, evidence)

    readme = ARTIFACT_ROOT / domain / "README.md"
    lines = [f"# {PILOT_LABEL[domain]}", "",
             f"- **Model**: {adapter.model_name}",
             f"- **Source**: {adapter.model_source}",
             f"- **License**: {adapter.model_license}",
             f"- **Structural question** ({adapter.profile.structure}):",
             f"  metrics {', '.join(adapter.profile.primary_metrics)}", "",
             "Key measurements:", "```json",
             json.dumps(evidence["profile_primary_metrics"], indent=2),
             "```", "", f"Limitations: {adapter.limitations}"]
    readme.write_text("\n".join(lines))
    return {"pilot_name": name, "status": "ok", **{k: evidence[k] for k in
            ("domain", "model", "license", "runtime_seconds")},
            "primary_metrics": evidence["profile_primary_metrics"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="append", help="registered pilot name(s)")
    args = ap.parse_args()
    names = args.pilot or registry.available()

    rows = []
    for name in names:
        print(f"=== {name} ===", flush=True)
        try:
            row = run_pilot(name)
        except PilotUnavailable as exc:
            row = {"pilot_name": name, "status": "unavailable", "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001
            row = {"pilot_name": name, "status": "failed", "reason": repr(exc)}
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)

    matrix_path = ARTIFACT_ROOT / "portability_matrix.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.is_file() else {}
    for row in rows:
        matrix[row["pilot_name"]] = row
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix, indent=2))
    print(f"matrix -> {matrix_path}")


if __name__ == "__main__":
    main()
