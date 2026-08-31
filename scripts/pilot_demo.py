#!/usr/bin/env python3
"""TopoScout cross-domain pilot harness — judge-runnable CLI (M7B).

    python scripts/pilot_demo.py list
    python scripts/pilot_demo.py doctor
    python scripts/pilot_demo.py run satellite
    python scripts/pilot_demo.py run microscopy --input my_cells.png

Runs a REGISTERED portability-pilot adapter (fixed public model) and the same
TopoScout structural audit that produced the frozen evidence in
artifacts/pilots/. Results are descriptive PILOT output — not validated
scientific, clinical, GIS, safety, or manufacturing decisions.

Safety properties:
- only trusted registry adapters can run (no model/checkpoint/threshold/
  executable/path selection surface exists on this CLI);
- new runs write ONLY under outputs/pilots/<domain>-<timestamp>/ — the frozen
  submission evidence in artifacts/pilots/ is never touched;
- each run prints the domain's model license/provenance notice.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pilots import registry  # noqa: E402
from pilots.base import ARTIFACT_ROOT, PilotUnavailable, audit_pilot_mask  # noqa: E402

OUT_ROOT = REPO / "outputs" / "pilots"
FROZEN = REPO / ARTIFACT_ROOT


@dataclass(frozen=True)
class Domain:
    name: str
    pilot: str                       # trusted registry key
    deps: tuple[str, ...]            # importable modules required
    sample: str | None               # bundled redistribution-safe sample
    setup: str
    note: str = ""


DOMAINS: dict[str, Domain] = {
    "satellite": Domain(
        "satellite", "satellite_road", ("keras", "huggingface_hub", "cv2"),
        "artifacts/pilots/satellite/input.tiff",
        "bash scripts/setup_pilot.sh satellite"),
    "materials": Domain(
        "materials", "materials_crack",
        ("torch", "segmentation_models_pytorch", "huggingface_hub", "cv2"),
        "artifacts/pilots/materials/input.png",
        "bash scripts/setup_pilot.sh materials"),
    "microscopy": Domain(
        "microscopy", "microscopy_cellpose", ("cellpose", "cv2"), None,
        "bash scripts/setup_pilot.sh microscopy",
        "provide your own cell image via --input (bundled sample omitted for licensing)"),
    "pathology": Domain(
        "pathology", "pathology_hovernet", ("tiatoolbox",), None,
        "bash scripts/setup_pilot.sh pathology",
        "provide your own H&E tile via --input (bundled sample omitted for licensing)"),
    "industrial": Domain(
        "industrial", "industrial_patchcore", ("anomalib", "torch"), None,
        "bash scripts/setup_pilot.sh industrial",
        "PatchCore fits a memory bank from NORMAL reference images: obtain MVTec AD "
        "from its owner (CC BY-NC-SA) and set TOPOSCOUT_MVTEC_ROOT, then pass a test "
        "image via --input; expect ~30 min on CPU"),
}


def _dep_ok(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def _status(d: Domain) -> tuple[bool, list[str]]:
    problems = []
    for m in d.deps:
        if not _dep_ok(m):
            problems.append(f"missing dependency: {m}  ({d.setup})")
    if d.sample is None:
        problems.append("no bundled sample — pass --input IMAGE" +
                        (f" ({d.note})" if d.note else ""))
    elif not (REPO / d.sample).is_file():
        problems.append(f"bundled sample not found: {d.sample}")
    if d.name == "industrial":
        import os
        if not os.environ.get("TOPOSCOUT_MVTEC_ROOT", "").strip():
            problems.append("TOPOSCOUT_MVTEC_ROOT not set (normal-image memory bank)")
    deps_ready = all(_dep_ok(m) for m in d.deps)
    return deps_ready and not problems, problems


def cmd_list() -> int:
    print("TopoScout portability pilots (maize is the separate, validated live app)\n")
    for d in DOMAINS.values():
        ready, problems = _status(d)
        state = "READY" if ready else ("runnable with --input" if all(_dep_ok(m) for m in d.deps)
                                       else "needs setup")
        print(f"  {d.name:<12} {state}")
    print("\nDetails: python scripts/pilot_demo.py doctor")
    return 0


def cmd_doctor() -> int:
    print("TopoScout cross-domain environment\n")
    for d in DOMAINS.values():
        ready, problems = _status(d)
        print(f"{d.name}")
        for m in d.deps:
            print(f"  {m:<28} {'OK' if _dep_ok(m) else 'MISSING'}")
        if d.sample:
            ok = (REPO / d.sample).is_file()
            print(f"  bundled sample               {'OK' if ok else 'MISSING'} ({d.sample})")
        for p in problems:
            print(f"  -> {p}")
        print(f"  runnable now                 {'YES' if ready else 'no'}\n")
    return 0


def cmd_run(name: str, input_path: str | None) -> int:
    if name not in DOMAINS:
        print(f"unknown domain {name!r}; choose from: {', '.join(DOMAINS)}", file=sys.stderr)
        return 2
    d = DOMAINS[name]

    image = Path(input_path) if input_path else (REPO / d.sample if d.sample else None)
    if image is None:
        print(f"{name}: no bundled sample — pass --input IMAGE. {d.note}", file=sys.stderr)
        return 2
    if not image.is_file():
        print(f"input image not found: {image}", file=sys.stderr)
        return 2

    try:
        adapter = registry.get_adapter(d.pilot)
    except PilotUnavailable as exc:
        print(f"{name}: dependencies missing ({exc}).\nInstall with: {d.setup}", file=sys.stderr)
        return 3

    out_dir = OUT_ROOT / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert FROZEN not in out_dir.parents and out_dir != FROZEN  # never the frozen evidence

    print(f"domain:  {name}")
    print(f"model:   {adapter.model_name}")
    print(f"license: {adapter.model_license}")
    print(f"profile: {adapter.profile.name} ({adapter.profile.structure})")
    print(f"input:   {image}")
    print("running fixed public model + TopoScout structural audit ...\n")

    result = adapter.predict(str(image), out_dir=out_dir)
    if result.get("status") != "ok":
        print(f"pilot failed: {json.dumps(result, indent=2)}", file=sys.stderr)
        return 4

    import cv2
    mask = cv2.imread(result["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
    audit = audit_pilot_mask(mask, adapter.profile)
    primary = {k: audit[k] for k in adapter.profile.primary_metrics if k in audit}

    evidence = {
        "pilot": True,
        "pilot_label": f"{name.upper()} PORTABILITY PILOT — descriptive audit only, "
                       "not a domain-validated decision",
        "domain": name,
        "adapter": adapter.adapter_name,
        "model": adapter.model_name,
        "model_source": adapter.model_source,
        "license": adapter.model_license,
        "input": str(image),
        "structural_audit": audit,
        "profile_primary_metrics": primary,
        "decision": None,
        "limitations": adapter.limitations,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2))

    print("structural audit (primary metrics):")
    for k, v in primary.items():
        print(f"  {k:<28} {v}")
    suspicious = (audit.get("fragmentation_score", 0) > 0.5
                  or audit.get("tiny_component_fraction", 0) > 0.5)
    print(f"\nassessment: {'STRUCTURALLY SUSPICIOUS (descriptive)' if suspicious else 'no gross structural anomaly (descriptive)'}")
    print("note: pilots define no validated ACCEPT rule; suspicious structure "
          "escalates to human review by construction.")
    try:
        shown = out_dir.relative_to(REPO)
    except ValueError:
        shown = out_dir
    print(f"\nartifacts: {shown}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    print(f"\nlimitations: {adapter.limitations}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pilot_demo",
                                 description="TopoScout portability pilots (judge CLI)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("doctor")
    runp = sub.add_parser("run")
    runp.add_argument("domain", choices=sorted(DOMAINS))
    runp.add_argument("--input", help="input image (defaults to the bundled sample where one exists)")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "doctor":
        return cmd_doctor()
    return cmd_run(args.domain, args.input)


if __name__ == "__main__":
    raise SystemExit(main())
