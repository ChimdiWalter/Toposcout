#!/usr/bin/env python3
"""Vendor the exact research sources + checkpoint the cloud worker needs.

Import-traces `scientific_worker.model_loader.build_cloud_model()` (the same
strict load the container performs) and copies ONLY the research-tree Python
files that were actually imported — preserving their sys.path-relative layout —
into a Docker build context, plus the fixed T0 checkpoint. Nothing else from
the research tree enters the image.

Output layout (default build/worker_context/):

    research/            vendored sources (TOPOSCOUT_SW_RESEARCH_ROOT)
    models/best_checkpoint.pth
    vendor_manifest.json sha256 of every vendored file + the checkpoint

Run inside the research venv:  python scripts/vendor_worker_sources.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scientific_worker.model_loader import (  # noqa: E402
    CHECKPOINT_PATH, RESEARCH_ROOT, build_cloud_model, sha256_file, _research_sys_path,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "build" / "worker_context"))
    ap.add_argument("--skip-checkpoint", action="store_true",
                    help="manifest + sources only (checkpoint copied separately)")
    args = ap.parse_args()
    out = Path(args.out)

    before = set(sys.modules)
    model, report = build_cloud_model()
    del model
    new_modules = [sys.modules[name] for name in set(sys.modules) - before]

    # Files actually imported from the research tree, keyed by the sys.path
    # root they were imported under (so the vendored tree reproduces the
    # exact package layout _research_sys_path() expects). Roots are matched
    # RESOLVED because v1_src/maize_lesion_study_v1_src is a symlink into
    # maize_lesion_study_v1/src — the vendored tree materializes that symlink
    # as a real directory at its sys.path location.
    root_pairs: list[tuple[Path, Path]] = []  # (resolved root, layout rel to RESEARCH_ROOT)
    for p in _research_sys_path():
        up = Path(p)
        rel = up.relative_to(RESEARCH_ROOT) if up != RESEARCH_ROOT else Path(".")
        root_pairs.append((up.resolve(), rel))

    vendored: list[tuple[Path, Path]] = []  # (resolved src, dest-rel)
    for mod in new_modules:
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        src = Path(f).resolve()
        matches = [(r, rel) for r, rel in root_pairs if src.is_relative_to(r)]
        if not matches:
            if src.is_relative_to(RESEARCH_ROOT.resolve()):
                raise SystemExit(f"imported research file outside known roots: {src}")
            continue  # site-packages etc.
        root, root_rel = max(matches, key=lambda t: len(t[0].parts))  # most specific
        vendored.append((src, Path("research") / root_rel / src.relative_to(root)))

    if not vendored:
        raise SystemExit("no research files traced — loader changed?")

    if out.exists():
        shutil.rmtree(out)
    manifest: dict = {"loader_report": report,
                      "research_root": str(RESEARCH_ROOT), "files": {}}
    for src, rel in sorted(vendored, key=lambda t: str(t[1])):
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        manifest["files"][str(rel)] = sha256_file(src)

    # package chain: copy __init__.py for every traced package directory
    for src, rel in vendored:
        sp, rp = src.parent, (out / rel).parent
        while rp != out / "research" and rp != out:
            init_src = sp / "__init__.py"
            init_dst = rp / "__init__.py"
            if init_src.is_file() and not init_dst.exists():
                shutil.copy2(init_src, init_dst)
                manifest["files"][str(init_dst.relative_to(out))] = sha256_file(init_src)
            sp, rp = sp.parent, rp.parent

    # the worker package itself + its requirements complete the build context
    for py in sorted((REPO / "scientific_worker").glob("*.py")):
        dest = out / "scientific_worker" / py.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py, dest)
        manifest["files"][f"scientific_worker/{py.name}"] = sha256_file(py)
    req = REPO / "scientific_worker" / "requirements.txt"
    shutil.copy2(req, out / "requirements.txt")
    manifest["files"]["requirements.txt"] = sha256_file(req)

    if not args.skip_checkpoint:
        (out / "models").mkdir(parents=True, exist_ok=True)
        shutil.copy2(CHECKPOINT_PATH, out / "models" / "best_checkpoint.pth")
    manifest["checkpoint_sha256"] = report["checkpoint_sha256"]

    (out / "vendor_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"out": str(out), "n_files": len(manifest["files"]),
                      "checkpoint_sha256": manifest["checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
