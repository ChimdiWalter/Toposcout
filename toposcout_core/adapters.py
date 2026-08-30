"""Segmentation adapter abstraction.

Two adapters share one structured-result contract:

  demo_dark_structure_v1  the hackathon demo adapter (toposcout_core.tools.
                          run_demo_segmentation) — kept as the fallback.
  real_lesion_model_v1    the pre-existing validated maize-lesion model
                          (critical_set_ph T0 tile baseline), run as a strict
                          subprocess inside its own scientific environment via
                          toposcout_core/real_worker.py.

Adapter selection is TRUSTED PYTHON CONFIGURATION ONLY (environment variable
TOPOSCOUT_SEGMENTATION_ADAPTER = "demo" | "real_lesion"). The language model
never chooses adapters, executables, checkpoints, thresholds, or output
locations. Remaining env overrides exist for trusted operators and tests.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import artifact_output_dir

ADAPTER_ENV = "TOPOSCOUT_SEGMENTATION_ADAPTER"
DEFAULT_ADAPTER = "demo"

REAL_ADAPTER_NAME = "real_lesion_model_v1"
DEFAULT_REAL_PYTHON = ""  # trusted operator config (private research venv)
DEFAULT_REAL_WORKER = str(Path(__file__).resolve().parent / "real_worker.py")
DEFAULT_REAL_CHECKPOINT = ""  # trusted operator config (private checkpoint)
DEFAULT_REAL_TIMEOUT_S = 900.0
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def selected_adapter() -> str:
    return os.environ.get(ADAPTER_ENV, "").strip() or DEFAULT_ADAPTER


def run_segmentation(image_path: str, output_dir: str | None = None, attempt: int = 1) -> dict[str, Any]:
    """Run the configured segmentation adapter (trusted Python-side selection)."""
    name = selected_adapter()
    if name == "demo":
        from .tools import run_demo_segmentation
        return run_demo_segmentation(image_path, output_dir, attempt=attempt)
    if name == "real_lesion":
        return run_real_lesion(image_path, output_dir, attempt=attempt)
    return {"status": "failed", "reason": "unknown_adapter_configured", "adapter": name}


def _fail(reason: str, **extra) -> dict[str, Any]:
    return {"status": "failed", "adapter": REAL_ADAPTER_NAME, "reason": reason, **extra}


def run_real_lesion(image_path: str, output_dir: str | None = None, attempt: int = 1) -> dict[str, Any]:
    """Strict subprocess adapter for the validated real lesion model.

    shell=False, fixed trusted script/interpreter/checkpoint, validated input
    path, timeout, machine-readable stdout, captured stderr, structured
    failures. No user-controlled command execution of any kind.
    """
    if attempt not in (1, 2):
        return _fail("invalid_attempt", attempt=attempt)

    p = Path(image_path)
    if not p.is_file():
        return _fail("input_not_found", image_path=str(p))
    if p.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        return _fail("unsupported_image_type", image_path=str(p), suffix=p.suffix)

    interpreter = Path(os.environ.get("TOPOSCOUT_REAL_PYTHON", "").strip() or DEFAULT_REAL_PYTHON)
    worker = Path(os.environ.get("TOPOSCOUT_REAL_WORKER", "").strip() or DEFAULT_REAL_WORKER)
    checkpoint = Path(os.environ.get("TOPOSCOUT_REAL_CHECKPOINT", "").strip() or DEFAULT_REAL_CHECKPOINT)
    if not interpreter.is_file():
        return _fail("missing_interpreter", interpreter=str(interpreter))
    if not worker.is_file():
        return _fail("missing_worker", worker=str(worker))
    if not checkpoint.is_file():
        return _fail("missing_checkpoint", checkpoint_path=str(checkpoint))
    try:
        timeout_s = float(os.environ.get("TOPOSCOUT_REAL_TIMEOUT_S", "").strip() or DEFAULT_REAL_TIMEOUT_S)
    except ValueError:
        timeout_s = DEFAULT_REAL_TIMEOUT_S

    out_dir = artifact_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{p.stem}.attempt{attempt}.mask.png"
    overlay_path = out_dir / f"{p.stem}.attempt{attempt}.overlay.jpg"
    prob_path = out_dir / f"{p.stem}.prob.npy"

    cmd = [
        str(interpreter), str(worker),
        "--image", str(p),
        "--out-mask", str(mask_path),
        "--out-overlay", str(overlay_path),
        "--out-prob", str(prob_path),
        "--attempt", str(attempt),
    ]
    try:
        proc = subprocess.run(cmd, shell=False, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return _fail("worker_timeout", timeout_seconds=timeout_s)
    except OSError as exc:
        return _fail("worker_spawn_error", error=str(exc))

    stderr_tail = proc.stderr.strip().splitlines()[-8:] if proc.stderr else []
    if proc.returncode != 0:
        return _fail("worker_crashed", returncode=proc.returncode, stderr_tail=stderr_tail)

    last_line = next((ln for ln in reversed(proc.stdout.strip().splitlines()) if ln.strip()), "")
    try:
        result = json.loads(last_line)
        if not isinstance(result, dict) or "status" not in result:
            raise ValueError("not a status dict")
    except (ValueError, json.JSONDecodeError):
        return _fail("malformed_worker_output",
                     stdout_tail=proc.stdout.strip().splitlines()[-4:], stderr_tail=stderr_tail)

    result.setdefault("adapter", REAL_ADAPTER_NAME)
    result.setdefault("attempt", attempt)
    return result
