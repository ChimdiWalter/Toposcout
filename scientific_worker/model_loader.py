#!/usr/bin/env python3
"""Cloud-safe loader for the validated T0 tile baseline (real_lesion_model_v1).

The research loader (`p8_significance_eval.load_model`) builds
GeometrySpecialist with pretrained=True, which (a) downloads the timm
`vit_base_patch14_dinov2` ImageNet/DINOv2 weights and (b) loads the ~1.3 GB
Stage-L CPT checkpoint — then STRICT-overwrites everything with the final
335 MB T0 checkpoint. For Cloud Run both preloads are redundant: strict
success proves `best_checkpoint.pth` carries the complete model state.

This loader builds the IDENTICAL architecture with pretrained=False (verified
in source: `DINOv2Backbone.__init__` gates the CPT load on `pretrained` and
passes `pretrained` straight to `timm.create_model`, so neither download nor
CPT load happens) and then strict-loads the same fixed checkpoint.

Model construction kwargs are FROZEN here, copied from the resolved
t0_tile_base.yaml config; the local equivalence check re-derives them from the
research config and asserts they match. Nothing here is LLM- or
request-controllable.

Scientific equivalence to the research loader is gated by
scripts/check_cloud_loader_equivalence.py (frozen tolerance: probability
max_abs_diff <= 1e-6 AND exact binary-mask equality at 0.5). Do not deploy a
worker whose loader has not passed that gate.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

# ── Fixed trusted locations ──────────────────────────────────────────────────
# Local research tree (read-only). In the container these point at the
# vendored copies baked into the image (see scientific_worker/Dockerfile).
# The research sources and the validated checkpoint live in a PRIVATE
# archive; deployments set these via environment (the container bakes in the
# vendored copies — see scientific_worker/Dockerfile).
DEFAULT_RESEARCH_ROOT = ""
DEFAULT_CHECKPOINT = ""

RESEARCH_ROOT = Path(os.environ.get("TOPOSCOUT_SW_RESEARCH_ROOT", "").strip() or DEFAULT_RESEARCH_ROOT)
CHECKPOINT_PATH = Path(os.environ.get("TOPOSCOUT_SW_CHECKPOINT", "").strip() or DEFAULT_CHECKPOINT)

MODEL_VERSION = "critical_set_ph/p5_t0_tile_base (T0 tile baseline, dice_bce, seed 1337)"
ADAPTER_NAME = "real_lesion_model_v1"

# Frozen from the resolved research config
# (experiments/critical_set_ph/configs/phase5/t0_tile_base.yaml extends
# configs/_base.yaml). The equivalence script asserts these against
# load_config output — never edit by hand without re-running the gate.
FROZEN_MODEL_KWARGS: dict[str, Any] = {
    "backbone": "dinov2_maize",
    "tune_mode": "partial",
    "decoder_embed": 256,
    "geometry": "hes",
    "latent_dims": {"dh": 8, "de": 32, "ds": 8},
    "curvature": {"learn": False, "init_kh": 1.0, "init_ks": 1.0},
    "manifold_mode": "tangent_proxy",
    "sanitize_outputs": True,
}

# Inference constants — identical to p8_significance_eval / real_worker.py.
TILE, STRIDE, NET = 512, 384, 518
PROB_THRESHOLD = 0.5
SIGNIFICANCE_MIN_AREA_PX = 500


def _research_sys_path() -> list[str]:
    """sys.path entries the research modules expect (mirrors p8's setup)."""
    mod = RESEARCH_ROOT / "experiments" / "critical_set_ph"
    return [
        str(RESEARCH_ROOT),                                        # src.*
        str(RESEARCH_ROOT / "v1_src" / "maize_lesion_study_v1_src"),  # models.*
        str(mod / "src"),
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cloud_model(checkpoint_path: Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Build GeometrySpecialist with pretrained=False and strict-load the
    fixed T0 checkpoint.

    Returns (model.eval(), report) where report has missing/unexpected keys
    (must be empty), checkpoint sha256, and architecture identity.
    Raises on any strict-load failure — a non-equivalent model must never
    come into existence silently.
    """
    import torch

    ckpt_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    for p in _research_sys_path():
        if p not in sys.path:
            sys.path.insert(0, p)
    from src.models.geometry_specialist import GeometrySpecialist  # noqa: E402

    net = GeometrySpecialist(pretrained=False, **FROZEN_MODEL_KWARGS)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    incompat = net.load_state_dict(state, strict=True)  # raises on mismatch
    missing = list(getattr(incompat, "missing_keys", []))
    unexpected = list(getattr(incompat, "unexpected_keys", []))
    if missing or unexpected:  # strict=True should already have raised
        raise RuntimeError(f"non-strict checkpoint load: missing={missing[:5]} unexpected={unexpected[:5]}")

    net = net.eval()
    report = {
        "loader": "toposcout_cloud_safe_v1",
        "adapter": ADAPTER_NAME,
        "model_version": MODEL_VERSION,
        "architecture": type(net).__name__,
        "backbone": FROZEN_MODEL_KWARGS["backbone"],
        "pretrained_preloads": False,
        "n_parameters": int(sum(p.numel() for p in net.parameters())),
        "n_state_tensors": len(net.state_dict()),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return net, report
