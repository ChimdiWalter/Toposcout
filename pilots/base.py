"""Cross-domain portability pilots — shared contract (M6A).

A pilot wraps ONE fixed public pretrained model behind the TopoScout evidence
contract so the SAME structural-audit layer can consume its mask. Pilots are
explicitly NOT scientifically validated applications; every result they emit
carries ``pilot: true`` plus provenance (model, source, license) and
limitations. The validated reference application remains the maize pipeline.

The LLM never chooses executables, models, checkpoints, thresholds, or output
directories: adapters are instantiated only through pilots.registry, and every
adapter freezes its own model identity and parameters at import time.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_RESULT_KEYS = {
    "status", "pilot", "domain", "adapter", "model_name", "model_source",
    "model_license", "input_path", "mask_path", "overlay_path", "runtime_seconds",
}

ARTIFACT_ROOT = Path("artifacts/pilots")


class PilotUnavailable(RuntimeError):
    """Raised when a pilot's optional dependency or checkpoint is missing."""


@dataclass(frozen=True)
class DomainProfile:
    """Which deterministic structural measurements matter for a domain.

    Profiles select measurements; they do NOT inherit maize policy thresholds.
    A pilot that defines no policy performs descriptive auditing only and
    escalates suspicious structure to HUMAN_REVIEW by construction.
    """

    name: str
    structure: str                      # e.g. many_instances, connected_network
    primary_metrics: tuple[str, ...] = ()


class PilotAdapter(ABC):
    """One fixed public model behind the TopoScout pilot contract."""

    domain: str
    adapter_name: str
    model_name: str
    model_source: str
    model_license: str
    profile: DomainProfile
    limitations: str

    @abstractmethod
    def _predict_mask(self, image_path: Path, out_dir: Path) -> dict[str, Any]:
        """Run the fixed model; return at least {mask (bool HxW), overlay_bgr}."""

    def predict(self, image_path: str, out_dir: str | Path | None = None) -> dict[str, Any]:
        """Run the fixed model on image_path.

        out_dir is TRUSTED Python-side configuration (the frozen-evidence
        runner uses artifacts/pilots/<domain>; the judge CLI uses a fresh
        outputs/pilots/... directory). It is never caller-controlled from any
        LLM or web surface.
        """
        import cv2

        p = Path(image_path)
        if not p.is_file():
            return {"status": "failed", "reason": "input_not_found", "pilot": True,
                    "domain": self.domain, "adapter": self.adapter_name}
        out_dir = Path(out_dir) if out_dir else ARTIFACT_ROOT / self.domain
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        raw = self._predict_mask(p, out_dir)
        mask = raw["mask"].astype(bool)
        overlay = raw["overlay_bgr"]

        mask_path = out_dir / "mask.png"
        overlay_path = out_dir / "overlay.jpg"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(overlay_path), overlay)

        result: dict[str, Any] = {
            "status": "ok",
            "pilot": True,
            "domain": self.domain,
            "adapter": self.adapter_name,
            "model_name": self.model_name,
            "model_source": self.model_source,
            "model_license": self.model_license,
            "input_path": str(p),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "runtime_seconds": round(time.time() - t0, 2),
        }
        for extra in ("prob_path", "anomaly_map_path", "n_instances"):
            if extra in raw:
                result[extra] = raw[extra]
        missing = REQUIRED_RESULT_KEYS - set(result)
        assert not missing, f"pilot contract violation: missing {missing}"
        return result


def component_stats(mask: np.ndarray) -> dict[str, Any]:
    """Deterministic component-level structure shared by all pilot audits."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    areas = (np.bincount(lab.ravel())[1:] if n else np.array([], dtype=int))
    total = int(mask.sum())
    tiny_cutoff = max(8, int(mask.size * 0.0005))
    tiny = int((areas < tiny_cutoff).sum()) if n else 0
    stats = {
        "beta_0": int(n),
        "foreground_pixels": total,
        "foreground_fraction": float(mask.mean()),
        "tiny_component_cutoff_px": tiny_cutoff,
        "tiny_components": tiny,
        "tiny_component_fraction": float(tiny / n) if n else 0.0,
        "largest_component_fraction": float(areas.max() / total) if total and n else 0.0,
        "component_area_quartiles_px": ([int(v) for v in np.percentile(areas, [25, 50, 75])]
                                        if n else []),
    }
    return stats


def hole_count(mask: np.ndarray) -> int:
    from scipy import ndimage

    inv = ~mask
    lab, n = ndimage.label(inv)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    return int(len({i for i in range(1, n + 1) if i not in border}))


def skeleton_stats(mask: np.ndarray) -> dict[str, Any]:
    """Skeleton continuity evidence for thin/network structures."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
    neigh = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                             mode="constant") - skel.astype(np.uint8)
    endpoints = int(((neigh == 1) & skel).sum())
    return {"skeleton_components": int(n), "skeleton_endpoints": endpoints,
            "skeleton_pixels": int(skel.sum())}


def audit_pilot_mask(mask: np.ndarray, profile: DomainProfile) -> dict[str, Any]:
    """Domain-profile-selected deterministic structural audit of a pilot mask."""
    audit: dict[str, Any] = {"profile": profile.name, "structure": profile.structure}
    audit.update(component_stats(mask))
    audit["beta_1"] = hole_count(mask)
    if profile.structure in {"connected_network", "connected_thin_network"}:
        audit.update(skeleton_stats(mask))
    audit["fragmentation_score"] = (float(audit["tiny_components"] / max(1, audit["beta_0"]))
                                    if audit["beta_0"] else 0.0)
    return audit


def write_evidence(domain: str, payload: dict[str, Any]) -> Path:
    out = ARTIFACT_ROOT / domain / "evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out
