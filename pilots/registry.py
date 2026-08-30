"""Trusted pilot adapter registry (M6A).

Only names listed here can ever be instantiated; callers (including any LLM
tool surface later) select a REGISTERED name, never a model, checkpoint,
executable, threshold, or path. Adapters import lazily so a missing optional
dependency degrades to PilotUnavailable instead of breaking the package.
"""
from __future__ import annotations

from typing import Callable

from .base import PilotAdapter, PilotUnavailable

_REGISTRY: dict[str, Callable[[], PilotAdapter]] = {}


def register(name: str):
    def deco(factory: Callable[[], PilotAdapter]):
        assert name not in _REGISTRY, f"duplicate pilot {name}"
        _REGISTRY[name] = factory
        return factory
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(name: str) -> PilotAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown pilot adapter {name!r}; allowed: {available()}")
    return _REGISTRY[name]()


@register("microscopy_cellpose")
def _microscopy() -> PilotAdapter:
    try:
        from .microscopy.cellpose_adapter import CellposePilot
    except ImportError as exc:
        raise PilotUnavailable(f"cellpose not installed: {exc}") from exc
    return CellposePilot()


@register("pathology_hovernet")
def _pathology() -> PilotAdapter:
    try:
        from .pathology.hovernet_adapter import HoverNetPilot
    except ImportError as exc:
        raise PilotUnavailable(f"tiatoolbox not installed: {exc}") from exc
    return HoverNetPilot()


@register("satellite_road")
def _satellite() -> PilotAdapter:
    try:
        from .satellite.road_adapter import RoadSegmentationPilot
    except ImportError as exc:
        raise PilotUnavailable(f"satellite pilot deps missing: {exc}") from exc
    return RoadSegmentationPilot()


@register("materials_crack")
def _materials() -> PilotAdapter:
    try:
        from .materials.crack_adapter import CrackSegmentationPilot
    except ImportError as exc:
        raise PilotUnavailable(f"crack pilot deps missing: {exc}") from exc
    return CrackSegmentationPilot()


@register("industrial_patchcore")
def _industrial() -> PilotAdapter:
    try:
        from .industrial.patchcore_adapter import PatchCorePilot
    except ImportError as exc:
        raise PilotUnavailable(f"anomalib not installed: {exc}") from exc
    return PatchCorePilot()
