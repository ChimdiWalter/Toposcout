"""M6A tests: pilot registry closure, evidence contract, maize isolation,
graceful degradation, and no-fabrication (audits recomputable from artifacts).

Hermetic: no Gemini, no model downloads. Pilot model deps are intentionally
NOT installed in the test venv, so the graceful-unavailable path is exercised
for real.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from pilots import registry
from pilots.base import (
    ARTIFACT_ROOT, REQUIRED_RESULT_KEYS, DomainProfile, PilotAdapter,
    PilotUnavailable, audit_pilot_mask,
)
from pilots.profiles import ALL_PROFILES

REPO = Path(__file__).resolve().parents[2]

EXPECTED_PILOTS = {
    "microscopy_cellpose", "pathology_hovernet", "satellite_road",
    "materials_crack", "industrial_patchcore",
}


# ---------- registry closure ----------

def test_registry_lists_exactly_the_known_pilots():
    assert set(registry.available()) == EXPECTED_PILOTS


def test_registry_rejects_unknown_adapters():
    with pytest.raises(KeyError):
        registry.get_adapter("arbitrary_executable")
    with pytest.raises(KeyError):
        registry.get_adapter("../../evil")


def test_missing_optional_dependencies_fail_gracefully():
    """In this venv no pilot model package is installed: every factory must
    either build (pure-python) or raise PilotUnavailable — never ImportError."""
    for name in registry.available():
        try:
            adapter = registry.get_adapter(name)
        except PilotUnavailable:
            continue
        assert isinstance(adapter, PilotAdapter)


def test_predict_exposes_no_scientific_choices():
    params = inspect.signature(PilotAdapter.predict).parameters
    assert set(params) == {"self", "image_path"}
    for factory in registry._REGISTRY.values():
        assert len(inspect.signature(factory).parameters) == 0


# ---------- profiles ----------

def test_every_expected_domain_has_a_profile():
    assert set(ALL_PROFILES) == {"microscopy_cells", "pathology_nuclei",
                                 "satellite_roads", "materials_crack",
                                 "industrial_defect"}
    for profile in ALL_PROFILES.values():
        assert isinstance(profile, DomainProfile)
        assert profile.primary_metrics  # explicit metric selection, never implicit


def test_pilots_never_import_maize_policy():
    for py in (REPO / "pilots").rglob("*.py"):
        src = py.read_text()
        assert "toposcout_core" not in src, f"{py} couples pilots to maize code"
        assert "bounded_policy" not in src


def test_maize_policy_untouched():
    from toposcout_core.tools import MAX_SEGMENTATION_ATTEMPTS, bounded_policy
    assert MAX_SEGMENTATION_ATTEMPTS == 2
    qc = {"status": "ok", "qc_pass": True}
    seg = {"foreground_fraction": 0.3}
    assert bounded_policy(qc, seg, {"fragmentation_score": 0.846, "beta_0": 65}, 1) == \
        {"action": "RETRY", "reason": "structural_or_coverage_anomaly"}
    assert bounded_policy(qc, seg, {"fragmentation_score": 0.5, "beta_0": 20}, 2) == \
        {"action": "ACCEPT", "reason": "qc_and_structural_checks_passed"}


# ---------- frozen evidence artifacts ----------

def _evidence_files():
    return sorted(ARTIFACT_ROOT.glob("*/evidence.json")) if ARTIFACT_ROOT.is_dir() else []


def test_generated_evidence_follows_the_pilot_contract():
    files = _evidence_files()
    if not files:
        pytest.skip("no pilot artifacts generated yet")
    for f in files:
        ev = json.loads(f.read_text())
        assert ev["pilot"] is True
        assert "PILOT" in ev["pilot_label"].upper()
        for key in ("domain", "model", "model_source", "license", "input",
                    "mask", "overlay", "runtime_seconds", "structural_audit",
                    "limitations"):
            assert key in ev, f"{f}: missing {key}"
        assert ev["decision"] is None  # descriptive pilots define no accept rule
        assert ev["structural_audit"]["profile"] in ALL_PROFILES


def test_evidence_metrics_are_recomputable_not_fabricated():
    """Every stored structural audit must equal a fresh recomputation from the
    frozen mask artifact."""
    files = _evidence_files()
    if not files:
        pytest.skip("no pilot artifacts generated yet")
    from PIL import Image
    for f in files:
        ev = json.loads(f.read_text())
        mask_path = REPO / ev["mask"]
        if not mask_path.is_file():
            # public distributions omit NC-licensed derived imagery; the full
            # verification runs wherever the frozen masks are present
            assert ev.get("public_imagery") is False, f"mask artifact missing for {f}"
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        fresh = audit_pilot_mask(mask, ALL_PROFILES[ev["structural_audit"]["profile"]])
        assert fresh == ev["structural_audit"], f"{f}: stored audit != recomputed"


def test_pilot_result_contract_keys_are_frozen():
    assert REQUIRED_RESULT_KEYS == {
        "status", "pilot", "domain", "adapter", "model_name", "model_source",
        "model_license", "input_path", "mask_path", "overlay_path",
        "runtime_seconds",
    }
