"""M7B tests: judge CLI is closed, safe, and never touches frozen evidence.

Hermetic — no model downloads, no Gemini, no cloud.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import pilot_demo
from pilots.base import PilotAdapter, DomainProfile

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "artifacts" / "pilots"


def _frozen_digest() -> str:
    h = hashlib.sha256()
    for f in sorted(FROZEN.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(FROZEN)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def test_list_and_doctor_run_without_model_deps(capsys):
    assert pilot_demo.main(["list"]) == 0
    out = capsys.readouterr().out
    for name in ("satellite", "microscopy", "pathology", "materials", "industrial"):
        assert name in out
    assert pilot_demo.main(["doctor"]) == 0


def test_unknown_domain_rejected():
    with pytest.raises(SystemExit):
        pilot_demo.main(["run", "arbitrary_model"])


def test_cli_exposes_no_scientific_choices():
    import argparse
    # rebuild the parser and inspect the run subcommand's options
    ap = argparse.ArgumentParser()
    # crude but effective: the module source must not define such flags
    src = (REPO / "scripts" / "pilot_demo.py").read_text()
    for forbidden in ("--model", "--checkpoint", "--threshold", "--executable",
                      "--command", "--python", "--min-area"):
        assert forbidden not in src
    assert "--input" in src  # the only image-selection surface


def test_missing_input_for_byoi_domain(capsys):
    rc = pilot_demo.main(["run", "microscopy"])  # no bundled sample
    assert rc == 2
    assert "--input" in capsys.readouterr().err


class _StubAdapter(PilotAdapter):
    domain = "satellite"
    adapter_name = "stub_v1"
    model_name = "stub"
    model_source = "stub"
    model_license = "STUB LICENSE NOTICE"
    profile = DomainProfile("satellite_roads", "connected_network", ("beta_0",))
    limitations = "stub limitations"

    def _predict_mask(self, image_path, out_dir):
        mask = np.zeros((32, 32), bool)
        mask[4:10, 4:10] = True
        return {"mask": mask, "overlay_bgr": np.zeros((32, 32, 3), np.uint8)}


def test_run_writes_only_under_outputs_and_prints_license(monkeypatch, capsys, tmp_path):
    frozen_before = _frozen_digest()
    monkeypatch.setattr(pilot_demo.registry, "get_adapter", lambda name: _StubAdapter())
    monkeypatch.setattr(pilot_demo, "OUT_ROOT", tmp_path / "outputs" / "pilots")

    rc = pilot_demo.main(["run", "satellite"])  # uses the bundled sample as input
    assert rc == 0
    out = capsys.readouterr().out
    assert "STUB LICENSE NOTICE" in out          # license notice printed
    assert "stub limitations" in out
    assert "descriptive" in out                   # pilot-only interpretation

    runs = list((tmp_path / "outputs" / "pilots").iterdir())
    assert len(runs) == 1
    ev = json.loads((runs[0] / "evidence.json").read_text())
    assert ev["pilot"] is True and ev["decision"] is None
    assert ev["structural_audit"]["beta_0"] == 1

    assert _frozen_digest() == frozen_before      # frozen evidence untouched


def test_out_root_is_never_the_frozen_evidence():
    assert pilot_demo.OUT_ROOT != pilot_demo.FROZEN
    assert pilot_demo.FROZEN not in pilot_demo.OUT_ROOT.parents
    assert "outputs" in pilot_demo.OUT_ROOT.parts
