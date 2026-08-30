"""Milestone 3 hardening tests: artifact path control, exact number
preservation, bounded retry behavior, and escalation."""
import inspect
import json
from pathlib import Path

import pytest

import app.tools as agent_tools
from toposcout_core import fixtures, run_local_workflow
from toposcout_core.config import DEFAULT_OUTPUT_DIR, ENV_OUTPUT_DIR, artifact_output_dir
from toposcout_core.tools import (
    MAX_SEGMENTATION_ATTEMPTS,
    bounded_policy,
    build_display_summary,
    create_report,
    run_demo_segmentation,
)


# ---------- canonical artifact output directory ----------

def test_default_output_dir_is_outputs(monkeypatch):
    monkeypatch.delenv(ENV_OUTPUT_DIR, raising=False)
    assert artifact_output_dir() == Path(DEFAULT_OUTPUT_DIR)


def test_env_var_controls_artifact_dir(monkeypatch, tmp_path: Path):
    canonical = tmp_path / "canonical"
    monkeypatch.setenv(ENV_OUTPUT_DIR, str(canonical))
    image = fixtures.write_fragmented_recoverable_image(tmp_path / "in" / "img.png")

    seg = run_demo_segmentation(str(image), attempt=1)
    assert Path(seg["mask_path"]).parent == canonical
    assert Path(seg["mask_path"]).exists()

    result = create_report("img", {"status": "ok"}, [], {"action": "ACCEPT", "reason": "x"})
    assert Path(result["report_path"]).parent == canonical
    assert Path(result["report_path"]).exists()


def test_explicit_override_beats_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(ENV_OUTPUT_DIR, str(tmp_path / "env_dir"))
    assert artifact_output_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_llm_facing_tools_expose_no_filesystem_choices():
    """The language model must not determine artifact locations or report identity."""
    seg_params = inspect.signature(agent_tools.run_segmentation).parameters
    assert "output_dir" not in seg_params
    report_params = inspect.signature(agent_tools.create_report).parameters
    assert "output_dir" not in report_params
    assert "sample_id" not in report_params


def test_agent_create_report_derives_sample_id(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(ENV_OUTPUT_DIR, str(tmp_path))
    qc = {"status": "ok", "image_path": "demo_inputs/synthetic_science.png"}
    result = agent_tools.create_report(qc, [], {"action": "ACCEPT", "reason": "x"})
    assert result["report"]["sample_id"] == "synthetic_science"
    assert Path(result["report_path"]).name == "synthetic_science.report.json"


def test_agent_report_restores_exact_values_after_llm_truncation(monkeypatch, tmp_path: Path):
    """Numbers truncated in the model round-trip must not reach the canonical report."""
    monkeypatch.setenv(ENV_OUTPUT_DIR, str(tmp_path / "out"))
    image = fixtures.write_fragmented_recoverable_image(tmp_path / "in" / "img.png")

    qc = agent_tools.inspect_image(str(image))
    seg = agent_tools.run_segmentation(str(image), attempt=1)
    topo = agent_tools.audit_topology(seg["mask_path"])

    # Simulate the LLM echoing rounded copies of the evidence back.
    rounded_qc = {**qc, "mean_intensity": round(qc["mean_intensity"], 4)}
    rounded_seg = {**seg, "foreground_fraction": round(seg["foreground_fraction"], 4)}
    rounded_topo = {**topo, "fragmentation_score": round(topo["fragmentation_score"], 2)}
    runs = [{"attempt": 1, "segmentation": rounded_seg, "topology": rounded_topo,
             "decision": {"action": "ACCEPT", "reason": "x"}}]

    report = agent_tools.create_report(rounded_qc, runs, {"action": "ACCEPT", "reason": "x"})["report"]
    assert report["qc"] == qc
    assert report["runs"][0]["segmentation"] == seg
    assert report["runs"][0]["topology"] == topo


def test_agent_report_rejects_unapproved_action():
    result = agent_tools.create_report({"image_path": "x.png"}, [], {"action": "DELETE_DATA"})
    assert result["status"] == "failed"
    assert result["reason"] == "invalid_final_action"


# ---------- exact numerical preservation ----------

QC_EXACT = {
    "status": "ok", "qc_pass": True, "image_path": "in/img.png",
    "width": 256, "height": 256, "mode": "L", "issues": [],
    "mean_intensity": 201.73843383789062,
    "contrast_std": 47.238319396972656,
    "sharpness_score": 103.05146789550781,
}
SEG_EXACT = {
    "status": "ok", "adapter": "demo_dark_structure_v1", "attempt": 1,
    "threshold": 220.0, "mask_path": "out/img.attempt1.mask.png",
    "foreground_fraction": 0.130523681640625,
}
TOPO_EXACT = {
    "status": "ok", "beta_0": 3, "beta_1": 0, "tiny_components": 0,
    "fragmentation_score": 0.0,
    "largest_component_fraction": 0.493804068272153,
    "foreground_pixels": 8554,
}


def _exact_report(tmp_path: Path) -> dict:
    runs = [{"attempt": 1, "segmentation": SEG_EXACT, "topology": TOPO_EXACT,
             "decision": {"action": "ACCEPT", "reason": "qc_and_structural_checks_passed"}}]
    final = {"action": "ACCEPT", "reason": "qc_and_structural_checks_passed"}
    return create_report("img", QC_EXACT, runs, final, str(tmp_path))


def test_report_preserves_exact_numbers_on_disk(tmp_path: Path):
    result = _exact_report(tmp_path)
    on_disk = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert on_disk["qc"] == QC_EXACT
    assert on_disk["runs"][0]["segmentation"] == SEG_EXACT
    assert on_disk["runs"][0]["topology"] == TOPO_EXACT


def test_display_summary_contains_exact_values(tmp_path: Path):
    report = _exact_report(tmp_path)["report"]
    summary = report["display_summary"]
    # Full-precision values, exactly as serialized in the JSON report.
    assert "0.130523681640625" in summary
    assert "0.493804068272153" in summary
    assert "201.73843383789062" in summary
    assert "47.238319396972656" in summary
    assert "beta_0=3" in summary
    assert "beta_1=0" in summary
    assert "foreground_pixels=8554" in summary
    assert "Final action: ACCEPT" in summary


def test_display_summary_is_deterministic(tmp_path: Path):
    r1 = _exact_report(tmp_path)["report"]
    r2 = _exact_report(tmp_path)["report"]
    assert r1["display_summary"] == r2["display_summary"]
    assert build_display_summary(r1) == r1["display_summary"]


# ---------- bounded retry behavior ----------

@pytest.fixture()
def recoverable_image(tmp_path: Path) -> Path:
    return fixtures.write_fragmented_recoverable_image(tmp_path / "in" / "synthetic_fragmented.png")


@pytest.fixture()
def unrecoverable_image(tmp_path: Path) -> Path:
    return fixtures.write_unrecoverable_fragmentation_image(tmp_path / "in" / "synthetic_unrecoverable.png")


def test_retry_then_accept(recoverable_image: Path, tmp_path: Path):
    report = run_local_workflow(str(recoverable_image), str(tmp_path / "out"))
    actions = [run["decision"]["action"] for run in report["runs"]]
    assert actions == ["RETRY", "ACCEPT"]
    assert report["final_decision"]["action"] == "ACCEPT"
    # Attempt 1 really was structurally fragmented; attempt 2 really recovered.
    assert report["runs"][0]["topology"]["fragmentation_score"] > 0.55
    assert report["runs"][1]["topology"]["beta_0"] == 1


def test_retry_then_human_review(unrecoverable_image: Path, tmp_path: Path):
    report = run_local_workflow(str(unrecoverable_image), str(tmp_path / "out"))
    actions = [run["decision"]["action"] for run in report["runs"]]
    assert actions == ["RETRY", "HUMAN_REVIEW"]
    assert report["final_decision"]["action"] == "HUMAN_REVIEW"
    assert report["final_decision"]["reason"] == "anomaly_persisted_after_retry"


def test_retry_limit_is_bounded(unrecoverable_image: Path, tmp_path: Path):
    report = run_local_workflow(str(unrecoverable_image), str(tmp_path / "out"))
    assert len(report["runs"]) <= MAX_SEGMENTATION_ATTEMPTS
    assert report["final_decision"]["action"] != "RETRY"


def test_policy_never_retries_at_or_past_limit():
    qc = {"status": "ok", "qc_pass": True}
    suspicious_seg = {"foreground_fraction": 0.0}
    suspicious_topo = {"fragmentation_score": 1.0, "beta_0": 0}
    for attempt in range(MAX_SEGMENTATION_ATTEMPTS, MAX_SEGMENTATION_ATTEMPTS + 3):
        action = bounded_policy(qc, suspicious_seg, suspicious_topo, attempt)["action"]
        assert action == "HUMAN_REVIEW"


# ---------- deterministic measurements ----------

def test_workflow_is_deterministic(recoverable_image: Path, tmp_path: Path):
    out = tmp_path / "out"
    first = run_local_workflow(str(recoverable_image), str(out))
    second = run_local_workflow(str(recoverable_image), str(out))
    assert first == second
