from pathlib import Path

import numpy as np
from PIL import Image

from toposcout_core.components import connected_components, hole_count
from toposcout_core.tools import audit_topology, bounded_policy, inspect_image


def test_components_and_hole_count():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:8, 2:8] = True
    mask[12:18, 12:18] = True
    assert connected_components(mask).count == 2

    ring = np.zeros((20, 20), dtype=bool)
    ring[2:18, 2:18] = True
    ring[6:14, 6:14] = False
    assert hole_count(ring) == 1


def test_qc_rejects_flat_black(tmp_path: Path):
    p = tmp_path / "black.png"
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(p)
    qc = inspect_image(str(p))
    assert qc["qc_pass"] is False
    assert "severely_underexposed" in qc["issues"]


def test_policy_retries_fragmented_result():
    qc = {"status": "ok", "qc_pass": True}
    seg = {"foreground_fraction": 0.1}
    topo = {"fragmentation_score": 0.9, "beta_0": 20}
    assert bounded_policy(qc, seg, topo, 1)["action"] == "RETRY"
    assert bounded_policy(qc, seg, topo, 2)["action"] == "HUMAN_REVIEW"
