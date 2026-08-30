"""MICROSCOPY PORTABILITY PILOT — NOT BIOLOGICALLY VALIDATED.

Fixed public model: Cellpose 4 (Cellpose-SAM) generalist cell segmentation
(MouseLand/cellpose). Code BSD-3; the Cellpose team notes its models are
trained on data under CC-BY-NC terms — flagged for review before publishing
demo assets.

Frozen pilot parameters: default CellposeModel weights, default diameter
estimation, CPU. No retry/recovery rule is defined for this domain —
suspicious structure escalates to HUMAN_REVIEW by construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..base import PilotAdapter
from ..profiles import MICROSCOPY_CELLS


class CellposePilot(PilotAdapter):
    domain = "microscopy"
    adapter_name = "microscopy_cellpose_v1"
    model_name = "Cellpose 4 (Cellpose-SAM generalist, default weights)"
    model_source = "https://github.com/MouseLand/cellpose"
    model_license = ("code BSD-3; model weights trained on CC-BY-NC data (per Cellpose "
                     "docs) — noncommercial terms flagged; input: cellpose.org sample image")
    profile = MICROSCOPY_CELLS
    limitations = ("Portability pilot only — not biologically validated; instance masks "
                   "flattened to a binary mask for the structural audit; no recovery rule.")

    def _predict_mask(self, image_path: Path, out_dir: Path) -> dict[str, Any]:
        import cv2
        from cellpose import models

        img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        model = models.CellposeModel(gpu=False)
        masks, flows, styles = model.eval(img)

        inst = np.asarray(masks)
        binary = inst > 0
        overlay = cv2.imread(str(image_path))
        contours, _ = cv2.findContours((binary.astype(np.uint8)) * 255,
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 80, 255), 1)
        return {"mask": binary, "overlay_bgr": overlay,
                "n_instances": int(inst.max())}
