"""COMPUTATIONAL-PATHOLOGY PORTABILITY PILOT — NOT FOR CLINICAL USE.

Fixed public model: TIAToolbox pretrained HoVer-Net ``hovernet_fast-pannuke``
(nucleus instance segmentation, trained on PanNuke). TIAToolbox code is
BSD-style; the PanNuke-trained weights are CC BY-NC-SA — noncommercial terms
flagged for review before publishing demo assets. NOT diagnostic; no tumor
claims.

Frozen pilot parameters: hovernet_fast-pannuke, tile mode, CPU, model's own
instance decoding; instances flattened to a binary nucleus mask for the
structural audit. No recovery rule — suspicious structure escalates.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..base import PilotAdapter
from ..profiles import PATHOLOGY_NUCLEI

PRETRAINED = "hovernet_fast-pannuke"


class HoverNetPilot(PilotAdapter):
    domain = "pathology"
    adapter_name = "pathology_hovernet_v1"
    model_name = f"TIAToolbox HoVer-Net ({PRETRAINED})"
    model_source = "https://github.com/TissueImageAnalytics/tiatoolbox"
    model_license = ("TIAToolbox code permissive; PanNuke HoVer-Net weights CC BY-NC-SA "
                     "(noncommercial — flagged); input: TIAToolbox sample tissue tile")
    profile = PATHOLOGY_NUCLEI
    limitations = ("Portability pilot only — NOT for clinical use, no diagnostic claims; "
                   "instances flattened to a binary mask; no recovery rule.")

    def _predict_mask(self, image_path: Path, out_dir: Path) -> dict[str, Any]:
        import cv2
        from tiatoolbox.models.engine.nucleus_instance_segmentor import (
            NucleusInstanceSegmentor,
        )

        segmentor = NucleusInstanceSegmentor(model=PRETRAINED, batch_size=2,
                                             num_workers=0, device="cpu")
        # tiatoolbox 2.x patch mode returns an in-memory dict:
        #   predictions: (n, h, w) instance-id map (model output resolution)
        #   contours/box/centroid/prob/type: per-image instance lists
        output = segmentor.run([str(image_path)], patch_mode=True)
        inst_map = np.asarray(output["predictions"][0])
        contours = output["contours"][0] if output.get("contours") else []
        n_instances = int(max(len(contours), inst_map.max()))

        bgr = cv2.imread(str(image_path))
        mask = cv2.resize((inst_map > 0).astype(np.uint8), bgr.shape[1::-1],
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = bgr.copy()
        cnts, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 80, 255), 1)
        return {"mask": mask, "overlay_bgr": overlay, "n_instances": n_instances}
