"""INDUSTRIAL-INSPECTION PORTABILITY PILOT — NO MANUFACTURING ACCEPTANCE CLAIMS.

Fixed public method: Anomalib PatchCore (memory-bank anomaly localization,
default configuration) on one MVTec AD category. MVTec AD is CC BY-NC-SA
(noncommercial — flagged for review before publishing demo assets). PatchCore
fits its memory bank from normal training images only; no gradient training.

Frozen pilot parameters: category "bottle", anomalib default PatchCore
(WideResNet-50 features, default coreset ratio), anomalib's own computed
threshold for the binary anomaly mask (the method's documented decision rule),
CPU. No recovery rule — suspicious structure escalates.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..base import PilotAdapter
from ..profiles import INDUSTRIAL_DEFECT

CATEGORY = "bottle"
DATA_ROOT_ENV = "TOPOSCOUT_MVTEC_ROOT"  # trusted operator config, not LLM-facing


class PatchCorePilot(PilotAdapter):
    domain = "industrial"
    adapter_name = "industrial_patchcore_v1"
    model_name = f"Anomalib PatchCore (default config) on MVTec AD '{CATEGORY}'"
    model_source = "https://github.com/open-edge-platform/anomalib"
    model_license = ("anomalib Apache-2.0; MVTec AD dataset CC BY-NC-SA 4.0 "
                     "(noncommercial — flagged)")
    profile = INDUSTRIAL_DEFECT
    limitations = ("Portability pilot only — no manufacturing acceptance claims; "
                   "memory bank fitted on the category's normal images at run time; "
                   "binary mask uses anomalib's computed threshold; no recovery rule.")

    def _predict_mask(self, image_path: Path, out_dir: Path) -> dict[str, Any]:
        import cv2
        import torch
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        root = os.environ.get(DATA_ROOT_ENV, "").strip() or str(
            Path(tempfile.gettempdir()) / "toposcout_mvtec")
        dm = MVTecAD(root=Path(root) / "MVTecAD", category=CATEGORY,
                     train_batch_size=8, eval_batch_size=8, num_workers=0)
        model = Patchcore()
        engine = Engine(accelerator="cpu", devices=1, logger=False,
                        default_root_dir=str(out_dir / "anomalib_runs"))
        engine.fit(model, datamodule=dm)

        preds = engine.predict(model, data_path=str(image_path))
        assert preds, "no prediction returned"
        item = preds[0]
        pred_mask = np.asarray(item.pred_mask.squeeze().cpu().numpy()
                               if torch.is_tensor(item.pred_mask) else item.pred_mask)
        amap = np.asarray(item.anomaly_map.squeeze().cpu().numpy()
                          if torch.is_tensor(item.anomaly_map) else item.anomaly_map)

        bgr = cv2.imread(str(image_path))
        if pred_mask.shape != bgr.shape[:2]:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), bgr.shape[1::-1],
                                   interpolation=cv2.INTER_NEAREST)
        mask = pred_mask.astype(bool)
        heat = cv2.applyColorMap(
            cv2.normalize(cv2.resize(amap, bgr.shape[1::-1]), None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)

        amap_path = out_dir / "anomaly_map.npy"
        np.save(amap_path, amap)
        return {"mask": mask, "overlay_bgr": overlay,
                "anomaly_map_path": str(amap_path)}
