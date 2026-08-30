"""SATELLITE / REMOTE-SENSING PORTABILITY PILOT — NOT VALIDATED FOR GIS USE.

Fixed public model: spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp
(Hugging Face, MIT license) — Keras U-Net trained on the Massachusetts Roads
Dataset at 256x256. Reported by its model card at ~71% IoU/Dice accuracy.

Frozen pilot parameters (pre-registered before viewing any output):
    tile 256, stride 256 (edge tiles anchored to the border),
    probability threshold 0.5, no post-processing, no recovery rule.
Suspicious structure escalates to HUMAN_REVIEW by construction — this pilot
defines no validated ACCEPT threshold.

Runs in the dedicated TF pilot venv (~/.venvs/pilotenv_tf); the maize
production environment is never touched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..base import PilotAdapter
from ..profiles import SATELLITE_ROADS

HF_REPO = "spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp"
HF_FILE = "aerial-image-road-segmentation-xp.keras"
TILE = 256
PROB_THRESHOLD = 0.5  # frozen before evaluation


class RoadSegmentationPilot(PilotAdapter):
    domain = "satellite"
    adapter_name = "satellite_road_unet_v1"
    model_name = "aerial-image-road-segmentation-with-U-NET-xp (Keras U-Net, Massachusetts Roads)"
    model_source = f"https://huggingface.co/{HF_REPO}"
    model_license = "MIT (model); input tile: Massachusetts Roads test set (Mnih 2013)"
    profile = SATELLITE_ROADS
    limitations = ("Portability pilot only — not validated for operational GIS/road "
                   "mapping. Fixed public checkpoint, frozen 0.5 threshold, no "
                   "topology-aware training, no recovery rule.")

    def _predict_mask(self, image_path: Path, out_dir: Path) -> dict[str, Any]:
        import cv2
        import keras
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(HF_REPO, HF_FILE)
        model = keras.models.load_model(weights, compile=False)

        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]

        prob = np.zeros((H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)
        ys = list(range(0, max(H - TILE, 0) + 1, TILE))
        xs = list(range(0, max(W - TILE, 0) + 1, TILE))
        if ys[-1] + TILE < H:
            ys.append(H - TILE)
        if xs[-1] + TILE < W:
            xs.append(W - TILE)
        batch, coords = [], []
        for y0 in ys:
            for x0 in xs:
                batch.append(rgb[y0:y0 + TILE, x0:x0 + TILE])
                coords.append((y0, x0))
        preds = model.predict(np.stack(batch), verbose=0)
        preds = np.squeeze(preds, axis=-1) if preds.ndim == 4 else preds
        for (y0, x0), p in zip(coords, preds):
            prob[y0:y0 + TILE, x0:x0 + TILE] += p
            cnt[y0:y0 + TILE, x0:x0 + TILE] += 1
        prob /= np.maximum(cnt, 1)

        mask = prob > PROB_THRESHOLD
        overlay = bgr.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 80, 255])).astype(np.uint8)

        prob_path = out_dir / "prob.npy"
        np.save(prob_path, prob)
        return {"mask": mask, "overlay_bgr": overlay, "prob_path": str(prob_path)}
