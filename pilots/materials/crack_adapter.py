"""MATERIALS / CRACK PORTABILITY PILOT — NOT A SAFETY CERTIFICATION SYSTEM.

Fixed public model: rievil/crackenpy model1 (Hugging Face) — a
segmentation-models-pytorch network trained on the CrackenPy building-material
dataset (BSD v2 licensed for research/education). Classes include background,
matrix, crack, pore; this pilot audits the CRACK class mask.

Frozen pilot parameters: model1.pt, 416x416 tiling with border anchoring,
argmax class assignment (the model's own decision rule), crack class only,
no post-processing, no recovery rule.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..base import PilotAdapter
from ..profiles import MATERIALS_CRACK

HF_REPO = "rievil/crackenpy"
TILE = 416
# crackenpy label order (crackest.cracks.list_labels): back, spec, mat, crack, pore
N_CLASSES = 5
CRACK_CLASS = 3
ENCODER = "resnext50_32x4d"  # model1.json model_type; built as smp.FPN per crackenpy


class CrackSegmentationPilot(PilotAdapter):
    domain = "materials"
    adapter_name = "materials_crackenpy_v1"
    model_name = "CrackenPy model1 (segmentation-models-pytorch, crack class)"
    model_source = f"https://huggingface.co/{HF_REPO}"
    model_license = "BSD v2 (CrackenPy model + dataset, research/education use)"
    profile = MATERIALS_CRACK
    limitations = ("Portability pilot only — structural-integrity visualization, not a "
                   "safety certification; fixed public checkpoint; crack class only; "
                   "no recovery rule.")

    def _predict_mask(self, image_path: Path, out_dir: Path) -> dict[str, Any]:
        import cv2
        import segmentation_models_pytorch as smp
        import torch
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(HF_REPO, "model1.pt")
        meta_path = Path(hf_hub_download(HF_REPO, "model1.json"))
        meta = json.loads(meta_path.read_text())
        assert meta["model_type"] == ENCODER and meta["num_classes"] == N_CLASSES

        model = smp.FPN(ENCODER, in_channels=3, classes=N_CLASSES,
                        activation=None, encoder_depth=5)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model = model.eval()

        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]
        # normalization per smp imagenet convention
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        rgb = (rgb - mean) / std

        classes = np.zeros((H, W), np.int64)
        ys = list(range(0, max(H - TILE, 0) + 1, TILE))
        xs = list(range(0, max(W - TILE, 0) + 1, TILE))
        if not ys or ys[-1] + TILE < H:
            ys.append(max(H - TILE, 0))
        if not xs or xs[-1] + TILE < W:
            xs.append(max(W - TILE, 0))
        with torch.no_grad():
            for y0 in ys:
                for x0 in xs:
                    win = rgb[y0:y0 + TILE, x0:x0 + TILE]
                    h, w = win.shape[:2]
                    if (h, w) != (TILE, TILE):
                        win = cv2.resize(win, (TILE, TILE))
                    t = torch.from_numpy(win.transpose(2, 0, 1))[None]
                    logits = model(t)
                    pred = logits.argmax(dim=1).squeeze().numpy().astype(np.int64)
                    if (h, w) != (TILE, TILE):
                        pred = cv2.resize(pred.astype(np.uint8), (w, h),
                                          interpolation=cv2.INTER_NEAREST).astype(np.int64)
                    classes[y0:y0 + h, x0:x0 + w] = pred

        mask = classes == CRACK_CLASS
        overlay = bgr.copy()
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
        return {"mask": mask, "overlay_bgr": overlay}
