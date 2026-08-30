#!/usr/bin/env python3
"""Deterministic inference engine for the scientific worker.

`to_tensor`, `predict_tiled`, and `filt_count` are EXACT transcriptions of the
validated research implementations in
`experiments/critical_set_ph/src/p8_significance_eval.py` (the sealed T0
evaluation path). They are transcribed rather than imported because the Cloud
Run container vendors only the model-architecture sources, not the experiment
scripts. Equivalence of this transcription against the research path is gated
by `scripts/check_worker_inference_equivalence.py` (exact equality on a real
image) — re-run it if any line here changes.

Attempt semantics are FROZEN (docs/MILESTONE4_DISCOVERY.md, settled findings
S3/S4):
  attempt 1  raw mask, probability threshold 0.5, min_area 0
  attempt 2  the SAME probability map + 500 px significance filter

Nothing here is request-controllable except run_id / image / attempt.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
import torch
from scipy import ndimage

from .model_loader import (
    ADAPTER_NAME, MODEL_VERSION, PROB_THRESHOLD, SIGNIFICANCE_MIN_AREA_PX,
    TILE, STRIDE, NET, build_cloud_model,
)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
OVERLAY_COLOR_BGR = (255, 100, 255)


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))


def predict_tiled(net, img: np.ndarray, dev: torch.device) -> np.ndarray:
    H, W = img.shape[:2]
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)) or [0]
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE)) or [0]
    if ys[-1] + TILE < H: ys.append(max(H - TILE, 0))
    if xs[-1] + TILE < W: xs.append(max(W - TILE, 0))
    for y0 in ys:
        for x0 in xs:
            win = img[y0:y0 + TILE, x0:x0 + TILE]
            h, w = win.shape[:2]
            win = cv2.resize(win, (NET, NET), interpolation=cv2.INTER_LINEAR)
            with torch.no_grad():
                p = torch.sigmoid(net(to_tensor(win)[None].to(dev))).squeeze().cpu().numpy()
            acc[y0:y0 + h, x0:x0 + w] += cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
            cnt[y0:y0 + h, x0:x0 + w] += 1
    return acc / np.maximum(cnt, 1)


def filt_count(mask: np.ndarray, min_area: int):
    lab, k = ndimage.label(mask)
    if k == 0:
        return 0, mask
    sz = np.array(ndimage.sum(mask, lab, range(1, k + 1)))
    keep = np.where(sz >= min_area)[0] + 1
    out = np.isin(lab, keep)
    return len(keep), out


def render_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    overlay = img_bgr.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, OVERLAY_COLOR_BGR, 2)
    return overlay


class InferenceEngine:
    """Process-wide singleton holding the strict-loaded model (CPU)."""

    _lock = threading.Lock()
    _instance: "InferenceEngine | None" = None

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        t0 = time.time()
        self.model, self.loader_report = build_cloud_model()
        self.load_seconds = round(time.time() - t0, 2)

    @classmethod
    def get(cls) -> "InferenceEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def segment(self, img_bgr: np.ndarray, attempt: int,
                cached_prob: np.ndarray | None = None) -> dict[str, Any]:
        """Run one frozen-semantics attempt. Returns prob/mask/overlay + evidence."""
        assert attempt in (1, 2)
        t0 = time.time()
        prob_reused = cached_prob is not None and cached_prob.shape == img_bgr.shape[:2]
        prob = (cached_prob if prob_reused
                else predict_tiled(self.model, img_bgr, self.device).astype(np.float32))

        mask = prob > PROB_THRESHOLD
        min_area = 0
        if attempt >= 2:
            min_area = SIGNIFICANCE_MIN_AREA_PX
            _, mask = filt_count(mask, min_area)
        n_significant, _ = filt_count(mask, SIGNIFICANCE_MIN_AREA_PX)

        return {
            "status": "ok",
            "adapter": ADAPTER_NAME,
            "attempt": attempt,
            "prob": prob,
            "mask": mask,
            "overlay": render_overlay(img_bgr, mask),
            "prob_reused": bool(prob_reused),
            "foreground_fraction": float(mask.mean()),
            "prob_threshold": PROB_THRESHOLD,
            "min_area_px": min_area,
            "n_components_significant": int(n_significant),
            "image_height": int(img_bgr.shape[0]),
            "image_width": int(img_bgr.shape[1]),
            "model_version": MODEL_VERSION,
            "checkpoint_sha256": self.loader_report["checkpoint_sha256"],
            "runtime_seconds": round(time.time() - t0, 2),
        }
