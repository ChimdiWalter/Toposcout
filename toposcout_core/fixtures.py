"""Deterministic synthetic inputs for exercising the bounded retry policy.

These generators are pure functions of their parameters (no randomness), so
the resulting workflow action sequences are reproducible byte-for-byte.

Design notes against the demo adapter + policy:
- Attempt 1 thresholds at the 28th intensity percentile and drops components
  smaller than max(8, 0.02% of pixels) (13 px at 256x256). The topology audit
  counts components smaller than max(8, 0.05% of pixels) (32 px) as tiny.
- Attempt 2 applies contrast enhancement and a 3x3 median filter, which erodes
  a 4x4 speckle to ~4 px (then dropped) and a 3x3 speckle to ~5 px (dropped).

So:
- fragmented_recoverable: one large blob + many 4x4 speckles. Attempt 1 keeps
  the speckles (>=13 px each) but they audit as tiny -> fragmentation_score
  above the 0.55 policy threshold -> RETRY. Attempt 2's median filter removes
  them, leaving the clean blob -> ACCEPT.
- unrecoverable_fragmentation: only 3x3 speckles. Attempt 1 drops them all
  (each < 13 px) -> empty mask (beta_0 = 0) -> RETRY. Attempt 2 erodes and
  drops them again -> still empty -> HUMAN_REVIEW.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256
BACKGROUND = 235


def _save(gray: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray.astype(np.uint8), mode="L").save(p)
    return p


def fragmented_recoverable_array() -> np.ndarray:
    gray = np.full((SIZE, SIZE), BACKGROUND, dtype=np.uint8)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    blob = (yy - 128) ** 2 + (xx - 128) ** 2 <= 28**2
    gray[blob] = 30
    for y in range(16, SIZE - 8, 40):
        for x in range(16, SIZE - 8, 40):
            # Keep speckles clear of the central blob.
            if (y - 128) ** 2 + (x - 128) ** 2 <= 48**2:
                continue
            gray[y : y + 4, x : x + 4] = 40
    return gray


def unrecoverable_fragmentation_array() -> np.ndarray:
    gray = np.full((SIZE, SIZE), BACKGROUND, dtype=np.uint8)
    for y in range(12, SIZE - 6, 36):
        for x in range(12, SIZE - 6, 36):
            gray[y : y + 3, x : x + 3] = 40
    return gray


def write_fragmented_recoverable_image(path: str | Path) -> Path:
    """Attempt 1 -> RETRY (fragmentation), attempt 2 -> ACCEPT (recovered)."""
    return _save(fragmented_recoverable_array(), path)


def write_unrecoverable_fragmentation_image(path: str | Path) -> Path:
    """Attempt 1 -> RETRY (empty/fragmented), attempt 2 -> HUMAN_REVIEW."""
    return _save(unrecoverable_fragmentation_array(), path)
