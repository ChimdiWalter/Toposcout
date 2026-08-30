from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ComponentStats:
    count: int
    areas: list[int]


def _neighbors(y: int, x: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if y > 0:
        yield y - 1, x
    if y + 1 < h:
        yield y + 1, x
    if x > 0:
        yield y, x - 1
    if x + 1 < w:
        yield y, x + 1


def connected_components(mask: np.ndarray) -> ComponentStats:
    """Count 4-connected foreground components and return their areas."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                for ny, nx in _neighbors(cy, cx, h, w):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            areas.append(area)
    return ComponentStats(count=len(areas), areas=areas)


def hole_count(mask: np.ndarray) -> int:
    """Approximate beta_1 by counting enclosed 4-connected background regions."""
    fg = np.asarray(mask, dtype=bool)
    bg = ~fg
    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    # Mark all background connected to the image border as exterior.
    for x in range(w):
        if bg[0, x] and not seen[0, x]:
            seen[0, x] = True; q.append((0, x))
        if bg[h - 1, x] and not seen[h - 1, x]:
            seen[h - 1, x] = True; q.append((h - 1, x))
    for y in range(h):
        if bg[y, 0] and not seen[y, 0]:
            seen[y, 0] = True; q.append((y, 0))
        if bg[y, w - 1] and not seen[y, w - 1]:
            seen[y, w - 1] = True; q.append((y, w - 1))

    while q:
        cy, cx = q.popleft()
        for ny, nx in _neighbors(cy, cx, h, w):
            if bg[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))

    holes = 0
    for y in range(h):
        for x in range(w):
            if not bg[y, x] or seen[y, x]:
                continue
            holes += 1
            seen[y, x] = True
            q.append((y, x))
            while q:
                cy, cx = q.popleft()
                for ny, nx in _neighbors(cy, cx, h, w):
                    if bg[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
    return holes


def save_mask(mask: np.ndarray, output_path: str | Path) -> None:
    out = (np.asarray(mask, dtype=np.uint8) * 255)
    Image.fromarray(out, mode="L").save(output_path)
