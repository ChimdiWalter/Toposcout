"""Request/response schemas for the scientific worker.

The request surface is deliberately minimal and CLOSED (`extra="forbid"`):
run_id, image_uri, attempt. A request cannot name a checkpoint, interpreter,
executable, architecture, threshold, min-area, shell command, or output
location — those are fixed, trusted server-side configuration.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class SegmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Opaque run identifier (also the GCS run prefix)")
    image_uri: str = Field(..., description="gs:// URI inside the trusted TopoScout bucket "
                                            "(local paths only in TOPOSCOUT_SW_ALLOW_LOCAL_IO dev mode)")
    attempt: int = Field(..., description="1 = raw T0 mask at 0.5; 2 = same prob map + 500 px significance filter")

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, v: str) -> str:
        if not RUN_ID_RE.match(v):
            raise ValueError("run_id must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
        return v

    @field_validator("attempt")
    @classmethod
    def _attempt(cls, v: int) -> int:
        if v not in (1, 2):
            raise ValueError("attempt must be 1 or 2")
        return v


class SegmentResponse(BaseModel):
    status: str
    adapter: str
    run_id: str
    attempt: int
    mask_uri: str
    overlay_uri: str
    prob_uri: str | None = None
    prob_reused: bool
    foreground_fraction: float
    prob_threshold: float
    min_area_px: int
    n_components_significant: int
    image_height: int
    image_width: int
    model_version: str
    checkpoint_sha256: str
    runtime_seconds: float


class WorkerError(BaseModel):
    status: str = "failed"
    adapter: str = "real_lesion_model_v1"
    reason: str
    detail: str | None = None
