"""Canonical artifact location control.

The language model must never determine filesystem artifact locations. All
artifact-writing tools resolve their output directory here, in Python, from a
single configurable root.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_OUTPUT_DIR = "TOPOSCOUT_OUTPUT_DIR"
DEFAULT_OUTPUT_DIR = "outputs"


def artifact_output_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the canonical artifact output directory.

    Precedence: explicit Python-side override (trusted callers such as
    run_local.py or tests) > TOPOSCOUT_OUTPUT_DIR environment variable >
    the default ``outputs``. LLM-facing tool schemas never expose this choice.
    """
    if override is not None and str(override).strip():
        return Path(override)
    env_value = os.environ.get(ENV_OUTPUT_DIR, "").strip()
    if env_value:
        return Path(env_value)
    return Path(DEFAULT_OUTPUT_DIR)
