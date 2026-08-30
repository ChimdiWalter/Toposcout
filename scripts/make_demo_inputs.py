"""Regenerate the deterministic demo inputs for the bounded-retry demos.

Usage: uv run python scripts/make_demo_inputs.py
"""
from __future__ import annotations

from pathlib import Path

from toposcout_core.fixtures import (
    write_fragmented_recoverable_image,
    write_unrecoverable_fragmentation_image,
)

DEMO_INPUTS = Path(__file__).resolve().parent.parent / "demo_inputs"


def main() -> None:
    p1 = write_fragmented_recoverable_image(DEMO_INPUTS / "synthetic_fragmented.png")
    p2 = write_unrecoverable_fragmentation_image(DEMO_INPUTS / "synthetic_unrecoverable.png")
    print(f"wrote {p1}  (expected: attempt1 RETRY -> attempt2 ACCEPT)")
    print(f"wrote {p2}  (expected: attempt1 RETRY -> attempt2 HUMAN_REVIEW)")


if __name__ == "__main__":
    main()
