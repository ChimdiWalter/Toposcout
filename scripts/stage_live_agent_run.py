#!/usr/bin/env python3
"""Trusted operator staging for a live deployed-agent run (M6 Phase 1).

Creates the Firestore run document and uploads one authorized input image to
the canonical GCS prefix runs/<run_id>/input/. Prints ONLY {run_id, image_uri}
as JSON — the exact payload the deployed Gemini agent contract accepts.

Never invokes Gemini. Operator credentials only.

Usage:
    python scripts/stage_live_agent_run.py --image /path/to/leaf.png \
        [--run-id live-...]  [--sample-id DSC_0059]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402

os.environ.setdefault("TOPOSCOUT_GCS_BUCKET", "toposcout-agent-runs")
os.environ.setdefault("TOPOSCOUT_RUNSTATE", "firestore")

from scientific_worker import storage  # noqa: E402
from toposcout_core import runstate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="authorized local input image")
    ap.add_argument("--run-id", default=None, help="optional; generated if omitted")
    ap.add_argument("--sample-id", default=None, help="optional; defaults to image stem")
    args = ap.parse_args()

    image = Path(args.image)
    if not image.is_file():
        raise SystemExit(f"input image not found: {image}")
    if image.suffix.lower() not in storage.ALLOWED_IMAGE_SUFFIXES:
        raise SystemExit(f"unsupported image type: {image.suffix}")

    run_id = args.run_id or f"live-{time.strftime('%Y%m%d-%H%M%S')}-{image.stem[:24]}"
    if not storage.RUN_ID_RE.match(run_id):
        raise SystemExit(f"invalid run_id: {run_id!r}")
    sample_id = args.sample_id or image.stem

    image_uri = storage.upload_input(run_id, image)
    storage.validate_run_input(run_id, image_uri)  # staged object must satisfy the contract
    runstate.create_run(run_id, sample_id, image_uri=image_uri)

    print(json.dumps({"run_id": run_id, "image_uri": image_uri}))


if __name__ == "__main__":
    main()
