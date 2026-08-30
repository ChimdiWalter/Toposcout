from __future__ import annotations

import argparse
import json

from toposcout_core import run_local_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic TopoScout MVP vertical slice.")
    parser.add_argument("image")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the canonical artifact directory (default: TOPOSCOUT_OUTPUT_DIR or 'outputs').",
    )
    args = parser.parse_args()
    report = run_local_workflow(args.image, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
