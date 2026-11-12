#!/usr/bin/env python3
"""Report the reviewed GNN interchange pin's release lineage and freshness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from geo_infer_test.gnn_pin_freshness import (
    read_pin_report,
    validate_gnn_pin_freshness,
)


def main(argv: list[str] | None = None) -> int:
    """Parse checkout locations and emit the bounded freshness result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--gnn-root", type=Path, required=True)
    parser.add_argument(
        "--pin-report",
        type=Path,
        required=True,
        help=(
            "JSON from `python -m gnn.importers check --repo . --json`, run in "
            "the environment built from the pinned revision"
        ),
    )
    parser.add_argument("--gnn-ref", default="HEAD")
    parser.add_argument("--require-current", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_gnn_pin_freshness(
            args.geo_root,
            args.gnn_root,
            pin=read_pin_report(args.pin_report),
            gnn_ref=args.gnn_ref,
            require_current=args.require_current,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        print(
            json.dumps(
                {
                    "schema": "geo-infer-gnn-pin-freshness/v1",
                    "status": "refused_pin_freshness_review",
                }
            )
        )
        return 1
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
