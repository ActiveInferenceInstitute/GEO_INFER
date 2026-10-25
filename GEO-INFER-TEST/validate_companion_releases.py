#!/usr/bin/env python3
"""Validate exact local GEO/GNN/FEP release identities; no network or proof execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from geo_infer_test.companion_releases import validate_companion_releases


def main(argv: list[str] | None = None) -> int:
    """Parse checkout locations and emit the bounded offline identity result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--gnn-root", type=Path, required=True)
    parser.add_argument("--fep-root", type=Path, required=True)
    parser.add_argument("--expected-geo-revision")
    args = parser.parse_args(argv)
    try:
        result = validate_companion_releases(
            args.geo_root,
            args.gnn_root,
            args.fep_root,
            expected_geo_revision=args.expected_geo_revision,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        print(
            json.dumps(
                {
                    "schema": "geo-infer-companion-release-identity-review/v1",
                    "status": "refused_local_identity_or_pin_review",
                }
            )
        )
        return 1
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
