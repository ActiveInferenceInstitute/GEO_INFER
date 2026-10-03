#!/usr/bin/env python3
"""Check one retained wheel attempt against the live package/profile registry."""

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from geo_infer_test.wheel_evidence import validate_wheel_receipt
from build_package_wheels import module_dirs, REQUIRED_PROFILES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--directory", type=Path)
    selection.add_argument(
        "--receipt", type=Path, help="Select one retained attempt explicitly"
    )
    parser.add_argument("--revision")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-imports", action="store_true")
    parser.add_argument("--require-profiles", action="store_true")
    args = parser.parse_args()
    paths = (
        [args.receipt]
        if args.receipt
        else sorted((args.directory / "attempts").glob("*/receipt.json"))
    )
    try:
        if len(paths) != 1:
            raise ValueError(
                "Select an output directory containing exactly one wheel attempt"
            )
        validate_wheel_receipt(
            paths[0],
            modules=[module.name for module in module_dirs()],
            profiles=[
                (
                    profile.package,
                    ",".join(profile.extras) or "base",
                    hashlib.sha256(profile.code.encode()).hexdigest(),
                )
                for profile in REQUIRED_PROFILES
            ],
            revision=args.revision,
            require_clean=args.require_clean,
            require_imports=args.require_imports,
            require_profiles=args.require_profiles,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Invalid wheel evidence: {exc}", file=sys.stderr)
        return 1
    print("Wheel source, inventories, completion receipts and artifact hashes: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
