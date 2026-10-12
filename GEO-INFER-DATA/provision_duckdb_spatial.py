#!/usr/bin/env python3
"""Provision and verify the DuckDB Spatial extension for the installed runtime."""

import json
import argparse
import sys
import time
import traceback

from geo_infer_data.utils.duckdb_spatial import provision_spatial_extension


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-timeout", type=float, default=120.0)
    parser.add_argument("--max-extension-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        result = provision_spatial_extension(
            download_timeout=args.download_timeout,
            max_extension_bytes=args.max_extension_bytes,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "diagnostic": str(exc),
            "duration_seconds": time.monotonic() - started,
        }
        for field in ("runtime_version", "platform"):
            if hasattr(exc, field):
                result[field] = getattr(exc, field)
        traceback.print_exc(file=sys.stderr)
        print(json.dumps(result, sort_keys=True))
        return 1
    result["duration_seconds"] = time.monotonic() - started
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
