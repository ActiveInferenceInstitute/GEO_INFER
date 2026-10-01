"""Generate the GEO-INFER-ACT spatial active-inference visualization gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Iterable

from geo_infer_act.runners import run_spatial_active_inference_gallery


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "output" / "spatial_active_inference_gallery"
)


def main(argv: Iterable[str] | None = None) -> int:
    """Run the package-owned gallery generator."""
    parser = argparse.ArgumentParser(
        description="Generate real-H3 + pymdp spatial active-inference gallery."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--h3-resolution", type=int, default=8)
    parser.add_argument("--h3-ring-size", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = run_spatial_active_inference_gallery(
        args.output_dir,
        seed=args.seed,
        timesteps=args.timesteps,
        h3_resolution=args.h3_resolution,
        h3_ring_size=args.h3_ring_size,
    )
    payload = {
        "gallery": str(args.output_dir / "index.html"),
        "manifest": str(args.output_dir / "gallery_manifest.json"),
        "runs": len(manifest["runs"]),
    }
    print(json.dumps(payload) if args.json else f"Gallery: {payload['gallery']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
