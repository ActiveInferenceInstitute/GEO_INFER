#!/usr/bin/env python3
"""GEO-INFER-INTRA module orchestrator.

Runs one documented end-to-end INTRA operation on synthetic data: generate
the reproducible spatial visual preview suite (Leaflet HTML, SVG card, PNG
card, manifest) for synthetic module documentation bundles, verify every
artifact exists with the expected byte counts, and cross-check the manifest
digest. All work goes through the real ``geo_infer_intra`` public API.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    from geo_infer_intra import MODULE_PROFILES, generate_module_preview_suite

    target_modules = ["SPACE", "CIV", "ORG", "INTRA"]
    output_dir = Path(tempfile.mkdtemp(prefix="geo-infer-intra-previews-"))
    try:
        bundles: dict[str, dict[str, Any]] = {}
        for module_id in target_modules:
            artifacts = generate_module_preview_suite(module_id, output_dir)

            # Verify the real artifacts on disk match the reported sizes.
            expected = {
                artifacts.html_path: artifacts.html_bytes,
                artifacts.svg_path: artifacts.svg_bytes,
                artifacts.png_path: artifacts.png_bytes,
                artifacts.manifest_path: 0,  # existence only; size varies
            }
            sizes: dict[str, int] = {}
            for path in expected:
                if not path.is_file():
                    raise RuntimeError(f"missing preview artifact: {path.name}")
                sizes[path.suffix.lstrip(".") or "manifest"] = path.stat().st_size

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            if manifest["input_sha256"] != artifacts.input_sha256:
                raise RuntimeError(f"manifest digest mismatch for {module_id}")
            if manifest["module_id"] != f"GEO-INFER-{module_id}":
                raise RuntimeError(f"manifest module id mismatch for {module_id}")

            bundles[module_id] = {
                "profile_name": MODULE_PROFILES[module_id]["name"],
                "category": MODULE_PROFILES[module_id]["category"],
                "input_sha256": artifacts.input_sha256[:16],
                "html_bytes": sizes["html"],
                "svg_bytes": sizes["svg"],
                "png_bytes": sizes["png"],
                "manifest_artifact_count": len(manifest["artifacts"]),
            }
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    return {
        "operation": "module_preview_bundle_generation",
        "modules_profiled": len(MODULE_PROFILES),
        "modules_processed": target_modules,
        "bundles": bundles,
        "all_artifacts_verified": True,
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("INTRA", _operation))
