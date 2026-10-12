"""Deterministic visualization receipt writer.

Writes a manifest JSON alongside a generated artifact recording the input
hash, H3 version metadata, artifact checks, and accessibility checks.

This is the canonical deterministic pattern used across GEO-INFER
visualization entry points (see ``InteractiveVisualizationEngine``). Every
visualization/report entry point that emits a durable artifact should pair it
with a receipt produced by :func:`write_visualization_receipt` so the output is
auditable and reproducible.

Receipt schema (``geo-infer-*-visualization/v1`` family)::

    {
      "schema_version": "<entry-point-specific>/v1",
      "generated_at": "<iso8601>",
      "input_sha256": "<sha256 of canonical JSON of the input payload>",
      "h3_version": "<installed h3 version, or null when unavailable>",
      "artifacts": [{"path": "<filename>", "bytes": <int>}],
      "accessibility": {
        "nonempty_html": <bool>,      # for .html artifacts
        "has_title": <bool>,          # for .html artifacts
        # OR, for non-html artifacts:
        "nonempty": <bool>,
        "valid_json": <bool>,         # for .json artifacts
        "has_heading": <bool>         # for .md artifacts
      }
    }

The manifest is written to ``<artifact_path>.manifest.json`` (the artifact
suffix is replaced, mirroring the SPACE/PLACE engine behaviour).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Mapping

try:  # pragma: no cover - import availability is environment dependent
    import h3

    _H3_VERSION: str | None = h3.__version__
except Exception:  # pragma: no cover - h3 is optional for non-spatial entry points
    _H3_VERSION = None


def _input_digest(payload: Any) -> str:
    """SHA-256 of the canonical (sorted-keys) JSON encoding of ``payload``."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _accessibility_checks(artifact_path: Path, *, title_marker: str | None) -> dict:
    """Build the accessibility-check sub-document for ``artifact_path``.

    The checks adapt to the artifact type so the receipt stays meaningful for
    HTML dashboards, JSON exports, and Markdown reports alike.
    """
    checks: dict[str, bool] = {}
    size = artifact_path.stat().st_size
    suffix = artifact_path.suffix.lower()

    if suffix == ".html":
        text = artifact_path.read_text(encoding="utf-8", errors="replace")
        checks["nonempty_html"] = size > 0
        checks["has_title"] = bool(title_marker and title_marker in text)
        return checks

    checks["nonempty"] = size > 0
    if suffix == ".json":
        try:
            json.loads(artifact_path.read_text(encoding="utf-8"))
            checks["valid_json"] = True
        except Exception:
            checks["valid_json"] = False
    elif suffix == ".md":
        text = artifact_path.read_text(encoding="utf-8", errors="replace")
        checks["has_heading"] = text.lstrip().startswith("#") or bool(
            title_marker and title_marker in text
        )
    return checks


def write_visualization_receipt(
    *,
    artifact_path: Path,
    input_payload: Any,
    schema_version: str,
    generated_at: str | None = None,
    title_marker: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a deterministic manifest JSON next to ``artifact_path``.

    Args:
        artifact_path: Path to the generated artifact (HTML/JSON/MD). The
            receipt is written to ``artifact_path.with_suffix(".manifest.json")``.
        input_payload: The payload that produced the artifact. Serialized with
            ``json.dumps(sort_keys=True, default=str)`` before hashing so the
            digest is stable across runs for equal inputs.
        schema_version: Entry-point-specific schema tag, e.g.
            ``"geo-infer-place-del-norte-visualization/v1"``.
        generated_at: Explicit generation timestamp (ISO 8601). When omitted
            the current time is used (non-deterministic but auditable).
        title_marker: Substring expected in the artifact for the
            ``has_title``/``has_heading`` accessibility check.
        extra: Optional additional top-level keys merged into the manifest
            (e.g. ``{"location": "Del Norte County, CA"}``).

    Returns:
        The path to the written manifest file.
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact not found: {artifact_path}")

    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "input_sha256": _input_digest(input_payload),
        "h3_version": _H3_VERSION,
        "artifacts": [
            {
                "path": artifact_path.name,
                "bytes": artifact_path.stat().st_size,
            }
        ],
        "accessibility": _accessibility_checks(
            artifact_path, title_marker=title_marker
        ),
    }
    if extra:
        manifest.update(extra)

    manifest_path = artifact_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


__all__ = ["write_visualization_receipt"]
