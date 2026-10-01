"""Output, schema, and manifest helpers for ACT scenario runners."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, UTC
from importlib import metadata
import mimetypes
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from geo_infer_act.runners.contracts import RunConfig


RUN_MANIFEST_SCHEMA_VERSION = "geo-infer-act-run-manifest/v1"
SUITE_MANIFEST_SCHEMA_VERSION = "geo-infer-act-suite-manifest/v1"
FIGURE_ARTIFACT_SCHEMA_VERSION = "geo-infer-act-figure-artifact/v1"
FIGURE_METADATA_SCRIPT_ID = "geo-infer-act-figure-metadata"
GEOSPATIAL_REQUIRED_FILES = {
    "data/h3_cells.csv",
    "data/h3_diagnostics.json",
    "data/h3_cells.geojson",
    "data/pymdp_h3_diagnostics.json",
    "data/pymdp_policy_posteriors.csv",
    "data/spatial_inference_trace.json",
    "data/spatial_research_statistics.json",
    "data/h3_lattice_animation.json",
    "data/h3_cell_diagnostics.csv",
    "data/h3_edge_diagnostics.csv",
}
GEOSPATIAL_REQUIRED_VISUALIZATIONS = {
    "visualizations/h3_cell_metric_map.png",
    "visualizations/free_energy_evolution.png",
    "visualizations/belief_entropy_coherence.png",
    "visualizations/interactive_h3_map.html",
    "visualizations/pymdp_policy_free_energy.html",
    "visualizations/h3_belief_flux_map.html",
    "visualizations/h3_policy_surface.html",
    "visualizations/h3_policy_transitions.html",
    "visualizations/h3_spatial_autocorrelation.html",
    "visualizations/h3_entropy_free_energy_phase.html",
    "visualizations/h3_active_inference_lattice.html",
    "visualizations/spatial_inference_research_report.html",
}
NESTED_H3_REQUIRED_FILES = {
    "data/h3_hierarchy.csv",
    "data/nested_h3_diagnostics.json",
    "data/nested_h3_cell_diagnostics.csv",
    "data/nested_h3_parent_child_diagnostics.csv",
    "data/nested_h3_level_diagnostics.csv",
}
NESTED_H3_REQUIRED_VISUALIZATIONS = {
    "visualizations/nested_h3_level_map.html",
    "visualizations/nested_h3_hierarchy_map.html",
    "visualizations/nested_h3_parent_child_residuals.html",
}


def to_jsonable(value: Any) -> Any:
    """Convert numpy-rich Active Inference diagnostics into JSON values."""
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith(
        "geo_infer_act"
    ):
        return to_jsonable(vars(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def package_version() -> str:
    """Return the installed ACT package version when available."""
    try:
        return metadata.version("geo-infer-act")
    except metadata.PackageNotFoundError:
        return "0.2.0"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for runner metadata."""
    return datetime.now(UTC).isoformat()


def ensure_output_tree(output_dir: Path) -> Path:
    """Create the canonical runner output directory tree."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("data", "analysis", "visualizations", "logs"):
        (output_dir / name).mkdir(exist_ok=True)
    return output_dir


def write_json(path: Path, payload: Any) -> Path:
    """Write a JSON file using ACT's JSON-safe conversion policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
    return path


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write row dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key)) for key in fieldnames})
    return path


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for a generated artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_mime_type(path: Path) -> str:
    """Return a stable MIME type for known ACT artifact files."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".png":
        return "image/png"
    if suffix in {".html", ".htm"}:
        return "text/html"
    if suffix == ".md":
        return "text/markdown"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def artifact_type_for_path(path: Path) -> str:
    """Classify generated output files for manifest enrichment."""
    name = path.name
    suffix = path.suffix.lower()
    if ".metadata." in name:
        return "figure_metadata"
    if ".data." in name:
        return "figure_data"
    if path.parts and path.parts[0] == "visualizations":
        if suffix in {".png", ".html", ".htm", ".gif"}:
            return "visualization"
    if path.parts and path.parts[0] == "data":
        return "data"
    if path.parts and path.parts[0] == "analysis":
        return "analysis"
    if path.parts and path.parts[0] == "logs":
        return "log"
    return "document"


def figure_sidecar_paths(path: Path) -> tuple[Path, Path]:
    """Return metadata and data sidecar paths for a visualization path."""
    metadata_path = path.with_name(f"{path.stem}.metadata.json")
    data_suffix = ".data.csv" if path.suffix.lower() != ".html" else ".data.json"
    data_path = path.with_name(f"{path.stem}{data_suffix}")
    return metadata_path, data_path


def _image_size(path: Path) -> dict[str, int]:
    """Return pixel dimensions for image artifacts when Pillow can read them."""
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as image:
            width, height = image.size
        return {"width_px": int(width), "height_px": int(height)}
    except Exception:
        return {}


def build_figure_metadata(
    config: RunConfig,
    artifact_path: Path,
    title: str,
    description: str,
    alt_text: str,
    plotted_metrics: Sequence[str],
    data_sources: Sequence[str],
    figure_data_path: Path,
    visualization_kind: str,
) -> dict[str, Any]:
    """Build JSON-safe metadata shared by embedded and sidecar figure records."""
    if config.output_dir is None:
        raise ValueError(
            "config.output_dir cannot be None when building figure metadata"
        )
    return {
        "schema_version": FIGURE_ARTIFACT_SCHEMA_VERSION,
        "package": "geo-infer-act",
        "package_version": package_version(),
        "scenario": config.scenario,
        "figure_id": artifact_path.stem,
        "title": title,
        "description": description,
        "alt_text": alt_text,
        "visualization_kind": visualization_kind,
        "generated_at": utc_now(),
        "run_config": config.to_manifest_dict(),
        "plotted_metrics": list(plotted_metrics),
        "data_sources": list(data_sources),
        "figure_path": artifact_path.relative_to(config.output_dir).as_posix(),
        "figure_data_path": figure_data_path.relative_to(config.output_dir).as_posix(),
    }


def _write_figure_sidecars(
    config: RunConfig,
    artifact_path: Path,
    metadata_payload: dict[str, Any],
    plotted_data: Any,
    prefer_csv: bool,
) -> tuple[Path, Path]:
    """Write figure metadata and plotted-data sidecars."""
    if config.output_dir is None:
        raise ValueError(
            "config.output_dir cannot be None when writing figure sidecars"
        )
    metadata_path, data_path = figure_sidecar_paths(artifact_path)
    if (
        prefer_csv
        and isinstance(plotted_data, list)
        and all(isinstance(row, Mapping) for row in plotted_data)
    ):
        write_csv(data_path, plotted_data)
    else:
        if prefer_csv:
            data_path = data_path.with_suffix(".json")
        write_json(data_path, plotted_data)
        metadata_payload["figure_data_path"] = data_path.relative_to(
            config.output_dir
        ).as_posix()
    return metadata_path, data_path


def _finalize_figure_metadata(
    config: RunConfig,
    artifact_path: Path,
    metadata_path: Path,
    data_path: Path,
    metadata_payload: dict[str, Any],
) -> dict[str, Any]:
    """Write final sidecar metadata after the artifact exists on disk."""
    if config.output_dir is None:
        raise ValueError(
            "config.output_dir cannot be None when finalizing figure metadata"
        )
    metadata_payload = dict(metadata_payload)
    metadata_payload.update(
        {
            "size_bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
            "mime_type": artifact_mime_type(artifact_path),
            "figure_metadata_path": metadata_path.relative_to(
                config.output_dir
            ).as_posix(),
            "figure_data_path": data_path.relative_to(config.output_dir).as_posix(),
        }
    )
    metadata_payload.update(_image_size(artifact_path))
    write_json(metadata_path, metadata_payload)
    return metadata_payload


def save_matplotlib_figure_artifact(
    config: RunConfig,
    fig: Any,
    relative_path: str,
    *,
    title: str,
    description: str,
    alt_text: str,
    plotted_metrics: Sequence[str],
    data_sources: Sequence[str],
    plotted_data: Any,
    dpi: int = 150,
) -> Path:
    """Save a Matplotlib figure with embedded metadata and data sidecars."""
    if config.output_dir is None:
        raise ValueError("config.output_dir cannot be None when saving figure artifact")
    artifact_path = config.output_dir / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path, data_path = figure_sidecar_paths(artifact_path)
    metadata_payload = build_figure_metadata(
        config=config,
        artifact_path=artifact_path,
        title=title,
        description=description,
        alt_text=alt_text,
        plotted_metrics=plotted_metrics,
        data_sources=data_sources,
        figure_data_path=data_path,
        visualization_kind="matplotlib-png",
    )
    metadata_path, data_path = _write_figure_sidecars(
        config, artifact_path, metadata_payload, plotted_data, prefer_csv=True
    )
    embedded_metadata = {
        "Title": title,
        "Description": description,
        "Author": "geo-infer-act",
        "Software": "geo-infer-act",
        "geo_infer_act_metadata": json.dumps(
            to_jsonable(metadata_payload), sort_keys=True
        ),
    }
    fig.savefig(
        artifact_path,
        dpi=dpi,
        bbox_inches="tight",
        metadata=embedded_metadata,
    )
    _finalize_figure_metadata(
        config, artifact_path, metadata_path, data_path, metadata_payload
    )
    return artifact_path


def write_html_figure_artifact(
    config: RunConfig,
    relative_path: str,
    html: str,
    *,
    title: str,
    description: str,
    alt_text: str,
    plotted_metrics: Sequence[str],
    data_sources: Sequence[str],
    plotted_data: Any,
) -> Path:
    """Write an HTML visualization with embedded JSON metadata and sidecars."""
    if config.output_dir is None:
        raise ValueError("config.output_dir cannot be None when writing HTML artifact")
    artifact_path = config.output_dir / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path, data_path = figure_sidecar_paths(artifact_path)
    metadata_payload = build_figure_metadata(
        config=config,
        artifact_path=artifact_path,
        title=title,
        description=description,
        alt_text=alt_text,
        plotted_metrics=plotted_metrics,
        data_sources=data_sources,
        figure_data_path=data_path,
        visualization_kind="html",
    )
    metadata_path, data_path = _write_figure_sidecars(
        config, artifact_path, metadata_payload, plotted_data, prefer_csv=False
    )
    metadata_json = json.dumps(to_jsonable(metadata_payload), sort_keys=True)
    metadata_block = (
        f'<script type="application/json" id="{FIGURE_METADATA_SCRIPT_ID}">'
        f"{metadata_json}</script>"
    )
    if "</head>" in html:
        html = html.replace("</head>", f"{metadata_block}</head>", 1)
    else:
        html = f"{metadata_block}\n{html}"
    artifact_path.write_text(html)
    _finalize_figure_metadata(
        config, artifact_path, metadata_path, data_path, metadata_payload
    )
    return artifact_path


def relative_files(output_dir: Path) -> list[dict[str, Any]]:
    """Return generated file metadata relative to an output directory."""
    files = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        relative_path = path.relative_to(output_dir)
        artifact_type = artifact_type_for_path(relative_path)
        record = {
            "path": relative_path.as_posix(),
            "size_bytes": path.stat().st_size,
            "artifact_type": artifact_type,
            "mime_type": artifact_mime_type(path),
            "sha256": sha256_file(path),
        }
        if artifact_type == "visualization":
            metadata_path, data_path = figure_sidecar_paths(path)
            record.update(
                {
                    "figure_metadata_path": metadata_path.relative_to(
                        output_dir
                    ).as_posix(),
                    "figure_data_path": data_path.relative_to(output_dir).as_posix(),
                }
            )
            record.update(_image_size(path))
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                record.update(
                    {
                        "description": metadata.get("description", ""),
                        "data_sources": metadata.get("data_sources", []),
                        "plotted_metrics": metadata.get("plotted_metrics", []),
                        "alt_text": metadata.get("alt_text", ""),
                    }
                )
        files.append(record)
    return files


def validate_generated_outputs(
    output_dir: Path, config: RunConfig, generated_files: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate required data and visualization artifacts for one run."""
    paths = {item["path"] for item in generated_files}
    errors: list[str] = []
    required = {"data/full_history.json", "data/step_metrics.csv"}
    if config.scenario in {"h3", "spatial"}:
        required.update(GEOSPATIAL_REQUIRED_FILES)
        if config.visualizations:
            required.update(GEOSPATIAL_REQUIRED_VISUALIZATIONS)
        if config.parameters.get("nested_h3"):
            required.update(NESTED_H3_REQUIRED_FILES)
            if config.visualizations:
                required.update(NESTED_H3_REQUIRED_VISUALIZATIONS)
    missing = sorted(required - paths)
    if missing:
        errors.extend(f"missing required file: {path}" for path in missing)

    if config.visualizations and not any(
        item.get("artifact_type") == "visualization" for item in generated_files
    ):
        errors.append("missing required visualization artifact")
    if config.visualizations:
        for item in generated_files:
            if item.get("artifact_type") != "visualization":
                continue
            for key in ("figure_metadata_path", "figure_data_path"):
                sidecar = item.get(key)
                if not sidecar:
                    errors.append(f"missing visualization {key}: {item['path']}")
                    continue
                sidecar_path = output_dir / sidecar
                if not sidecar_path.exists() or sidecar_path.stat().st_size <= 0:
                    errors.append(f"missing or empty visualization sidecar: {sidecar}")
            for key in ("sha256", "mime_type", "description", "data_sources"):
                if not item.get(key):
                    errors.append(f"missing visualization {key}: {item['path']}")

    empty = [
        item["path"]
        for item in generated_files
        if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] <= 0
    ]
    if empty:
        errors.extend(f"empty generated file: {path}" for path in empty)

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "checked_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
    }


def write_run_manifest(
    output_dir: Path,
    config: RunConfig,
    metrics: dict[str, Any],
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write ``manifest.json`` for one scenario run."""
    generated_files = relative_files(output_dir)
    validation = validate_generated_outputs(output_dir, config, generated_files)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "package": "geo-infer-act",
        "package_version": package_version(),
        "scenario": config.scenario,
        "config": config.to_manifest_dict(),
        "command": command or [],
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_files": generated_files,
        "metrics": to_jsonable(metrics),
        "validation": validation,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def write_suite_manifest(
    output_dir: Path,
    results: list[Any],
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write a suite-level manifest for ``run_all_scenarios``."""
    scenarios = [
        {
            "scenario": result.scenario,
            "output_dir": str(result.output_dir),
            "manifest": str(result.manifest_path),
            "status": result.manifest.get("validation", {}).get("status"),
            "metrics": result.metrics,
        }
        for result in results
    ]
    errors = [item for item in scenarios if item.get("status") != "passed"]
    manifest = {
        "schema_version": SUITE_MANIFEST_SCHEMA_VERSION,
        "package": "geo-infer-act",
        "package_version": package_version(),
        "generated_at": datetime.now(UTC).isoformat(),
        "command": command or [],
        "scenarios": to_jsonable(scenarios),
        "validation": {
            "status": "failed" if errors else "passed",
            "errors": errors,
        },
    }
    write_json(output_dir / "suite_manifest.json", manifest)
    return manifest
