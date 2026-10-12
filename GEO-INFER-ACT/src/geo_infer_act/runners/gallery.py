"""Deterministic visualization gallery for real H3 + pymdp ACT runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geo_infer_act.runners.contracts import RunConfig, ScenarioRunResult
from geo_infer_act.runners.io import ensure_output_tree, to_jsonable, utc_now
from geo_infer_act.runners.scenarios import run_scenario


GALLERY_SCHEMA_VERSION = "geo-infer-act-spatial-gallery/v1"


def run_spatial_active_inference_gallery(
    output_dir: Path,
    *,
    seed: int = 73,
    timesteps: int = 4,
    h3_resolution: int = 8,
    h3_ring_size: int = 1,
) -> dict[str, Any]:
    """Generate the four-run spatial active-inference visualization gallery."""
    output_dir = ensure_output_tree(Path(output_dir))
    scenarios = [
        ("h3", "h3", {"research_profile": True}),
        ("h3_nested", "h3", {"research_profile": True, "nested_h3": True}),
        ("spatial", "spatial", {"research_profile": True}),
        (
            "spatial_nested",
            "spatial",
            {"research_profile": True, "nested_h3": True},
        ),
    ]
    results: list[ScenarioRunResult] = []
    for index, (gallery_name, scenario, parameters) in enumerate(scenarios):
        result = run_scenario(
            RunConfig(
                scenario=scenario,
                output_dir=output_dir / gallery_name,
                seed=seed + index,
                deterministic=True,
                timesteps=timesteps,
                visualizations=True,
                h3_resolution=h3_resolution,
                h3_ring_size=h3_ring_size,
                parameters=dict(parameters),
            ),
            command=[
                "geo-infer-act-spatial-gallery",
                "--scenario",
                gallery_name,
            ],
        )
        results.append(result)

    gallery_manifest = _build_gallery_manifest(output_dir, results)
    (output_dir / "gallery_manifest.json").write_text(
        json.dumps(to_jsonable(gallery_manifest), indent=2, sort_keys=True)
    )
    (output_dir / "index.html").write_text(_build_gallery_index(gallery_manifest))
    return gallery_manifest


def _build_gallery_manifest(
    output_dir: Path, results: list[ScenarioRunResult]
) -> dict[str, Any]:
    """Build a JSON-safe gallery manifest from scenario manifests."""
    entries = []
    for result in results:
        manifest = result.manifest
        visualizations = [
            {
                "path": item["path"],
                "title": item.get("description", item["path"]),
                "href": f"{result.output_dir.name}/{item['path']}",
                "metrics": item.get("plotted_metrics", []),
            }
            for item in manifest.get("generated_files", [])
            if item.get("artifact_type") == "visualization"
        ]
        entries.append(
            {
                "name": result.output_dir.name,
                "scenario": result.scenario,
                "manifest": f"{result.output_dir.name}/manifest.json",
                "status": manifest.get("validation", {}).get("status"),
                "metrics": manifest.get("metrics", {}),
                "visualizations": visualizations,
            }
        )
    return {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "runs": entries,
    }


def _build_gallery_index(gallery_manifest: dict[str, Any]) -> str:
    """Build the static gallery HTML index."""
    cards = "\n".join(_run_card(run) for run in gallery_manifest["runs"])
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>GEO-INFER-ACT Spatial Active Inference Gallery</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1f2328; }}
    header {{ max-width: 1040px; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
    section {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 14px 16px; background: #ffffff; }}
    h1 {{ margin-bottom: 6px; }}
    h2 {{ margin: 0 0 10px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    a {{ color: #0969da; }}
  </style>
</head>
<body>
  <header>
    <h1>GEO-INFER-ACT Spatial Active Inference Gallery</h1>
    <p>Deterministic real-H3 simulations using inferactively-pymdp 1.0.3 and h3-py 4.5.0. Each link opens a manifest-backed visualization with metadata and plotted-data sidecars.</p>
    <p><a href="gallery_manifest.json">gallery_manifest.json</a></p>
  </header>
  <main class="grid">{cards}</main>
</body>
</html>
"""


def _run_card(run: dict[str, Any]) -> str:
    """Build one gallery card."""
    metrics = run.get("metrics", {})
    metric_rows = "\n".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in [
            ("status", run.get("status", "")),
            ("pymdp", metrics.get("pymdp_version", "")),
            ("h3", metrics.get("h3_version", "")),
            ("cells", metrics.get("cell_count", "")),
            ("policy switches", metrics.get("spatial_policy_switch_count", "")),
            (
                "policy probability std",
                _format_float(metrics.get("spatial_policy_probability_std")),
            ),
            ("entropy std", _format_float(metrics.get("spatial_entropy_std"))),
            (
                "nested residual",
                _format_float(metrics.get("nested_mean_parent_child_residual")),
            ),
        ]
    )
    links = "\n".join(
        f'<li><a href="{item["href"]}">{Path(item["path"]).name}</a></li>'
        for item in run.get("visualizations", [])
    )
    return f"""<section>
  <h2>{run["name"]}</h2>
  <p><a href="{run["manifest"]}">manifest.json</a></p>
  <table><tbody>{metric_rows}</tbody></table>
  <ul>{links}</ul>
</section>"""


def _format_float(value: Any | None) -> str:
    if value is None:
        return ""
    try:
        return f"{float(str(value)):.6f}"
    except (TypeError, ValueError):
        return ""
