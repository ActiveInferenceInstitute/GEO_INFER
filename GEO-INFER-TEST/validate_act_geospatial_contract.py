#!/usr/bin/env python3
"""Validate GEO-INFER-ACT geospatial Active Inference contracts."""

from __future__ import annotations

import ast
import csv
import importlib
import inspect
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = REPO_ROOT / "GEO-INFER-ACT"
ACT_SRC = ACT_ROOT / "src"
sys.path.insert(0, str(ACT_SRC))

REQUIRED_GEOSPATIAL_FILES = {
    "data/full_history.json",
    "data/step_metrics.csv",
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
    "analysis/run_summary.json",
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
REQUIRED_NESTED_H3_FILES = {
    "data/h3_hierarchy.csv",
    "data/nested_h3_diagnostics.json",
    "data/nested_h3_cell_diagnostics.csv",
    "data/nested_h3_parent_child_diagnostics.csv",
    "data/nested_h3_level_diagnostics.csv",
    "visualizations/nested_h3_level_map.html",
    "visualizations/nested_h3_hierarchy_map.html",
    "visualizations/nested_h3_parent_child_residuals.html",
}
REQUIRED_FIGURE_METADATA_FIELDS = {
    "schema_version",
    "package",
    "package_version",
    "scenario",
    "figure_id",
    "title",
    "description",
    "alt_text",
    "generated_at",
    "run_config",
    "plotted_metrics",
    "data_sources",
    "figure_path",
    "figure_data_path",
    "sha256",
}

REQUIRED_DOCS = [
    ACT_ROOT / "README.md",
    ACT_ROOT / "docs" / "geospatial_applications.md",
    ACT_ROOT / "docs" / "active_inference_overview.md",
    ACT_ROOT / "docs" / "mathematical_framework.md",
    ACT_ROOT / "docs" / "free_energy_principle.md",
    ACT_ROOT / "examples" / "README.md",
    ACT_ROOT / "src" / "README.md",
    ACT_ROOT / "src" / "geo_infer_act" / "README.md",
]

REQUIRED_MERMAID_HEADINGS = [
    "Geospatial Active Inference Architecture",
    "H3 Perception-Action Sequence",
    "Runner Output Manifest Pipeline",
    "H3 Result and Schema Contracts",
    "Spatial Belief Propagation and Neighbor Diffusion",
    "Multi-Agent H3 Lattice Coordination",
    "Validation Flow",
    "Figure Artifact Metadata Pipeline",
    "Figure Sidecar Traceability",
    "Manifest Visualization Validation",
]

DOCSTRING_TARGETS = [
    ("geo_infer_act.core.generative_model", "GenerativeModel.enable_h3_spatial"),
    ("geo_infer_act.core.generative_model", "GenerativeModel.update_h3_beliefs"),
    ("geo_infer_act.core.active_inference", "ActiveInferenceModel.apply_to_h3"),
    ("geo_infer_act.core.active_inference", "ActiveInferenceModel.infer_over_h3_grid"),
    ("geo_infer_act.core.spatial_agent", "SpatialActiveInferenceAgent"),
    ("geo_infer_act.core.spatial_agent", "SpatialActiveInferenceAgent.step"),
    ("geo_infer_act.models.multi_agent", "MultiAgentModel.enable_h3_spatial"),
    ("geo_infer_act.models.multi_agent", "MultiAgentModel.simulate_h3_lattice"),
    (
        "geo_infer_ai.models.predictive.geospatial_ai",
        "EnvironmentalActiveInferenceEngine",
    ),
    (
        "geo_infer_ai.models.predictive.geospatial_ai",
        "EnvironmentalActiveInferenceEngine.compute_spatial_priors",
    ),
    ("geo_infer_ai.models.predictive.geospatial_ai", "H3SpatialGraph"),
    ("geo_infer_ai.models.predictive.geospatial_ai", "LevelSpatialGraph"),
    ("geo_infer_act.utils.spatial_diagnostics", "SpatialDiagnostics"),
    ("geo_infer_act.runners.scenarios", "_run_h3_scenario"),
    ("geo_infer_act.runners.scenarios", "_plot_h3_cell_metric_map"),
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def resolve_attr(module_name: str, dotted_name: str) -> Any:
    obj: Any = importlib.import_module(module_name)
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def validate_docstrings() -> None:
    for module_name, dotted_name in DOCSTRING_TARGETS:
        obj = resolve_attr(module_name, dotted_name)
        if not inspect.getdoc(obj):
            fail(f"Missing docstring: {module_name}.{dotted_name}")

    for path in [
        REPO_ROOT
        / "GEO-INFER-AI"
        / "src"
        / "geo_infer_ai"
        / "models"
        / "predictive"
        / "geospatial_ai.py",
    ]:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name.lower()
                if any(key in name for key in ("h3", "spatial", "geo", "plot")):
                    if not ast.get_docstring(node):
                        fail(f"Missing geospatial docstring: {path}:{node.lineno}")


def validate_packaging() -> None:
    if (ACT_ROOT / "setup.py").exists():
        fail("setup.py is retired; ACT packaging is pyproject-only")
    pyproject_text = (ACT_ROOT / "pyproject.toml").read_text()
    if "h3>=4.5.0,<5" not in pyproject_text:
        fail("pyproject.toml does not require h3>=4.5.0,<5")
    if "inferactively-pymdp==1.0.3" not in pyproject_text:
        fail("pyproject.toml does not pin inferactively-pymdp==1.0.3")


def validate_geospatial_outputs() -> None:
    from geo_infer_act.runners import RunConfig, run_scenario
    from geo_infer_act.utils.h3_adapter import get_h3_adapter
    from geo_infer_act.utils.pymdp_adapter import validate_pymdp_version

    if validate_pymdp_version() != "1.0.3":
        fail("inferactively-pymdp runtime is not exactly 1.0.3")
    adapter = get_h3_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in ("h3", "spatial"):
            result = run_scenario(
                RunConfig(
                    scenario=scenario,
                    output_dir=Path(tmp) / scenario,
                    seed=23,
                    deterministic=True,
                    timesteps=3,
                    visualizations=True,
                    h3_resolution=8,
                    h3_ring_size=1,
                )
            )
            manifest = json.loads(result.manifest_path.read_text())
            if manifest["validation"]["status"] != "passed":
                fail(f"{scenario} manifest validation failed: {manifest['validation']}")

            generated = {item["path"] for item in manifest["generated_files"]}
            generated_entries = {
                item["path"]: item for item in manifest["generated_files"]
            }
            missing = sorted(REQUIRED_GEOSPATIAL_FILES - generated)
            if missing:
                fail(f"{scenario} missing geospatial outputs: {missing}")

            geojson_path = result.output_dir / "data" / "h3_cells.geojson"
            geojson = json.loads(geojson_path.read_text())
            if geojson.get("type") != "FeatureCollection":
                fail(f"{scenario} GeoJSON is not a FeatureCollection")
            if len(geojson.get("features", [])) != manifest["metrics"]["cell_count"]:
                fail(f"{scenario} GeoJSON feature count does not match cell_count")
            for feature in geojson["features"]:
                cell = feature["properties"]["h3_cell"]
                if not adapter.is_valid_cell(cell):
                    fail(f"{scenario} GeoJSON contains invalid H3 cell {cell}")
                if feature["geometry"]["type"] != "Polygon":
                    fail(f"{scenario} GeoJSON feature is not a Polygon")
                properties = feature["properties"]
                if properties.get("pymdp_version") != "1.0.3":
                    fail(f"{scenario} GeoJSON missing pymdp 1.0.3 metadata")
                if properties.get("h3_version") != "4.5.0":
                    fail(f"{scenario} GeoJSON missing h3 4.5.0 metadata")

            diagnostics = json.loads(
                (result.output_dir / "data" / "h3_diagnostics.json").read_text()
            )
            for row in diagnostics:
                if not math.isfinite(float(row["aggregate_free_energy"])):
                    fail(f"{scenario} aggregate free energy is not finite")
                consistency = row["spatial_consistency"]
                for key in ("global_coherence", "neighbor_correlations"):
                    if not math.isfinite(float(consistency[key])):
                        fail(f"{scenario} {key} is not finite")
            validate_pymdp_outputs(result.output_dir, scenario, expected_timesteps=3)
            validate_spatial_trace_outputs(
                result.output_dir,
                scenario,
                expected_timesteps=3,
                nested=False,
            )

            with (result.output_dir / "data" / "step_metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 3:
                fail(f"{scenario} step_metrics.csv has wrong row count")
            for row in rows:
                for key in ("free_energy", "expected_free_energy", "belief_entropy"):
                    if not math.isfinite(float(row[key])):
                        fail(f"{scenario} {key} is not finite")

            validate_visualization_metadata(
                result.output_dir, scenario, generated_entries
            )
            summary = json.loads(
                (result.output_dir / "analysis" / "run_summary.json").read_text()
            )
            if summary.get("pymdp_version") != "1.0.3":
                fail(f"{scenario} summary missing pymdp 1.0.3")
            if summary.get("h3_version") != "4.5.0":
                fail(f"{scenario} summary missing h3 4.5.0")

        nested_result = run_scenario(
            RunConfig(
                scenario="h3",
                output_dir=Path(tmp) / "h3_nested",
                seed=29,
                deterministic=True,
                timesteps=1,
                visualizations=True,
                h3_resolution=9,
                h3_ring_size=0,
                parameters={"nested_h3": True},
            )
        )
        nested_manifest = json.loads(nested_result.manifest_path.read_text())
        if nested_manifest["validation"]["status"] != "passed":
            fail(
                f"nested H3 manifest validation failed: {nested_manifest['validation']}"
            )
        nested_generated = {item["path"] for item in nested_manifest["generated_files"]}
        missing_nested = sorted(REQUIRED_NESTED_H3_FILES - nested_generated)
        if missing_nested:
            fail(f"nested H3 missing outputs: {missing_nested}")
        nested_entries = {
            item["path"]: item for item in nested_manifest["generated_files"]
        }
        validate_visualization_metadata(
            nested_result.output_dir,
            "h3",
            nested_entries,
        )
        summary = json.loads(
            (nested_result.output_dir / "analysis" / "run_summary.json").read_text()
        )
        if not summary.get("nested_h3"):
            fail("nested H3 run summary did not record nested_h3=true")
        if summary.get("nested_orphan_count") != 0:
            fail(f"nested H3 produced orphan cells: {summary}")
        for key in (
            "nested_cross_level_coherence",
            "nested_aggregate_free_energy",
        ):
            if not math.isfinite(float(summary[key])):
                fail(f"nested H3 summary has non-finite {key}")
        nested_diagnostics = json.loads(
            (
                nested_result.output_dir / "data" / "nested_h3_diagnostics.json"
            ).read_text()
        )
        if not nested_diagnostics or not nested_diagnostics[0].get("level_summaries"):
            fail("nested H3 diagnostics missing level summaries")
        validate_pymdp_outputs(
            nested_result.output_dir, "h3_nested", expected_timesteps=1
        )
        validate_spatial_trace_outputs(
            nested_result.output_dir,
            "h3_nested",
            expected_timesteps=1,
            nested=True,
        )
        if summary.get("pymdp_version") != "1.0.3":
            fail("nested H3 summary missing pymdp 1.0.3")
        if summary.get("h3_version") != "4.5.0":
            fail("nested H3 summary missing h3 4.5.0")

        validate_gallery_outputs(Path(tmp))


def validate_pymdp_outputs(
    output_dir: Path, scenario: str, expected_timesteps: int
) -> None:
    """Validate real pymdp H3 diagnostics and reject cosmetic placeholders."""
    diagnostics_path = output_dir / "data" / "pymdp_h3_diagnostics.json"
    posterior_path = output_dir / "data" / "pymdp_policy_posteriors.csv"
    if not diagnostics_path.exists() or not posterior_path.exists():
        fail(f"{scenario} missing pymdp diagnostic files")
    diagnostics = json.loads(diagnostics_path.read_text())
    if not diagnostics:
        fail(f"{scenario} pymdp diagnostics are empty")
    with posterior_path.open() as handle:
        posterior_rows = list(csv.DictReader(handle))
    if len(posterior_rows) != len(diagnostics):
        fail(f"{scenario} pymdp CSV/JSON row count mismatch")
    timesteps = {int(row["timestep"]) for row in diagnostics}
    if len(timesteps) != expected_timesteps:
        fail(f"{scenario} pymdp diagnostics have wrong timestep count")
    for row in diagnostics:
        if row.get("pymdp_version") != "1.0.3":
            fail(f"{scenario} pymdp row has wrong version: {row}")
        if row.get("h3_version") != "4.5.0":
            fail(f"{scenario} pymdp row has wrong h3 version: {row}")
        posterior = [
            float(value)
            for key, value in row.items()
            if key.startswith("policy_posterior_")
        ]
        neg_efe = [
            float(value)
            for key, value in row.items()
            if key.startswith("negative_expected_free_energy_")
            and key != "selected_negative_expected_free_energy"
        ]
        if not posterior or len(posterior) != len(neg_efe):
            fail(f"{scenario} pymdp row lacks policy/negative-EFE arrays")
        if not math.isclose(sum(posterior), 1.0, abs_tol=1e-6):
            fail(f"{scenario} pymdp posterior is not normalized")
        if not all(math.isfinite(value) for value in posterior + neg_efe):
            fail(f"{scenario} pymdp arrays contain non-finite values")
        for key in (
            "free_energy",
            "belief_entropy",
            "selected_action_probability",
            "selected_negative_expected_free_energy",
        ):
            if not math.isfinite(float(row[key])):
                fail(f"{scenario} pymdp {key} is not finite")


def validate_spatial_trace_outputs(
    output_dir: Path,
    scenario: str,
    expected_timesteps: int,
    nested: bool,
) -> None:
    """Validate trace-first spatial diagnostics and reject cosmetic rows."""
    trace_path = output_dir / "data" / "spatial_inference_trace.json"
    cell_csv = output_dir / "data" / "h3_cell_diagnostics.csv"
    edge_csv = output_dir / "data" / "h3_edge_diagnostics.csv"
    for path in (trace_path, cell_csv, edge_csv):
        if not path.exists() or path.stat().st_size <= 0:
            fail(f"{scenario} missing spatial trace artifact: {path.name}")

    trace = json.loads(trace_path.read_text())
    if trace.get("schema_version") != "geo-infer-act-spatial-inference-trace/v1":
        fail(f"{scenario} spatial trace schema is wrong")
    if len(trace.get("timesteps", [])) != expected_timesteps:
        fail(f"{scenario} spatial trace has wrong timestep count")
    cell_rows = trace.get("cell_diagnostics", [])
    level_rows = trace.get("level_diagnostics", [])
    if not cell_rows or not level_rows:
        fail(f"{scenario} spatial trace lacks cell or level diagnostics")
    validate_research_statistics(
        output_dir,
        scenario,
        trace.get("research_statistics", {}),
        require_non_degenerate=False,
    )
    validate_lattice_animation_payload(
        output_dir,
        scenario,
        expected_timesteps=expected_timesteps,
        nested=nested,
    )

    leaf_rows = [
        row for row in cell_rows if not bool(row.get("aggregate_parent_cell", False))
    ]
    if not leaf_rows:
        fail(f"{scenario} spatial trace has no runtime leaf-cell rows")
    for row in leaf_rows:
        if row.get("pymdp_version") != "1.0.3":
            fail(f"{scenario} trace row missing pymdp version: {row}")
        if row.get("h3_version") != "4.5.0":
            fail(f"{scenario} trace row missing h3 version: {row}")
        belief = [float(value) for value in row.get("belief", [])]
        posterior = [float(value) for value in row.get("action_posterior", [])]
        neg_efe = [
            float(value) for value in row.get("negative_expected_free_energy", [])
        ]
        if not belief or not posterior or len(posterior) != len(neg_efe):
            fail(f"{scenario} trace row lacks belief/policy arrays: {row}")
        if not math.isclose(sum(belief), 1.0, abs_tol=1e-6):
            fail(f"{scenario} trace belief is not normalized")
        if not math.isclose(sum(posterior), 1.0, abs_tol=1e-6):
            fail(f"{scenario} trace posterior is not normalized")
        for key in (
            "entropy",
            "free_energy",
            "expected_free_energy",
            "policy_entropy",
            "local_coherence",
            "posterior_delta",
            "belief_flux_in",
            "belief_flux_out",
            "belief_flux_divergence",
            "selected_action_probability",
            "selected_negative_expected_free_energy",
        ):
            if not math.isfinite(float(row[key])):
                fail(f"{scenario} trace {key} is not finite")

    with cell_csv.open() as handle:
        cell_csv_rows = list(csv.DictReader(handle))
    if len(cell_csv_rows) != len(cell_rows):
        fail(f"{scenario} trace JSON/CSV cell row count mismatch")
    for row in cell_csv_rows:
        for key in (
            "entropy",
            "free_energy",
            "policy_entropy",
            "local_coherence",
            "posterior_delta",
            "belief_flux_divergence",
            "lat",
            "lng",
        ):
            if not math.isfinite(float(row[key])):
                fail(f"{scenario} trace CSV {key} is not finite")

    if nested:
        hierarchy = trace.get("hierarchy_metadata", {})
        if not hierarchy.get("nested_h3"):
            fail(f"{scenario} spatial trace did not mark nested_h3")
        parent_rows = [
            row for row in cell_rows if bool(row.get("aggregate_parent_cell", False))
        ]
        child_rows = [row for row in cell_rows if row.get("parent_cell")]
        if not parent_rows or not child_rows:
            fail(f"{scenario} nested trace missing parent or child rows")
        parent_child_path = (
            output_dir / "data" / "nested_h3_parent_child_diagnostics.csv"
        )
        level_path = output_dir / "data" / "nested_h3_level_diagnostics.csv"
        for path in (parent_child_path, level_path):
            if not path.exists() or path.stat().st_size <= 0:
                fail(f"{scenario} missing nested trace CSV: {path.name}")
        with parent_child_path.open() as handle:
            parent_child_rows = list(csv.DictReader(handle))
        if not parent_child_rows:
            fail(f"{scenario} nested parent-child diagnostics are empty")
        for row in parent_child_rows:
            for key in ("cross_level_consistency", "cross_level_residual"):
                value = float(row[key])
                if not math.isfinite(value) or value < -1e-9:
                    fail(f"{scenario} nested {key} is invalid")


def validate_lattice_animation_payload(
    output_dir: Path,
    scenario: str,
    *,
    expected_timesteps: int,
    nested: bool,
) -> None:
    """Validate the lattice animation JSON payload."""
    path = output_dir / "data" / "h3_lattice_animation.json"
    if not path.exists() or path.stat().st_size <= 0:
        fail(f"{scenario} missing h3_lattice_animation.json")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        fail(f"{scenario} lattice animation payload is not a dict")
    for key in ("schema_version", "timesteps", "cells", "frames"):
        if key not in payload:
            fail(f"{scenario} lattice animation missing key: {key}")
    if payload.get("schema_version") not in (
        "geo-infer-act-h3-lattice-animation/v1",
        "geo-infer-act-h3-lattice-animation/v2",
    ):
        fail(f"{scenario} lattice animation schema version unknown")
    if not isinstance(payload["timesteps"], list) or len(payload["timesteps"]) < 1:
        fail(f"{scenario} lattice animation has no timesteps")
    if not isinstance(payload["cells"], list) or len(payload["cells"]) < 1:
        fail(f"{scenario} lattice animation has no cells")
    if not isinstance(payload["frames"], list) or len(payload["frames"]) < 1:
        fail(f"{scenario} lattice animation has no frames")
    if len(payload["frames"]) != len(payload["timesteps"]):
        fail(f"{scenario} lattice animation frame/timestep count mismatch")
    for frame in payload["frames"]:
        if not isinstance(frame, dict):
            fail(f"{scenario} lattice animation frame is not a dict")
        for key in ("timestep", "cell_metrics"):
            if key not in frame:
                fail(f"{scenario} frame missing key: {key}")
        metrics = frame.get("cell_metrics", {})
        if not isinstance(metrics, dict) or len(metrics) < 1:
            fail(f"{scenario} frame has no cell metrics")
        for cell_id, cell_data in metrics.items():
            if not isinstance(cell_data, dict):
                fail(f"{scenario} cell metric is not a dict")
            if "free_energy" not in cell_data and "entropy" not in cell_data:
                fail(f"{scenario} cell metric missing free_energy or entropy")
            for value in cell_data.values():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    fail(f"{scenario} non-finite cell metric value")


def validate_research_statistics(
    output_dir: Path,
    scenario: str,
    trace_statistics: dict[str, Any],
    *,
    require_non_degenerate: bool,
) -> None:
    """Validate spatial research statistics and optional variation thresholds."""
    path = output_dir / "data" / "spatial_research_statistics.json"
    if not path.exists() or path.stat().st_size <= 0:
        fail(f"{scenario} missing spatial research statistics")
    statistics = json.loads(path.read_text())
    if statistics != trace_statistics:
        fail(f"{scenario} trace/statistics payload mismatch")
    if statistics.get("schema_version") != (
        "geo-infer-act-spatial-research-statistics/v1"
    ):
        fail(f"{scenario} research statistics schema is wrong")
    for group in ("metric_summaries", "temporal_slopes", "policy", "spatial_graph"):
        if group not in statistics:
            fail(f"{scenario} missing research statistics group: {group}")
    for summary in statistics["metric_summaries"].values():
        for key in ("mean", "std", "min", "max"):
            if not math.isfinite(float(summary[key])):
                fail(f"{scenario} non-finite research statistic {key}")
    for value in statistics["temporal_slopes"].values():
        if not math.isfinite(float(value)):
            fail(f"{scenario} non-finite temporal slope")
    for value in statistics["spatial_graph"].values():
        if not math.isfinite(float(value)):
            fail(f"{scenario} non-finite spatial graph statistic")
    non_degenerate = statistics.get("non_degenerate", {})
    for key in (
        "entropy_std",
        "selected_action_probability_std",
        "local_coherence_std",
        "belief_flux_divergence_std",
    ):
        if not math.isfinite(float(non_degenerate.get(key, 0.0))):
            fail(f"{scenario} non-finite non-degenerate statistic: {key}")
    if require_non_degenerate:
        if float(non_degenerate.get("entropy_std", 0.0)) <= 1e-3:
            fail(f"{scenario} research profile entropy collapsed")
        if float(non_degenerate.get("selected_action_probability_std", 0.0)) <= 1e-4:
            fail(f"{scenario} research profile policy probabilities collapsed")
        if float(non_degenerate.get("belief_flux_divergence_std", 0.0)) <= 1e-4:
            fail(f"{scenario} research profile belief flux collapsed")
        if int(non_degenerate.get("unique_selected_action_count", 0)) < 2:
            fail(f"{scenario} research profile selected only one action")


def validate_gallery_outputs(tmp_dir: Path) -> None:
    """Validate the deterministic four-run visualization gallery contract."""
    from geo_infer_act.runners import run_spatial_active_inference_gallery

    output_dir = tmp_dir / "spatial_gallery"
    manifest = run_spatial_active_inference_gallery(
        output_dir,
        seed=73,
        timesteps=2,
        h3_resolution=8,
        h3_ring_size=1,
    )
    if manifest.get("schema_version") != "geo-infer-act-spatial-gallery/v1":
        fail("gallery manifest schema is wrong")
    if not (output_dir / "index.html").exists():
        fail("gallery index.html was not generated")
    names = {run["name"] for run in manifest.get("runs", [])}
    if names != {"h3", "h3_nested", "spatial", "spatial_nested"}:
        fail(f"gallery did not produce the expected runs: {names}")
    for run in manifest["runs"]:
        if run.get("status") != "passed":
            fail(f"gallery run did not pass manifest validation: {run}")
        run_dir = output_dir / run["name"]
        run_manifest = json.loads((run_dir / "manifest.json").read_text())
        generated = {item["path"] for item in run_manifest["generated_files"]}
        missing = sorted(REQUIRED_GEOSPATIAL_FILES - generated)
        if missing:
            fail(f"gallery run {run['name']} missing outputs: {missing}")
        if run["name"].endswith("nested"):
            missing_nested = sorted(REQUIRED_NESTED_H3_FILES - generated)
            if missing_nested:
                fail(
                    f"gallery nested run {run['name']} missing outputs: {missing_nested}"
                )
        validate_research_statistics(
            run_dir,
            run["name"],
            json.loads(
                (run_dir / "data" / "spatial_inference_trace.json").read_text()
            ).get("research_statistics", {}),
            require_non_degenerate=True,
        )


def validate_visualization_metadata(
    output_dir: Path, scenario: str, generated_entries: dict[str, dict[str, Any]]
) -> None:
    """Validate manifest-linked sidecars and embedded visualization metadata."""
    visualizations = [
        item
        for item in generated_entries.values()
        if item.get("artifact_type") == "visualization"
    ]
    if not visualizations:
        fail(f"{scenario} did not emit visualization manifest entries")

    for item in visualizations:
        path = item["path"]
        figure_path = output_dir / path
        metadata_rel = item.get("figure_metadata_path")
        data_rel = item.get("figure_data_path")
        if not metadata_rel or not data_rel:
            fail(f"{scenario} visualization missing sidecar fields: {path}")
        metadata_path = output_dir / metadata_rel
        data_path = output_dir / data_rel
        if not metadata_path.exists() or metadata_path.stat().st_size <= 0:
            fail(f"{scenario} missing figure metadata sidecar: {metadata_rel}")
        if not data_path.exists() or data_path.stat().st_size <= 0:
            fail(f"{scenario} missing figure data sidecar: {data_rel}")

        metadata = json.loads(metadata_path.read_text())
        missing_fields = sorted(REQUIRED_FIGURE_METADATA_FIELDS - set(metadata))
        if missing_fields:
            fail(f"{scenario} figure metadata missing {missing_fields}: {metadata_rel}")
        if metadata["schema_version"] != "geo-infer-act-figure-artifact/v1":
            fail(f"{scenario} figure metadata schema is wrong: {metadata_rel}")
        if metadata["scenario"] != scenario:
            fail(f"{scenario} figure metadata scenario mismatch: {metadata_rel}")
        if metadata["figure_path"] != path:
            fail(f"{scenario} figure metadata path mismatch: {metadata_rel}")
        if metadata["figure_data_path"] != data_rel:
            fail(f"{scenario} figure data path mismatch: {metadata_rel}")
        if metadata["sha256"] != item.get("sha256"):
            fail(f"{scenario} figure digest mismatch: {path}")
        if not metadata["data_sources"] or not metadata["plotted_metrics"]:
            fail(
                f"{scenario} figure metadata lacks data source or metric details: {path}"
            )
        for source in metadata["data_sources"]:
            if not (output_dir / source).exists():
                fail(f"{scenario} figure data source does not exist: {source}")

        if path.endswith(".png"):
            with Image.open(figure_path) as image:
                embedded_raw = image.info.get("geo_infer_act_metadata")
            if not embedded_raw:
                fail(f"{scenario} PNG lacks embedded ACT metadata: {path}")
            embedded = json.loads(embedded_raw)
            for key in ("schema_version", "scenario", "figure_id"):
                if embedded.get(key) != metadata.get(key):
                    fail(f"{scenario} PNG embedded metadata mismatch for {key}: {path}")
        elif path.endswith(".html"):
            html = figure_path.read_text()
            if "geo-infer-act-figure-metadata" not in html:
                fail(f"{scenario} HTML lacks embedded ACT metadata: {path}")
            if metadata["schema_version"] not in html:
                fail(f"{scenario} HTML metadata schema not embedded: {path}")

        if data_path.suffix == ".csv":
            with data_path.open() as handle:
                data_rows = list(csv.DictReader(handle))
            if not data_rows:
                fail(f"{scenario} figure data sidecar has no rows: {data_rel}")
            for row in data_rows:
                for value in row.values():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(numeric):
                        fail(
                            f"{scenario} figure data sidecar has non-finite value: {data_rel}"
                        )
        else:
            if not json.loads(data_path.read_text()):
                fail(f"{scenario} figure data sidecar has no JSON payload: {data_rel}")


def validate_docs() -> None:
    docs_text = "\n".join(path.read_text() for path in REQUIRED_DOCS)
    if docs_text.count("```mermaid") < len(REQUIRED_MERMAID_HEADINGS):
        fail("ACT docs do not contain the required number of Mermaid diagrams")
    for heading in REQUIRED_MERMAID_HEADINGS:
        if heading not in docs_text:
            fail(f"Missing Mermaid section heading: {heading}")
    stale_patterns = [
        r"\bLines?\s+\d+",
        r"\bL\d+",
        r"\bcell_a\b",
        r"\bcell_b\b",
        r"\bcell_c\b",
        r"h3>=3\.7\.0",
    ]
    for pattern in stale_patterns:
        match = re.search(pattern, docs_text)
        if match:
            fail(f"Stale docs pattern found: {match.group(0)}")

    for path in REQUIRED_DOCS:
        text = path.read_text()
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target_path = (path.parent / target.split("#", 1)[0]).resolve()
            if target_path.suffix and not target_path.exists():
                fail(f"Broken local link in {path}: {target}")


def main() -> None:
    validate_docstrings()
    validate_packaging()
    validate_geospatial_outputs()
    validate_docs()
    print("[OK] GEO-INFER-ACT geospatial contract is valid")


if __name__ == "__main__":
    main()
