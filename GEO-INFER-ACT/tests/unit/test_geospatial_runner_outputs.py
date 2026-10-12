"""Geospatial runner output contracts for ACT H3 and spatial scenarios."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path

import pytest
from PIL import Image

from geo_infer_act.runners import (
    RunConfig,
    run_scenario,
    run_spatial_active_inference_gallery,
)


GEOSPATIAL_REQUIRED_FILES = {
    "data/full_history.json",
    "data/step_metrics.csv",
    "data/h3_cells.csv",
    "data/h3_diagnostics.json",
    "data/pymdp_h3_diagnostics.json",
    "data/pymdp_policy_posteriors.csv",
    "data/spatial_inference_trace.json",
    "data/spatial_research_statistics.json",
    "data/h3_lattice_animation.json",
    "data/h3_cell_diagnostics.csv",
    "data/h3_edge_diagnostics.csv",
    "data/h3_cells.geojson",
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


def _assert_visualization_artifact(result, relative_path: str) -> None:
    manifest = json.loads(result.manifest_path.read_text())
    entries = {item["path"]: item for item in manifest["generated_files"]}
    assert relative_path in entries
    entry = entries[relative_path]
    assert entry["artifact_type"] == "visualization"
    assert entry["sha256"]
    assert entry["mime_type"] in {"image/png", "text/html"}
    assert entry["description"]
    assert entry["alt_text"]
    assert entry["data_sources"]
    assert entry["plotted_metrics"]

    figure_path = result.output_dir / relative_path
    metadata_path = result.output_dir / entry["figure_metadata_path"]
    data_path = result.output_dir / entry["figure_data_path"]
    assert figure_path.exists() and figure_path.stat().st_size > 0
    assert metadata_path.exists() and metadata_path.stat().st_size > 0
    assert data_path.exists() and data_path.stat().st_size > 0

    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema_version"] == "geo-infer-act-figure-artifact/v1"
    assert metadata["scenario"] == result.scenario
    assert metadata["figure_path"] == relative_path
    assert metadata["figure_data_path"] == entry["figure_data_path"]
    assert metadata["sha256"] == entry["sha256"]
    assert metadata["data_sources"] == entry["data_sources"]
    assert metadata["plotted_metrics"] == entry["plotted_metrics"]
    for source in metadata["data_sources"]:
        assert (result.output_dir / source).exists(), source

    if relative_path.endswith(".png"):
        with Image.open(figure_path) as image:
            assert image.size[0] > 100 and image.size[1] > 100
            embedded = json.loads(image.info["geo_infer_act_metadata"])
        assert embedded["schema_version"] == metadata["schema_version"]
        assert embedded["scenario"] == metadata["scenario"]
        assert embedded["figure_id"] == metadata["figure_id"]
    elif relative_path.endswith(".html"):
        html = figure_path.read_text()
        assert 'id="geo-infer-act-figure-metadata"' in html
        assert metadata["schema_version"] in html


def _assert_numeric_sidecar(data_path: Path) -> None:
    if data_path.suffix == ".csv":
        with data_path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            for value in row.values():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                assert math.isfinite(numeric)
    else:
        payload = json.loads(data_path.read_text())
        assert payload


def _assert_probability_vector(values, label: str) -> None:
    vector = [float(value) for value in values]
    assert vector, label
    assert all(math.isfinite(value) for value in vector), label
    assert math.isclose(sum(vector), 1.0, abs_tol=1e-6), label


def _assert_h3_lattice_animation(result, *, expected_timesteps: int) -> None:
    html_path = (
        result.output_dir / "visualizations" / "h3_active_inference_lattice.html"
    )
    if html_path.exists():
        html = html_path.read_text()
        for marker in (
            "h3-lattice-svg",
            "h3-lattice-animation-data",
            "playButton",
            "timeSlider",
            "metricSelect",
            "layerObservation",
            "layerAction",
            "layerFlux",
        ):
            assert marker in html
    payload = json.loads(
        (result.output_dir / "data" / "h3_lattice_animation.json").read_text()
    )
    assert payload["schema_version"] == "geo-infer-act-h3-lattice-animation/v1"
    assert payload["scenario"] == result.scenario
    assert payload["backend_metadata"]["pymdp_version"] == "1.0.3"
    assert payload["backend_metadata"]["h3_version"] == "4.5.0"
    assert payload["frame_count"] == expected_timesteps
    assert len(payload["frames"]) == expected_timesteps
    assert payload["cells"]
    for cell in payload["cells"]:
        assert cell["cell"]
        assert cell["geometry"]["type"] == "Polygon"
        ring = cell["geometry"]["coordinates"][0]
        assert len(ring) >= 6
        assert ring[0] == ring[-1]
        assert all(len(point) == 2 for point in ring)
    for frame in payload["frames"]:
        assert frame["cells"]
        assert "level_summaries" in frame
        leaf_states = [
            state for state in frame["cells"] if not state["is_aggregate_parent"]
        ]
        assert leaf_states
        for state in leaf_states:
            _assert_probability_vector(state["belief"], f"belief {state['cell']}")
            _assert_probability_vector(
                state["action_posterior"], f"posterior {state['cell']}"
            )
            _assert_probability_vector(
                state["observation"], f"observation {state['cell']}"
            )
            for key in (
                "entropy",
                "free_energy",
                "policy_entropy",
                "selected_action_probability",
                "observation_strength",
                "local_coherence",
                "posterior_delta",
                "belief_flux_divergence",
            ):
                assert math.isfinite(float(state[key])), (state["cell"], key)
        for edge in frame["edges"]:
            assert edge["flux_source"] in {edge["source"], edge["target"]}
            assert edge["flux_target"] in {edge["source"], edge["target"]}
            assert edge["flux_source"] != edge["flux_target"]
            assert math.isfinite(float(edge["weight"]))


@pytest.mark.parametrize("scenario", ["h3", "spatial"])
@pytest.mark.slow
def test_geospatial_scenarios_emit_complete_data_and_visualizations(
    scenario: str, tmp_path: Path
) -> None:
    result = run_scenario(
        RunConfig(
            scenario=scenario,
            output_dir=tmp_path / scenario,
            seed=31,
            deterministic=True,
            timesteps=3,
            visualizations=True,
            h3_resolution=8,
            h3_ring_size=1,
        )
    )

    manifest = json.loads(result.manifest_path.read_text())
    generated = {item["path"] for item in manifest["generated_files"]}
    assert manifest["validation"]["status"] == "passed"
    assert GEOSPATIAL_REQUIRED_FILES <= generated

    for relative_path in GEOSPATIAL_REQUIRED_FILES:
        path = result.output_dir / relative_path
        assert path.exists(), relative_path
        assert path.stat().st_size > 0, relative_path

    geojson = json.loads((result.output_dir / "data" / "h3_cells.geojson").read_text())
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == manifest["metrics"]["cell_count"]
    for feature in geojson["features"]:
        assert feature["geometry"]["type"] == "Polygon"
        assert feature["properties"]["h3_cell"]
        assert feature["properties"]["resolution"] == 8
        assert feature["properties"]["pymdp_version"] == "1.0.3"
        assert feature["properties"]["h3_version"] == "4.5.0"

    diagnostics = json.loads(
        (result.output_dir / "data" / "h3_diagnostics.json").read_text()
    )
    assert len(diagnostics) == 3
    for row in diagnostics:
        assert math.isfinite(float(row["aggregate_free_energy"]))
        consistency = row["spatial_consistency"]
        assert math.isfinite(float(consistency["global_coherence"]))
        assert math.isfinite(float(consistency["neighbor_correlations"]))

    with (result.output_dir / "data" / "step_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for row in rows:
        assert math.isfinite(float(row["free_energy"]))
        assert math.isfinite(float(row["expected_free_energy"]))
        assert math.isfinite(float(row["belief_entropy"]))
        assert math.isfinite(float(row["coherence"]))

    trace = json.loads(
        (result.output_dir / "data" / "spatial_inference_trace.json").read_text()
    )
    assert trace["schema_version"] == "geo-infer-act-spatial-inference-trace/v1"
    assert trace["cell_diagnostics"]
    assert trace["level_diagnostics"]
    assert trace["research_statistics"]["schema_version"].endswith(
        "spatial-research-statistics/v1"
    )
    _assert_h3_lattice_animation(result, expected_timesteps=3)
    statistics = json.loads(
        (result.output_dir / "data" / "spatial_research_statistics.json").read_text()
    )
    assert statistics["row_count"] >= manifest["metrics"]["cell_count"] * 3
    for group in ("metric_summaries", "temporal_slopes", "policy", "spatial_graph"):
        assert group in statistics
    with (result.output_dir / "data" / "h3_cell_diagnostics.csv").open() as handle:
        cell_trace_rows = list(csv.DictReader(handle))
    with (result.output_dir / "data" / "h3_edge_diagnostics.csv").open() as handle:
        edge_trace_rows = list(csv.DictReader(handle))
    assert len(cell_trace_rows) >= manifest["metrics"]["cell_count"] * 3
    assert edge_trace_rows
    for row in cell_trace_rows:
        assert row["cell"]
        assert row["pymdp_version"] == "1.0.3"
        assert row["h3_version"] == "4.5.0"
        for key in (
            "entropy",
            "free_energy",
            "policy_entropy",
            "local_coherence",
            "posterior_delta",
            "belief_flux_divergence",
            "selected_action_probability",
        ):
            assert math.isfinite(float(row[key]))

    pymdp_diagnostics = json.loads(
        (result.output_dir / "data" / "pymdp_h3_diagnostics.json").read_text()
    )
    with (result.output_dir / "data" / "pymdp_policy_posteriors.csv").open() as handle:
        posterior_rows = list(csv.DictReader(handle))
    assert pymdp_diagnostics
    assert len(posterior_rows) == len(pymdp_diagnostics)
    for row in pymdp_diagnostics:
        posterior = [
            float(value)
            for key, value in row.items()
            if key.startswith("policy_posterior_")
        ]
        neg_efe = [
            float(value)
            for key, value in row.items()
            if key.startswith("negative_expected_free_energy_")
        ]
        assert row["pymdp_version"] == "1.0.3"
        assert row["h3_version"] == "4.5.0"
        assert posterior
        assert len(posterior) == len(neg_efe)
        assert math.isclose(sum(posterior), 1.0, abs_tol=1e-6)
        assert all(math.isfinite(value) for value in posterior + neg_efe)

    summary = json.loads(
        (result.output_dir / "analysis" / "run_summary.json").read_text()
    )
    assert summary["pymdp_backend"] == "inferactively-pymdp"
    assert summary["pymdp_version"] == "1.0.3"
    assert summary["h3_version"] == "4.5.0"
    assert "spatial_research_statistics_schema" in summary


@pytest.mark.slow
def test_geospatial_manifest_references_only_existing_files(tmp_path: Path) -> None:
    result = run_scenario(
        RunConfig(
            scenario="h3",
            output_dir=tmp_path / "h3",
            seed=9,
            timesteps=2,
            visualizations=True,
        )
    )
    manifest = json.loads(result.manifest_path.read_text())

    for item in manifest["generated_files"]:
        referenced = result.output_dir / item["path"]
        assert referenced.exists(), item["path"]
        assert referenced.stat().st_size == item["size_bytes"]


@pytest.mark.parametrize("scenario", ["h3", "spatial"])
@pytest.mark.slow
def test_geospatial_visualizations_have_metadata_and_data_sidecars(
    scenario: str, tmp_path: Path
) -> None:
    result = run_scenario(
        RunConfig(
            scenario=scenario,
            output_dir=tmp_path / scenario,
            seed=19,
            deterministic=True,
            timesteps=3,
            visualizations=True,
            h3_resolution=8,
            h3_ring_size=1,
        )
    )
    manifest = json.loads(result.manifest_path.read_text())
    visualizations = [
        item["path"]
        for item in manifest["generated_files"]
        if item["artifact_type"] == "visualization"
    ]
    assert {
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
    } <= set(visualizations)

    for relative_path in visualizations:
        _assert_visualization_artifact(result, relative_path)
        entry = {item["path"]: item for item in manifest["generated_files"]}[
            relative_path
        ]
        _assert_numeric_sidecar(result.output_dir / entry["figure_data_path"])


@pytest.mark.parametrize("scenario", ["h3", "spatial"])
@pytest.mark.slow
def test_research_profile_produces_non_degenerate_spatial_statistics(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A short environmental sample can legitimately prefer one action. Supply
    # independently identified, cycling observations so this artifact test
    # actually exercises policy diversity without sharing cell posteriors.
    def varied_observations(cells, timestep, **_):
        observations = {}
        for index, cell in enumerate(cells):
            state = (index + int(timestep)) % 4
            observations[cell] = {
                "air_quality": 0.0 if state == 0 else 1.0,
                "water_stress": float(state == 1),
                "vegetation_density": float(state == 2),
                "carbon_flux": 40.0 if state == 3 else -40.0,
            }
        return observations

    monkeypatch.setattr(
        "geo_infer_act.runners.scenarios.generate_realistic_environmental_observations",
        varied_observations,
    )
    result = run_scenario(
        RunConfig(
            scenario=scenario,
            output_dir=tmp_path / scenario,
            seed=61,
            deterministic=True,
            timesteps=3,
            visualizations=True,
            h3_resolution=8,
            h3_ring_size=1,
            parameters={"research_profile": True},
        )
    )

    statistics = json.loads(
        (result.output_dir / "data" / "spatial_research_statistics.json").read_text()
    )
    non_degenerate = statistics["non_degenerate"]
    assert non_degenerate["entropy_std"] > 1e-3
    assert non_degenerate["selected_action_probability_std"] > 1e-4
    assert non_degenerate["local_coherence_std"] > 1e-4
    assert non_degenerate["belief_flux_divergence_std"] > 1e-4
    assert non_degenerate["unique_selected_action_count"] >= 2
    assert statistics["policy"]["switch_count"] >= 1


@pytest.mark.slow
def test_spatial_active_inference_gallery_emits_four_manifested_runs(
    tmp_path: Path,
) -> None:
    manifest = run_spatial_active_inference_gallery(
        tmp_path / "gallery",
        seed=73,
        timesteps=2,
        h3_resolution=8,
        h3_ring_size=1,
    )

    gallery_dir = tmp_path / "gallery"
    assert (gallery_dir / "index.html").exists()
    assert (gallery_dir / "gallery_manifest.json").exists()
    assert manifest["schema_version"] == "geo-infer-act-spatial-gallery/v1"
    assert {run["name"] for run in manifest["runs"]} == {
        "h3",
        "h3_nested",
        "spatial",
        "spatial_nested",
    }
    for run in manifest["runs"]:
        run_manifest = json.loads((gallery_dir / run["manifest"]).read_text())
        assert run_manifest["validation"]["status"] == "passed"
        assert run["status"] == "passed"
        assert run["visualizations"]
        assert run["metrics"]["pymdp_version"] == "1.0.3"
        assert run["metrics"]["h3_version"] == "4.5.0"
        assert run["metrics"]["spatial_policy_probability_std"] > 1e-4
        assert any(
            item["path"] == "visualizations/h3_active_inference_lattice.html"
            for item in run["visualizations"]
        )
        _assert_h3_lattice_animation(
            type(
                "GalleryResult",
                (),
                {"output_dir": gallery_dir / run["name"], "scenario": run["scenario"]},
            ),
            expected_timesteps=2,
        )


def test_plotly_or_table_degrades_to_logged_static_table(caplog) -> None:
    """A Plotly failure is logged and downgraded to a static HTML table."""
    from geo_infer_act.runners.scenarios import _plotly_or_table

    def broken_builder() -> object:
        raise RuntimeError("plotly unavailable")

    with caplog.at_level(logging.WARNING, logger="geo_infer_act.runners.scenarios"):
        html = _plotly_or_table(
            broken_builder,
            [{"cell": "811", "value": 0.25}],
            ["Cell", "Value"],
            lambda row: (row["cell"], f"{row['value']:.6f}"),
            "H3 Diagnostics",
            page_title="Diagnostics Map",
            caption="Source: data/example.csv",
        )

    assert "<!doctype html>" in html
    assert "<title>Diagnostics Map</title>" in html
    assert "<h1>H3 Diagnostics</h1>" in html
    assert "<p>Source: data/example.csv</p>" in html
    assert "<th>Cell</th><th>Value</th>" in html
    assert "<td>811</td><td>0.250000</td>" in html
    assert "plotly" not in html
    fallback_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "downgrading to a static HTML table" in record.getMessage()
        and record.exc_info is not None
        and record.exc_info[0] is RuntimeError
    ]
    assert fallback_warnings, "fallback must log a warning with the exception"


def test_plotly_or_table_prefers_figure_html_when_builder_succeeds(caplog) -> None:
    """A working Plotly builder produces figure HTML with no fallback warning."""
    from geo_infer_act.runners.scenarios import _plotly_or_table

    class FakeFigure:
        @staticmethod
        def to_html(include_plotlyjs: str, full_html: bool) -> str:
            assert include_plotlyjs == "cdn"
            assert full_html is True
            return "<html><body>plotly-figure-html</body></html>"

    with caplog.at_level(logging.WARNING, logger="geo_infer_act.runners.scenarios"):
        html = _plotly_or_table(
            FakeFigure,
            [],
            ["Unused"],
            lambda row: (),
            "Report",
        )

    assert "plotly-figure-html" in html
    assert "<table>" not in html
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
