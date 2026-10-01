#!/usr/bin/env python3
"""
Validate the GEO-INFER H3 + Active Inference runtime contract.

This check is intentionally executable: it uses real H3 v4 cells through
GEO-INFER-SPACE, runs the canonical ACT H3 paths, and verifies normalized
beliefs plus finite free-energy and spatial diagnostics.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
ACT_SRC = REPO_ROOT / "GEO-INFER-ACT" / "src"
SPACE_SRC = REPO_ROOT / "GEO-INFER-SPACE" / "src"
ACT_PACKAGE = ACT_SRC / "geo_infer_act"
SPACE_PACKAGE = SPACE_SRC / "geo_infer_space"

# The H3 contract applies to every shipped module and example, not only the
# original ACT/SPACE integration. Scan module trees while excluding tests and
# cache files in ``_is_runtime_file`` below.
RUNTIME_SOURCE_ROOTS = [
    module_dir
    for module_dir in sorted(REPO_ROOT.glob("GEO-INFER-*"))
    if module_dir.is_dir()
]
TARGETED_H3_FILES = [
    ACT_PACKAGE / "core" / "active_inference.py",
    ACT_PACKAGE / "core" / "generative_model.py",
    ACT_PACKAGE / "core" / "spatial_agent.py",
    ACT_PACKAGE / "models" / "multi_agent.py",
    ACT_PACKAGE / "utils" / "h3_adapter.py",
    SPACE_PACKAGE / "core" / "spatial_indexing.py",
    SPACE_PACKAGE / "backends" / "h3" / "h3_backend.py",
    SPACE_PACKAGE / "backends" / "h3" / "operations.py",
]
DOC_FILES = sorted(
    (
        REPO_ROOT / "GEO-INFER-INTRA" / "docs" / "geospatial" / "data_formats" / "h3"
    ).glob("*.md")
)
DOC_FILES.extend(
    sorted(
        path
        for path in (REPO_ROOT / "GEO-INFER-SPACE" / "docs").glob("*.md")
        if "h3" in path.name.lower()
        or path.name in {"CLI_TOOLS.md", "H3_MODULE_CONFIGURATION_GUIDE.md"}
    )
)
DOC_FILES.extend(
    [
        REPO_ROOT / "GEO-INFER-ACT" / "README.md",
        REPO_ROOT / "GEO-INFER-ACT" / "SKILL.md",
        REPO_ROOT / "GEO-INFER-ACT" / "docs" / "mathematical_framework.md",
    ]
)
H3_V3_API_CALLS = [
    "h3.geo_to_h3(",
    "h3.h3_to_geo(",
    "h3.k_ring(",
    "h3.hex_ring(",
    "h3.polyfill(",
    "h3.h3_to_parent(",
    "h3.h3_to_children(",
    "h3.h3_to_geo_boundary(",
    "h3.h3_set_to_multi_polygon(",
    "h3.h3_is_valid(",
    "h3.hex_area(",
    "h3.grid_ring_unsafe(",
]
OBSOLETE_PYMDP_RUNTIME_IMPORTS = [
    "pymdp.control",
    "pymdp.inference",
]
REQUIRED_PYMDP_VERSION = "1.0.3"
REQUIRED_H3_VERSION = "4.5.0"


def _is_runtime_file(path: Path) -> bool:
    parts = set(path.parts)
    if {"tests", "__pycache__"} & parts:
        return False
    if path.name == "validate_h3_active_inference_contract.py":
        return False
    return path.suffix == ".py"


def _validate_no_h3_v3_calls() -> None:
    offenders: list[str] = []
    for root in RUNTIME_SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if not _is_runtime_file(path):
                continue
            text = path.read_text()
            for pattern in H3_V3_API_CALLS:
                if pattern in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {pattern}")
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: syntax error: {exc.msg}"
                )
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "cell_to_boundary"
                ):
                    continue
                if any(keyword.arg == "geo_json" for keyword in node.keywords):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
                        "cell_to_boundary does not accept geo_json in H3 v4"
                    )
    for path in DOC_FILES:
        if not path.exists():
            continue
        text = path.read_text()
        for pattern in H3_V3_API_CALLS:
            if pattern in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {pattern}")
    assert not offenders, "H3 v3 API calls found:\n" + "\n".join(offenders)


def _validate_no_inert_h3_methods() -> None:
    offenders: list[str] = []
    for path in TARGETED_H3_FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: pass")
            if isinstance(node, ast.Raise):
                text = ast.get_source_segment(path.read_text(), node) or ""
                if "NotImplemented" in text:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {text.strip()}"
                    )
    assert not offenders, "Inert H3 method bodies found:\n" + "\n".join(offenders)


def _validate_no_obsolete_pymdp_runtime_imports() -> None:
    offenders: list[str] = []
    for path in sorted(ACT_PACKAGE.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        text = path.read_text()
        for pattern in OBSOLETE_PYMDP_RUNTIME_IMPORTS:
            if pattern in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {pattern}")
    assert not offenders, "Obsolete pymdp runtime imports found:\n" + "\n".join(
        offenders
    )


def _validate_public_h3_docstrings() -> None:
    missing: list[str] = []
    for path in TARGETED_H3_FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            name = node.name.lower()
            if "h3" not in name and "cell" not in name and "spatial" not in name:
                continue
            if not ast.get_docstring(node):
                missing.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.name}"
                )
    assert not missing, "Public H3 methods missing docstrings:\n" + "\n".join(missing)


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "states" in value:
        value = value["states"]
    return np.asarray(value, dtype=float).reshape(-1)


def _assert_probability_vector(value: Any, label: str) -> None:
    array = _as_array(value)
    assert array.size > 0, f"{label} is empty"
    assert np.all(np.isfinite(array)), f"{label} contains non-finite values"
    assert np.all(array >= -1e-12), f"{label} contains negative values"
    assert np.isclose(np.sum(array), 1.0, atol=1e-6), (
        f"{label} is not normalized: sum={np.sum(array)}"
    )


def _assert_finite(value: Any, label: str) -> None:
    assert math.isfinite(float(value)), f"{label} is not finite: {value!r}"


def _assert_pymdp_metadata(metadata: dict[str, Any] | None, label: str) -> None:
    assert metadata, f"{label} missing pymdp metadata"
    assert metadata.get("backend") == "inferactively-pymdp", (
        f"{label} did not use inferactively-pymdp"
    )
    assert metadata.get("pymdp_version") == REQUIRED_PYMDP_VERSION, (
        f"{label} wrong pymdp version: {metadata}"
    )
    assert metadata.get("h3_version") == REQUIRED_H3_VERSION, (
        f"{label} wrong h3 version: {metadata}"
    )
    assert "action_posterior" in metadata, f"{label} missing action posterior"
    assert "negative_expected_free_energy" in metadata, (
        f"{label} missing negative expected free energy"
    )
    posterior = np.asarray(metadata.get("action_posterior"), dtype=float)
    neg_efe = np.asarray(metadata.get("negative_expected_free_energy"), dtype=float)
    assert posterior.size > 0, f"{label} missing action posterior"
    assert neg_efe.size == posterior.size, f"{label} mismatched pymdp policy arrays"
    assert np.all(np.isfinite(posterior)), f"{label} posterior non-finite"
    assert np.all(np.isfinite(neg_efe)), f"{label} negative EFE non-finite"
    assert np.isclose(posterior.sum(), 1.0, atol=1e-6), (
        f"{label} posterior not normalized"
    )


def _assert_spatial_trace(
    trace: Any,
    *,
    expected_leaf_cells: int,
    label: str,
) -> None:
    from geo_infer_act import SpatialInferenceTrace

    assert isinstance(trace, SpatialInferenceTrace), f"{label} wrong trace type"
    assert trace.cell_diagnostics, f"{label} missing cell diagnostics"
    assert trace.level_diagnostics, f"{label} missing level diagnostics"
    _assert_pymdp_metadata(trace.backend_metadata, f"{label} backend metadata")
    leaf_rows = [
        row
        for row in trace.cell_diagnostics
        if not row.metadata.get("aggregate_parent_cell")
    ]
    assert len(leaf_rows) == expected_leaf_cells, f"{label} wrong leaf row count"
    for row in trace.cell_diagnostics:
        _assert_probability_vector(row.belief, f"{label} belief[{row.cell}]")
        for key, value in {
            "entropy": row.entropy,
            "free_energy": row.free_energy,
            "expected_free_energy": row.expected_free_energy,
            "policy_entropy": row.policy_entropy,
            "local_coherence": row.local_coherence,
            "posterior_delta": row.posterior_delta,
            "belief_flux_in": row.belief_flux_in,
            "belief_flux_out": row.belief_flux_out,
            "belief_flux_divergence": row.belief_flux_divergence,
        }.items():
            _assert_finite(value, f"{label} {key}[{row.cell}]")
        if not row.metadata.get("aggregate_parent_cell"):
            _assert_probability_vector(
                row.action_posterior,
                f"{label} action posterior[{row.cell}]",
            )
            assert len(row.action_posterior) == len(
                row.negative_expected_free_energy
            ), f"{label} policy/EFE length mismatch"
            assert row.metadata.get("pymdp_version") == REQUIRED_PYMDP_VERSION
            assert row.metadata.get("h3_version") == REQUIRED_H3_VERSION
    for edge in trace.edge_diagnostics:
        assert edge.source != edge.target, f"{label} self edge"
        _assert_finite(edge.belief_distance, f"{label} edge distance")
        _assert_finite(edge.coherence, f"{label} edge coherence")
    trace.to_dict()


def _validate_space_indexing_contract() -> list[str]:
    import h3
    from geo_infer_space.core.spatial_indexing import (
        SpatialIndexingInterface,
    )

    versions = h3.versions()
    assert versions.get("python") == REQUIRED_H3_VERSION, versions
    assert versions.get("c") == REQUIRED_H3_VERSION, versions

    indexer = SpatialIndexingInterface(backend="h3")
    center = indexer.latlng_to_cell(37.7749, -122.4194, 8)
    lat, lng = indexer.cell_to_latlng(center)
    assert 37.0 < lat < 38.5
    assert -123.5 < lng < -121.5

    boundary = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.42, 37.77],
                [-122.42, 37.79],
                [-122.39, 37.79],
                [-122.39, 37.77],
                [-122.42, 37.77],
            ]
        ],
    }
    cells = indexer.polygon_to_cells(boundary, 8)
    if len(cells) < 2:
        cells = indexer.get_cell_neighbors(center, 1)[:6] + [center]
    cells = sorted(set(cells))
    assert len(cells) >= 2, "SPACE H3 indexing did not produce enough cells"

    parent = indexer.get_cell_parent(cells[0], 7)
    children = indexer.get_cell_children(parent, 8)
    assert cells[0] in children
    assert indexer.get_cell_resolution(cells[0]) == 8
    assert indexer.get_cell_ring(cells[0], 1)
    assert indexer.get_cell_boundary(cells[0])
    assert indexer.get_cell_area(cells[0]) > 0
    compacted = indexer.compact_cells(children)
    uncompacted = indexer.uncompact_cells(compacted, 8)
    assert set(children).issubset(set(uncompacted))
    multipolygon = indexer.cells_to_multipolygon(cells[:2])
    assert multipolygon["type"] == "MultiPolygon"
    return cells


def _validate_act_h3_runtime(cells: list[str]) -> None:
    from geo_infer_act import (
        ActiveInferenceModel,
        ActiveInferenceStepResult,
        GenerativeModel,
        H3BeliefUpdateResult,
        H3GridInferenceResult,
        H3SpatialConsistency,
        NestedH3BeliefUpdateResult,
        NestedH3GridInferenceResult,
        SpatialActiveInferenceAgent,
    )
    from geo_infer_act.models.multi_agent import MultiAgentModel
    from geo_infer_act.utils.h3_adapter import get_h3_adapter
    from geo_infer_act.utils.pymdp_adapter import (
        validate_pymdp_version,
    )
    from geo_infer_act.runners.h3 import (
        generate_realistic_environmental_observations,
        h3_cells_for_config,
        observation_dict_to_vector,
    )
    from geo_infer_act.utils.spatial_research import (
        apply_h3_research_profile,
        build_spatial_research_statistics,
    )
    from geo_infer_space.nested import NestedH3Grid

    assert validate_pymdp_version() == REQUIRED_PYMDP_VERSION

    boundary = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.42, 37.77],
                [-122.42, 37.79],
                [-122.39, 37.79],
                [-122.39, 37.77],
                [-122.42, 37.77],
            ]
        ],
    }

    gen = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 3})
    gen.enable_h3_spatial(8, boundary)
    selected_cells = gen.h3_cells[: max(2, min(4, len(gen.h3_cells)))]
    observations = {
        cell: np.eye(3)[index % 3] for index, cell in enumerate(selected_cells)
    }

    belief_result = gen.update_h3_beliefs(observations, return_result=True)
    assert isinstance(belief_result, H3BeliefUpdateResult)
    assert isinstance(belief_result.spatial_consistency, H3SpatialConsistency)
    _assert_finite(belief_result.aggregate_free_energy, "H3 belief aggregate FE")
    _assert_finite(
        belief_result.spatial_consistency.global_coherence,
        "H3 belief coherence",
    )
    _assert_finite(
        belief_result.spatial_consistency.neighbor_correlations,
        "H3 neighbor correlation",
    )
    for cell, belief in belief_result.h3_beliefs.items():
        _assert_probability_vector(belief, f"belief[{cell}]")
        _assert_pymdp_metadata(
            belief_result.metadata["pymdp_cell_metadata"].get(cell),
            f"H3 belief update[{cell}]",
        )
    _assert_probability_vector(belief_result.average, "average H3 belief")

    try:
        gen.update_h3_beliefs({"not-a-real-h3-cell": np.array([1.0, 0.0, 0.0])})
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid H3 cell did not fail clearly")

    model = ActiveInferenceModel(
        model_type="categorical",
        policy_selection_mode="deterministic",
        random_seed=11,
    )
    model.set_generative_model(gen)
    grid_result = model.infer_over_h3_grid(observations, return_result=True)
    assert isinstance(grid_result, H3GridInferenceResult)
    _assert_finite(grid_result.aggregate_free_energy, "H3 grid aggregate FE")
    _assert_finite(
        grid_result.spatial_consistency.global_coherence,
        "H3 grid coherence",
    )
    assert grid_result.spatial_consistency.cell_count == len(observations)
    for cell, step_result in grid_result.cell_results.items():
        assert isinstance(step_result, ActiveInferenceStepResult)
        _assert_finite(step_result.free_energy, f"step FE[{cell}]")
        _assert_finite(step_result.expected_free_energy, f"step EFE[{cell}]")
        _assert_probability_vector(step_result.beliefs, f"step belief[{cell}]")
        _assert_pymdp_metadata(step_result.metadata.get("pymdp"), f"H3 grid[{cell}]")
    trace = model.trace_over_h3_grid(
        observations,
        timestep=0,
        previous_beliefs={cell: np.ones(3) / 3 for cell in observations},
        grid_result=grid_result,
    )
    _assert_spatial_trace(
        trace,
        expected_leaf_cells=len(observations),
        label="H3 grid trace",
    )

    research_cells = h3_cells_for_config(resolution=8, ring_size=1)
    research_gen = GenerativeModel(
        "categorical",
        {"state_dim": 4, "obs_dim": 4, "spatial_mode": True},
    )
    research_gen.spatial_mode = True
    research_gen.h3_cells = research_cells
    research_gen.spatial_graph = research_gen._build_h3_neighbor_graph(research_cells)
    research_model = ActiveInferenceModel(
        model_type="categorical",
        policy_selection_mode="deterministic",
        random_seed=41,
    )
    research_model.set_generative_model(research_gen)
    apply_h3_research_profile(research_gen, research_model)
    cell_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    previous_research_beliefs: dict[str, Any] = {}
    for timestep in range(3):
        env = generate_realistic_environmental_observations(
            research_cells,
            timestep=float(timestep),
            spatial_seed=41,
        )
        vector_observations = {
            cell: observation_dict_to_vector(observation)
            for cell, observation in env.items()
        }
        research_grid = research_model.infer_over_h3_grid(
            vector_observations,
            return_result=True,
        )
        research_trace = research_model.trace_over_h3_grid(
            vector_observations,
            timestep=timestep,
            previous_beliefs=previous_research_beliefs,
            grid_result=research_grid,
        )
        previous_research_beliefs = {
            row.cell: row.belief for row in research_trace.cell_diagnostics
        }
        cell_rows.extend(row.to_dict() for row in research_trace.cell_diagnostics)
        edge_rows.extend(row.to_dict() for row in research_trace.edge_diagnostics)
    research_statistics = build_spatial_research_statistics(cell_rows, edge_rows, [])
    non_degenerate = research_statistics["non_degenerate"]
    assert non_degenerate["entropy_std"] > 1e-3
    assert non_degenerate["selected_action_probability_std"] > 1e-4
    assert non_degenerate["belief_flux_divergence_std"] > 1e-4
    assert non_degenerate["unique_selected_action_count"] >= 2

    spatial_agent = SpatialActiveInferenceAgent(
        initial_cells=selected_cells,
        state_dim=3,
        obs_dim=3,
        h3_resolution=8,
        enable_logging=False,
    )
    agent_result = spatial_agent.step(observations, return_result=True)
    assert isinstance(agent_result, H3GridInferenceResult)
    _assert_finite(agent_result.aggregate_free_energy, "spatial-agent aggregate FE")
    assert agent_result.spatial_consistency.cell_count == len(selected_cells)
    for cell, step_result in agent_result.cell_results.items():
        _assert_probability_vector(step_result.beliefs, f"spatial-agent belief[{cell}]")
        _assert_pymdp_metadata(
            step_result.metadata.get("pymdp"),
            f"spatial-agent[{cell}]",
        )
    agent_trace = spatial_agent.trace_step(
        observations,
        grid_result=agent_result,
        timestep=0,
        previous_beliefs={cell: np.ones(3) / 3 for cell in observations},
    )
    _assert_spatial_trace(
        agent_trace,
        expected_leaf_cells=len(observations),
        label="spatial-agent trace",
    )

    multi = MultiAgentModel(n_agents=2)
    multi.enable_h3_spatial(8, boundary)
    history = multi.simulate_h3_lattice(2, lambda _cell: np.array([1.0, 0.0, 0.0, 0.0]))
    assert len(history) == 2
    for timestep in history:
        assert timestep
        for cell, payload in timestep.items():
            _assert_probability_vector(
                payload["beliefs"], f"multi-agent belief[{cell}]"
            )
            _assert_finite(payload["free_energy"], f"multi-agent FE[{cell}]")

    coordination = multi.coordinate_agents()
    assert "coordination_matrix" in coordination
    assert np.all(np.isfinite(coordination["coordination_matrix"]))

    assert set(selected_cells).issubset(set(cells) | set(gen.h3_cells))

    adapter = get_h3_adapter()
    nested_leaf_cells = [
        adapter.latlng_to_cell(37.7749, -122.4194, 9),
        *adapter.grid_ring(adapter.latlng_to_cell(37.7749, -122.4194, 9), 1)[:2],
    ]

    nested_grid = NestedH3Grid("contract_nested")
    hierarchy = nested_grid.build_h3_hierarchy_from_cells(
        nested_leaf_cells,
        [7, 8, 9],
    )
    assert hierarchy["validation"]["is_valid"], hierarchy["validation"]
    assert hierarchy["validation"]["orphan_count"] == 0
    assert hierarchy["validation"]["multi_child_parent"] is True

    nested_gen = GenerativeModel("categorical", {"state_dim": 4, "obs_dim": 4})
    nested_gen.enable_nested_h3_spatial([7, 8, 9], cells=nested_leaf_cells)
    nested_observations = {
        cell: np.eye(4)[index % 4] for index, cell in enumerate(nested_gen.h3_cells)
    }
    nested_update = nested_gen.update_nested_h3_beliefs(
        nested_observations,
        return_result=True,
    )
    assert isinstance(nested_update, NestedH3BeliefUpdateResult)
    _assert_finite(nested_update.aggregate_free_energy, "nested aggregate FE")
    assert nested_update.parent_beliefs
    for cell, belief in nested_update.fine_beliefs.items():
        _assert_probability_vector(belief, f"nested fine belief[{cell}]")
        _assert_pymdp_metadata(
            nested_update.metadata["pymdp_cell_metadata"].get(cell),
            f"nested update[{cell}]",
        )
    for cell, belief in nested_update.parent_beliefs.items():
        _assert_probability_vector(belief, f"nested parent belief[{cell}]")
    assert nested_update.spatial_consistency.metadata["cross_level_coherence"] >= 0.0

    nested_model = ActiveInferenceModel(
        model_type="categorical",
        policy_selection_mode="deterministic",
        random_seed=17,
    )
    nested_model.set_generative_model(nested_gen)
    nested_grid_result = nested_model.infer_over_nested_h3_grid(
        nested_observations,
        return_result=True,
    )
    assert isinstance(nested_grid_result, NestedH3GridInferenceResult)
    _assert_finite(
        nested_grid_result.aggregate_free_energy,
        "nested grid aggregate FE",
    )
    assert nested_grid_result.nested_belief_update.parent_child_map
    for cell, step_result in nested_grid_result.cell_results.items():
        _assert_pymdp_metadata(
            step_result.metadata.get("pymdp"),
            f"nested grid[{cell}]",
        )
    nested_trace = nested_model.trace_over_nested_h3_grid(
        nested_observations,
        timestep=0,
        grid_result=nested_grid_result,
    )
    _assert_spatial_trace(
        nested_trace,
        expected_leaf_cells=len(nested_observations),
        label="nested grid trace",
    )
    assert any(
        row.metadata.get("aggregate_parent_cell")
        for row in nested_trace.cell_diagnostics
    ), "nested trace missing parent aggregate diagnostics"

    nested_agent = SpatialActiveInferenceAgent(
        initial_cells=nested_leaf_cells,
        h3_resolution=9,
        state_dim=4,
        obs_dim=4,
        enable_logging=False,
    )
    nested_agent.enable_nested_h3_spatial([7, 8, 9], cells=nested_leaf_cells)
    nested_agent_result = nested_agent.step_nested(
        nested_observations,
        return_result=True,
    )
    assert isinstance(nested_agent_result, NestedH3GridInferenceResult)
    assert nested_agent_result.nested_belief_update.level_summaries
    nested_agent_trace = nested_agent.trace_nested_step(
        nested_observations,
        grid_result=nested_agent_result,
        timestep=0,
    )
    _assert_spatial_trace(
        nested_agent_trace,
        expected_leaf_cells=len(nested_observations),
        label="nested spatial-agent trace",
    )

    nested_multi = MultiAgentModel(n_agents=2)
    nested_multi.enable_nested_h3_spatial([7, 8, 9], cells=nested_leaf_cells)
    nested_history = nested_multi.simulate_nested_h3_lattice(
        1,
        lambda _cell: np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert nested_history["parent_count"] > 0
    assert nested_history["nested_history"][0]["level_summaries"]

    broken = dict(hierarchy)
    broken["child_parent_map"] = dict(hierarchy["child_parent_map"])
    child = next(iter(broken["child_parent_map"]))
    del broken["child_parent_map"][child]
    broken_report = nested_grid.validate_h3_hierarchy(broken)
    assert not broken_report["is_valid"]
    assert broken_report["orphan_count"] >= 1


def main() -> int:

    _validate_no_h3_v3_calls()
    _validate_no_inert_h3_methods()
    _validate_no_obsolete_pymdp_runtime_imports()
    _validate_public_h3_docstrings()
    cells = _validate_space_indexing_contract()
    _validate_act_h3_runtime(cells)

    print("H3 Active Inference contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
