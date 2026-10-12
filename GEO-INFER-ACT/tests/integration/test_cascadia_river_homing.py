"""Cascadia river homing: fixtures, closed loop, light cone and identity checks."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from geo_infer_act.core.gnn_contract import GNNArtifact
from geo_infer_act.models.river_homing import RiverHomingParameters

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cascadia/cascadia_homing.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("cascadia_homing", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    """The example module loaded from its file."""
    return _load_example()


@pytest.fixture(scope="module")
def compiled(example):
    """Config, compiled model and rendered sources."""
    config = example.load_config()
    model, features, provenance = example.build_model(config)
    return config, model, example.render_sources(model, config, provenance)


@pytest.fixture(scope="module")
def manifest(example, tmp_path_factory):
    """A 30-step run without figures from the committed artifact."""
    output = tmp_path_factory.mktemp("cascadia")
    return example.run(output=output, figures=False, steps=30)


def test_committed_fixtures_match_a_fresh_compile(example, compiled) -> None:
    """GNN source, state IDs, sidecar and GNN-exported artifact are all fresh."""
    config, model, sources = compiled
    assert example.fixture_problems(model, sources, config["data"]["fixtures"]) == []
    export = json.loads(
        (config["data"]["fixtures"] / example.FIXTURE_NAMES["export"]).read_text()
    )
    assert export["source_sha256"] == example.sha256_bytes(sources["source"].encode())
    artifact = GNNArtifact.load(
        config["data"]["fixtures"] / example.FIXTURE_NAMES["artifact"]
    )
    assert artifact.digest == export["artifact_canonical_sha256"]


def test_exact_tables_are_stochastic_and_dyadic(compiled) -> None:
    """Every A/B column, D and E sum to exactly one with power-of-two denominators."""
    _, model, _ = compiled
    states, actions = len(model.D), len(model.E)
    for state in range(states):
        assert sum(row[state] for row in model.A) == 1
        for action in range(actions):
            assert sum(model.B[nxt][state][action] for nxt in range(states)) == 1
    assert sum(model.D) == 1 and sum(model.E) == 1
    assert model.denominators() == {"A": 64, "B": 4, "D": 4, "E": 8}


def test_transitions_are_local_for_the_typed_relations(compiled) -> None:
    """Positive B mass only stays put or follows successor/upstream/side."""
    _, model, _ = compiled
    network = model.network
    allowed = {(a, b) for _, a, b in network.relation_pairs()} | {
        (c, c) for c in network.cells
    }
    cells = network.cells
    for nxt, plane in enumerate(model.B):
        for previous, row in enumerate(plane):
            if any(row):
                assert (cells[previous], cells[nxt]) in allowed


def test_parameters_reject_inexact_probabilities() -> None:
    """Float probabilities and non-dyadic fractions are rejected."""
    with pytest.raises(ValueError, match="exact fractions"):
        RiverHomingParameters.from_mapping({"upstream_move": 0.5})
    with pytest.raises(ValueError, match="dyadic"):
        RiverHomingParameters.from_mapping({"upstream_move": "1/3"})


def test_closed_loop_reaches_rowdy_creek_inside_the_light_cone(manifest) -> None:
    """The salmon homes no faster than the cone allows; replay and identity hold."""
    summary = manifest["summary"]
    assert summary["home_lower_bound_steps"] == 13
    assert summary["home_prior_mass_before_bound"] == 0.0
    assert summary["prior_light_cone_violations"] == []
    assert summary["posterior_support_violations"] == []
    assert summary["entered_natal_step"] is not None
    assert summary["reached_home_step"] > summary["home_lower_bound_steps"]
    assert summary["final_posterior_natal"] > 0.99
    assert summary["max_pymdp_identity_gap"] < 1e-5
    assert manifest["fixtures_fresh"] is True


def test_reference_run_events_are_robust(manifest) -> None:
    """Seed 7: Rowdy Creek at step 23, home at 26, no near-tied decisions."""
    summary = manifest["summary"]
    assert summary["min_decision_margin"] > 1e-3
    assert (
        summary["entered_river_step"],
        summary["entered_natal_step"],
        summary["reached_home_step"],
    ) == (1, 23, 26)
    assert summary["tributary_steps"] == 0 and summary["slough_steps"] == 0


def test_first_update_matches_exact_bayes(manifest) -> None:
    """The float posterior after the first observation equals the exact fraction."""
    summary = manifest["summary"]
    exact = summary["exact_first_update"]
    assert exact["evidence"] == "57/128"
    for cell, value in summary["first_posterior_float"].items():
        assert value == pytest.approx(float(Fraction(exact[cell])), abs=1e-12)


def test_run_manifest_hashes_written_files(manifest, tmp_path) -> None:
    """Manifest file digests are SHA-256 values and the trace is replayable JSON."""
    assert set(manifest["files"]) >= {
        "closed_loop.json",
        "diagnostics.json",
        "summary.json",
    }
    assert all(len(digest) == 64 for digest in manifest["files"].values())
    assert manifest["versions"]["inferactively-pymdp"] == "1.0.3"


def test_figures_and_animation_are_rendered(example, tmp_path) -> None:
    """A short run writes every figure, the GIF and manifest-listed hashes."""
    manifest = example.run(output=tmp_path, figures=True, steps=4)
    figures = sorted(name for name in manifest["files"] if name.startswith("figures/"))
    assert {
        "figures/cascadia_context.png",
        "figures/smith_river_network.png",
        "figures/belief_snapshots.png",
        "figures/free_energy_trace.png",
        "figures/efe_decomposition.png",
        "figures/light_cone.png",
        "figures/belief_animation.gif",
    } <= set(figures)
    for name in figures:
        assert (tmp_path / name).stat().st_size > 1000
    closed = json.loads((tmp_path / "closed_loop.json").read_text())
    beliefs = np.array([step["posterior"] for step in closed["trace"]["steps"]])
    assert np.allclose(beliefs.sum(axis=1), 1.0)
