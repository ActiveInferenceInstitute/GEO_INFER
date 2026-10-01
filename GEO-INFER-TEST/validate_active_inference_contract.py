#!/usr/bin/env python3
"""
Validate the canonical GEO-INFER Active Inference API contract.

This check is intentionally small and executable. It verifies that ACT exports
the typed result objects, free-energy decomposition is mathematically coherent,
policy selection minimizes expected free energy in deterministic mode, and a
full ActiveInferenceModel step can return the typed step result.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
ACT_SRC = REPO_ROOT / "GEO-INFER-ACT" / "src"
SPACE_SRC = REPO_ROOT / "GEO-INFER-SPACE" / "src"
ACT_PACKAGE = ACT_SRC / "geo_infer_act"
ACT_DOCS = REPO_ROOT / "GEO-INFER-ACT" / "docs"

PUBLIC_DOCSTRING_FILES = [
    ACT_PACKAGE / "api" / "interface.py",
    ACT_PACKAGE / "core" / "active_inference.py",
    ACT_PACKAGE / "core" / "belief_updating.py",
    ACT_PACKAGE / "core" / "dynamic_causal_model.py",
    ACT_PACKAGE / "core" / "free_energy.py",
    ACT_PACKAGE / "core" / "generative_model.py",
    ACT_PACKAGE / "core" / "markov_decision_process.py",
    ACT_PACKAGE / "core" / "policy_selection.py",
    ACT_PACKAGE / "core" / "spatial_agent.py",
    ACT_PACKAGE / "core" / "variational_inference.py",
    ACT_PACKAGE / "models" / "base.py",
    ACT_PACKAGE / "models" / "multi_agent.py",
    ACT_PACKAGE / "models" / "urban.py",
]

STALE_DOC_SYMBOLS = [
    "EnvironmentalAgent",
    "SurveyAgent",
    "TrackingAgent",
    "SwarmCoordinator",
]


def _validate_no_inert_methods() -> None:
    """Fail when ACT source contains pass or NotImplemented placeholders."""
    offenders: list[str] = []
    for path in sorted(ACT_PACKAGE.rglob("*.py")):
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
    assert not offenders, "Inert ACT method bodies found:\n" + "\n".join(offenders)


def _validate_public_method_docs() -> None:
    """Fail when public ACT methods in core surfaces lack docstrings."""
    missing: list[str] = []
    for path in PUBLIC_DOCSTRING_FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                if not ast.get_docstring(node):
                    missing.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.name}"
                    )
    assert not missing, "Public ACT methods missing docstrings:\n" + "\n".join(missing)


def _validate_docs_reference_real_symbols() -> None:
    """Fail when ACT docs advertise removed high-level class names."""
    stale_hits: list[str] = []
    for path in sorted(ACT_DOCS.glob("*.md")):
        text = path.read_text()
        for symbol in STALE_DOC_SYMBOLS:
            if symbol in text:
                stale_hits.append(f"{path.relative_to(REPO_ROOT)}: {symbol}")
    assert not stale_hits, "Stale Active Inference docs references:\n" + "\n".join(
        stale_hits
    )


def main() -> int:

    from geo_infer_act import (
        ActiveInferenceModel,
        ActiveInferenceStepResult,
        FreeEnergyBreakdown,
        GenerativeModel,
        H3BeliefUpdateResult,
        H3SpatialConsistency,
        PolicyEvaluation,
        PolicySelector,
    )
    from geo_infer_act.core.free_energy import FreeEnergyCalculator

    calc = FreeEnergyCalculator()
    breakdown = calc.compute_categorical_free_energy(
        np.array([0.7, 0.2, 0.1]),
        np.array([0.8, 0.15, 0.05]),
        np.array([0.6, 0.25, 0.15]),
        return_breakdown=True,
    )
    assert isinstance(breakdown, FreeEnergyBreakdown)
    assert np.isfinite(breakdown.free_energy)
    assert np.isclose(breakdown.free_energy, breakdown.complexity - breakdown.accuracy)

    policy_breakdown = calc.compute_expected_free_energy(
        np.array([0.5, 0.3, 0.2]),
        {"predicted_beliefs": [0.2, 0.7, 0.1], "exploration_bonus": 0.2},
        np.array([0.1, 0.8, 0.1]),
        return_breakdown=True,
    )
    assert isinstance(policy_breakdown, FreeEnergyBreakdown)
    assert np.isfinite(policy_breakdown.free_energy)

    selector = PolicySelector(selection_mode="deterministic", random_seed=1)
    policy_result = selector.select_policy(
        np.array([0.5, 0.3, 0.2]),
        [
            {"action": "higher-cost", "expected_free_energy": 1.0},
            {"action": "lower-cost", "expected_free_energy": -0.5},
        ],
    )
    assert policy_result["policy"]["action"] == "lower-cost"
    assert isinstance(policy_result["evaluation"], PolicyEvaluation)

    model = ActiveInferenceModel(
        model_type="categorical",
        policy_selection_mode="deterministic",
        random_seed=1,
    )
    model.set_generative_model(
        GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 2})
    )
    step = model.step(np.array([1.0, 0.0]), return_result=True)
    assert isinstance(step, ActiveInferenceStepResult)
    assert step.beliefs is not None
    assert step.action is not None
    assert np.isfinite(step.free_energy)

    gaussian_model = ActiveInferenceModel(model_type="gaussian")
    gaussian_model.set_generative_model(
        GenerativeModel(
            "gaussian",
            {
                "state_dim": 2,
                "obs_dim": 2,
                "mean": np.zeros(2),
                "precision": np.eye(2),
            },
        )
    )
    gaussian_beliefs = gaussian_model.perceive(np.array([1.0, 0.0]))
    assert "mean" in gaussian_beliefs
    assert "precision" in gaussian_beliefs
    assert np.all(np.isfinite(gaussian_beliefs["mean"]))

    h3_boundary = {
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
    h3_model = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 3})
    h3_model.enable_h3_spatial(8, h3_boundary)
    observed_cells = h3_model.h3_cells[: min(2, len(h3_model.h3_cells))]
    h3_result = h3_model.update_h3_beliefs(
        {cell: np.eye(3)[index % 3] for index, cell in enumerate(observed_cells)},
        return_result=True,
    )
    assert isinstance(h3_result, H3BeliefUpdateResult)
    consistency = h3_result.spatial_consistency
    assert isinstance(consistency, H3SpatialConsistency)
    assert np.isfinite(consistency.global_coherence)
    assert np.isfinite(consistency.neighbor_correlations)
    assert np.isfinite(h3_result.aggregate_free_energy)

    _validate_no_inert_methods()
    _validate_public_method_docs()
    _validate_docs_reference_real_symbols()

    print("Active Inference contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
