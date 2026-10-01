"""Integration smoke test for GEO-INFER-COG.

Runs a full perception -> working memory -> reasoning -> memory update ->
decision cycle through ``CognitiveProcessingEngine`` on synthetic input and
asserts that the resulting cognitive state advances deterministically: two
engines built with the same seed produce identical results, up to keys that
embed wall-clock time or module-level counters (IDs, timestamps).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest

from geo_infer_cog.core.cognitive_engine import CognitiveProcessingEngine

#: Result keys that embed wall-clock time or process-level counters and are
#: therefore excluded from the cross-run determinism comparison.
VOLATILE_KEYS = {
    "processing_id",
    "analysis_id",
    "item_id",
    "reasoning_id",
    "step_id",
    "timestamp",
    "processing_time",
    "creation_time",
    "last_access_time",
}


def _synthetic_spatial_data() -> dict[str, Any]:
    """Small, fixed spatial scene used for every integration run."""
    return {
        "geometries": [
            {"type": "Point", "coordinates": [0.0, 0.0]},
            {"type": "Point", "coordinates": [1.5, 2.0]},
            {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                ],
            },
        ],
        "attributes": {"region": "integration_test", "scale": "local"},
    }


def _strip_volatile(value: Any) -> Any:
    """Recursively drop volatile keys and normalise datetimes for comparison."""
    if isinstance(value, dict):
        return {
            key: _strip_volatile(sub)
            for key, sub in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_volatile(item) for item in value]
    if isinstance(value, datetime):
        return "<datetime>"
    return value


def _run_cycle(seed: int) -> tuple[dict[str, Any], CognitiveProcessingEngine]:
    """Build a seeded engine, process one synthetic scene, return the result."""
    engine = CognitiveProcessingEngine(rng=np.random.default_rng(seed))
    result = engine.process_spatial_input(
        _synthetic_spatial_data(),
        context={"task_type": "navigation"},
    )
    return result, engine


class TestCognitiveIntegration:
    """End-to-end cognitive pipeline integration checks."""

    @pytest.mark.integration
    def test_perception_memory_decision_cycle_advances_state(self) -> None:
        result, engine = _run_cycle(seed=42)

        # Every pipeline stage reported back.
        for stage in (
            "perception_result",
            "reasoning_result",
            "memory_result",
            "decision_result",
            "cognitive_state",
        ):
            assert stage in result, f"missing pipeline stage: {stage}"

        # State actually advanced: items were perceived, stored and decided on.
        perception = result["perception_result"]
        assert len(perception["spatial_elements"]) == 3
        assert perception["perception_metrics"]["elements_processed"] == 3
        assert perception["perception_metrics"]["attention_allocations"] == 1

        memory = result["memory_result"]
        assert memory["items_stored"] > 0
        assert engine.performance_metrics["perception_updates"] == 1
        assert engine.performance_metrics["reasoning_chains"] == 1
        assert engine.performance_metrics["memory_operations"] == 1
        assert engine.performance_metrics["decisions_made"] == 1

        # Working memory holds the spatial facts added during processing.
        assert engine.state.working_memory, "working memory should not be empty"
        assert engine.state.cognitive_load > 0.0

    @pytest.mark.integration
    def test_cycle_is_deterministic_for_fixed_seed(self) -> None:
        """Two seeded engines produce identical pipelines, stage by stage."""
        first, _ = _run_cycle(seed=42)
        second, _ = _run_cycle(seed=42)

        assert _strip_volatile(first) == _strip_volatile(second)

        # Spot-check the most behaviour-bearing fields directly.
        assert (
            first["perception_result"]["attention_weights"]
            == second["perception_result"]["attention_weights"]
        )
        assert (
            first["memory_result"]["items_stored"]
            == second["memory_result"]["items_stored"]
        )
        assert (
            first["cognitive_state"]["cognitive_load"]
            == second["cognitive_state"]["cognitive_load"]
        )

    @pytest.mark.integration
    def test_different_seed_can_diverge(self) -> None:
        """A different seed is not silently aliased to the default stream."""
        default_run, _ = _run_cycle(seed=42)
        other_run, _ = _run_cycle(seed=7)

        # IDs embed generator draws; with distinct seeds they must differ.
        assert (
            default_run["perception_result"]["processing_id"]
            != other_run["perception_result"]["processing_id"]
        )
