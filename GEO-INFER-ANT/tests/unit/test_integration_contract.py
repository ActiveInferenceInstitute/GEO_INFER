"""Regression tests for ANT integration contracts."""

import asyncio
import logging
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from geo_infer_ant.core.population import AgentPopulation
from geo_infer_ant.core.agent_base import SwarmAgent


def test_package_imports_without_optional_integrations():
    """geo_infer_ant imports and works with no sibling workspace modules installed."""
    probe = textwrap.dedent(
        """
        import sys

        class _BlockIntegrations:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {
                    "geo_infer_act",
                    "geo_infer_space",
                    "geo_infer_agent",
                }:
                    raise ModuleNotFoundError(f"{name} intentionally blocked", name=name.split(".")[0])
                return None

        sys.meta_path.insert(0, _BlockIntegrations())

        import geo_infer_ant

        agent = geo_infer_ant.SwarmAgent("standalone", [37.7, -122.4])
        assert agent.agent_id == "standalone"
        assert agent.active_inference_model is None
        assert agent.spatial_indexer is None
        assert agent.spatial_analytics is None
        assert agent.to_dict()["state"] == {}
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_perceive_environment_preserves_unconfigured_active_inference_context(
    caplog,
):
    """The explicit rule backend retains sensory context without claiming ACT."""
    agent = SwarmAgent("integration-agent", np.array([37.7, -122.4]))

    with caplog.at_level(logging.WARNING):
        sensory_input = asyncio.run(
            agent.perceive_environment(
                environmental_signals={"temperature": 18.0},
            )
        )

    processed = sensory_input.process()
    assert processed["env_temperature"] == 18.0
    assert "active_inference_observations" not in processed
    assert agent.active_inference_enabled is False
    assert "Active Inference processing failed" not in caplog.text
    assert "Spatial analysis failed" not in caplog.text


def test_population_social_context_counts_nearby_agents_without_cell_api(caplog):
    """Population social context uses agent positions and stays warning-free."""
    population = AgentPopulation(population_size=2, agent_types=["worker"])
    population.agents[0].position = np.array([0.0, 0.0])
    population.agents[1].position = np.array([0.5, 0.0])
    population.agents[0].sensory_range = 1.0

    with caplog.at_level(logging.WARNING):
        context = population._get_social_context(population.agents[0])

    assert context["nearby_agents"] == 1
    assert context["nearby_agent_types"] == {"worker": 1}
    assert "Spatial neighbor search failed" not in caplog.text


def test_enabling_act_requires_a_real_configured_model():
    """An enable flag cannot manufacture inference from an unconfigured model."""
    with pytest.raises(ValueError, match="configured active_inference_model"):
        SwarmAgent("unconfigured", [0, 0], active_inference_enabled=True)


def test_internal_optional_import_failure_propagates():
    """A broken installed integration must not be classified as absent."""
    probe = textwrap.dedent(
        """
        import sys
        class BrokenIntegration:
            def find_spec(self, name, path=None, target=None):
                if name == 'geo_infer_act':
                    raise ModuleNotFoundError('broken ACT transitive dependency', name='missing_internal_dependency')
                return None
        sys.meta_path.insert(0, BrokenIntegration())
        import geo_infer_ant
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "broken ACT transitive dependency" in result.stderr
