"""
Integration tests for ACT-SPACE module integration.

Tests the cross-module integration between GEO-INFER-ACT and GEO-INFER-SPACE,
verifying that the geo_infer_space import is safely wrapped in try/except
and that spatial functions work with or without the SPACE module available.
"""

import numpy as np


class TestSpaceImportSafety:
    """Test that SPACE module imports are safely wrapped."""

    def test_integration_module_imports_without_space(self) -> None:
        """Test that the integration module loads even without geo_infer_space."""
        # This should never raise ImportError since the import is wrapped
        from geo_infer_act.utils.integration import integrate_space

        assert callable(integrate_space)

    def test_space_h3_fallback(self) -> None:
        """Test that space_h3 variable is None when SPACE unavailable."""
        from geo_infer_act.utils import integration

        # space_h3 is either a valid module or None
        # Either way, the module should have loaded
        assert hasattr(integration, "space_h3")

    def test_h3_spatial_model_creation(self) -> None:
        """Test H3 spatial model creation (uses h3 directly, not space module)."""
        from geo_infer_act.utils.integration import create_h3_spatial_model

        import h3  # noqa: F401 -- hard root dependency (h3>=4.5.0); plain import

        result = create_h3_spatial_model(
            config={},
            h3_resolution=8,  # r=4 produces 0 cells for a ~100 km² polygon
            boundary={
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.5, 37.7],
                        [-122.4, 37.7],
                        [-122.4, 37.8],
                        [-122.5, 37.8],
                        [-122.5, 37.7],
                    ]
                ],
            },
        )

        assert result["status"] == "success"
        assert "model_config" in result
        assert "boundary_cells" in result["model_config"]

    def test_h3_spatial_model_enforces_cell_budget(self) -> None:
        """Large fills fail before an unbounded model is constructed."""
        from geo_infer_act.utils.integration import create_h3_spatial_model

        result = create_h3_spatial_model(
            config={"max_cells": 1},
            h3_resolution=8,
            boundary={
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.5, 37.7],
                        [-122.4, 37.7],
                        [-122.4, 37.8],
                        [-122.5, 37.8],
                        [-122.5, 37.7],
                    ]
                ],
            },
        )

        assert result["status"] == "error"
        assert "max_cells" in result["message"]


class TestSpaceIntegrationFunction:
    """Test the integrate_space function behavior."""

    def test_disabled_integration_returns_empty(self) -> None:
        """Test that disabled integration returns empty dict."""
        from geo_infer_act.utils.integration import integrate_space

        result = integrate_space(config={})
        assert result == {}

    def test_no_endpoint_returns_empty(self) -> None:
        """Test that missing endpoint returns empty dict."""
        from geo_infer_act.utils.integration import integrate_space

        config = {
            "integration": {
                "space_module": {
                    "enabled": True
                    # No api_endpoint
                }
            }
        }
        result = integrate_space(config)
        assert result == {}

    def test_invalid_endpoint_returns_error(self) -> None:
        """Test that invalid endpoint returns error gracefully."""
        from geo_infer_act.utils.integration import integrate_space

        config = {
            "integration": {
                "space_module": {
                    "enabled": True,
                    "api_endpoint": "nonexistent.module.Class",
                }
            }
        }
        data = {"action": "test"}
        result = integrate_space(config, data)
        assert result.get("status") == "error"


class TestMultiAgentSpatialIntegration:
    """Test multi-agent model spatial integration capabilities."""

    def test_multi_agent_model_has_spatial_mode(self) -> None:
        """Test that MultiAgentModel can be created with spatial attributes."""
        from geo_infer_act.models.multi_agent import MultiAgentModel

        model = MultiAgentModel(n_agents=2, n_locations=3)
        assert hasattr(model, "spatial_mode")
        assert model.spatial_mode is False

    def test_multi_agent_coordination_without_spatial(self) -> None:
        """Test coordination works in non-spatial mode."""
        from geo_infer_act.models.multi_agent import MultiAgentModel

        model = MultiAgentModel(n_agents=3, n_locations=4)
        result = model.coordinate_agents()
        assert "coordination_matrix" in result
        assert "average_coordination" in result
        assert result["coordination_matrix"].shape[0] == 3

    def test_multi_agent_step_integration(self) -> None:
        """Test that multi-agent step works regardless of spatial module."""
        from geo_infer_act.models.multi_agent import MultiAgentModel

        model = MultiAgentModel(n_agents=2, n_locations=3)
        state, done = model.step()
        assert "resource_distribution" in state
        assert "agent_locations" in state
        assert not done


class TestModernToolsIntegration:
    """Test the modern tools integration hub."""

    def test_integration_hub_creates(self) -> None:
        """Test that ModernToolsIntegration can be instantiated."""
        from geo_infer_act.utils.integration import ModernToolsIntegration

        hub = ModernToolsIntegration()
        assert isinstance(hub.available_tools, dict)
        # Should have checked for various tools
        expected_keys = ["rxinfer", "bayeux", "pymdp", "pymc", "pyro", "jax"]
        for key in expected_keys:
            assert key in hub.available_tools

    def test_rxinfer_explicit_local_gaussian_contract(self) -> None:
        """Test deterministic local Gaussian behavior without Julia."""
        from geo_infer_act.utils.integration import ModernToolsIntegration

        hub = ModernToolsIntegration()
        result = hub.create_rxinfer_model(
            "", {"observations": np.array([1.0, 2.0, 3.0])}, backend="local_gaussian"
        )
        assert result["status"] == "success"
        assert result["backend"] == "local_gaussian"
        assert result["posterior_marginals"]["mean"] == 1.5
        assert result["posterior_marginals"]["variance"] == 0.25
