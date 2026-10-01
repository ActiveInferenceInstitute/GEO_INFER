"""
Tests for GEO-INFER-MATH integration convenience API.

Tests cover: cross_module_helper and IntegrationConvenience class.
"""

import numpy as np
import pytest


from geo_infer_math.api.convenience.integration_convenience import (
    cross_module_helper,
    IntegrationConvenience,
)


class TestCrossModuleHelper:
    """Tests for cross_module_helper function."""

    def test_bayes_posterior_operation(self):
        prior = np.array([0.5, 0.3, 0.2])

        def likelihood(d, p):
            return np.array([0.1, 0.8, 0.1])

        data = np.array([1.0])
        result = cross_module_helper(
            "bayes",
            "posterior",
            {"prior": prior, "likelihood": likelihood, "data": data},
        )
        assert result is not None
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_ai_gradient_operation(self):
        def f(x):
            return float(np.sum(x**2))

        result = cross_module_helper(
            "ai", "gradient", {"function": f, "parameters": np.array([2.0])}
        )
        assert result is not None
        assert abs(result[0] - 4.0) < 1e-4

    def test_unknown_module_raises(self):
        with pytest.raises(ValueError, match="Unknown module"):
            cross_module_helper("nonexistent", "op", {})

    def test_unknown_operation_returns_none(self):
        result = cross_module_helper("bayes", "nonexistent_op", {})
        assert result is None


class TestIntegrationConvenience:
    """Tests for IntegrationConvenience class."""

    def test_initialization(self):
        ic = IntegrationConvenience()
        assert hasattr(ic, "logger")
        assert hasattr(ic, "_module_registry")

    def test_execute_cross_module_bayes(self):
        ic = IntegrationConvenience()
        prior = np.array([0.5, 0.3, 0.2])

        def likelihood(d, p):
            return np.array([0.2, 0.6, 0.2])

        data = np.array([1.0])
        result = ic.execute_cross_module(
            "bayes",
            "posterior",
            {"prior": prior, "likelihood": likelihood, "data": data},
        )
        assert result is not None
        assert len(result) == 3
