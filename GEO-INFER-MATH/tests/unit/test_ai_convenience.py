"""
Tests for GEO-INFER-MATH AI/ML convenience API.

Tests cover: gradient_helper, spatial_loss_function, AIConvenience class.
"""

import numpy as np
import pytest


from geo_infer_math.api.convenience.ai_convenience import (
    gradient_helper,
    spatial_loss_function,
    AIConvenience,
)


class TestGradientHelper:
    """Tests for gradient computation."""

    def test_gradient_of_quadratic(self):
        # f(x) = x^2, f'(x) = 2x
        def f(x):
            return float(np.sum(x**2))

        params = np.array([3.0])
        grad = gradient_helper(f, params)
        assert abs(grad[0] - 6.0) < 1e-4

    def test_gradient_multidimensional(self):
        # f(x,y) = x^2 + y^2, grad = [2x, 2y]
        def f(x):
            return float(np.sum(x**2))

        params = np.array([1.0, 2.0])
        grad = gradient_helper(f, params)
        assert abs(grad[0] - 2.0) < 1e-4
        assert abs(grad[1] - 4.0) < 1e-4

    def test_gradient_at_minimum(self):
        def f(x):
            return float(np.sum(x**2))

        params = np.array([0.0, 0.0])
        grad = gradient_helper(f, params)
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-4)

    def test_automatic_method_fallback(self):
        def f(x):
            return float(np.sum(x**2))

        params = np.array([2.0])
        grad = gradient_helper(f, params, method="automatic")
        assert abs(grad[0] - 4.0) < 1e-4


class TestSpatialLossFunction:
    """Tests for spatial loss functions."""

    def test_mse_loss(self):
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0])
        loss = spatial_loss_function(predictions, targets, loss_type="mse")
        assert abs(loss) < 1e-10

    def test_mse_loss_nonzero(self):
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])
        loss = spatial_loss_function(predictions, targets, loss_type="mse")
        assert abs(loss - 1.0) < 1e-10

    def test_mae_loss(self):
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])
        loss = spatial_loss_function(predictions, targets, loss_type="mae")
        assert abs(loss - 1.0) < 1e-10

    def test_huber_loss(self):
        predictions = np.array([1.0, 2.0])
        targets = np.array([1.0, 2.0])
        loss = spatial_loss_function(predictions, targets, loss_type="huber")
        assert abs(loss) < 1e-10

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            spatial_loss_function(
                np.array([1.0]), np.array([1.0]), loss_type="nonexistent"
            )

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="same length"):
            spatial_loss_function(np.array([1.0, 2.0]), np.array([1.0]))

    def test_spatial_regularization(self):
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0])
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        loss_no_reg = spatial_loss_function(predictions, targets, loss_type="mse")
        loss_with_reg = spatial_loss_function(
            predictions,
            targets,
            coordinates=coords,
            loss_type="mse",
            spatial_weight=0.1,
        )
        # With regularization, loss should be higher or equal
        assert loss_with_reg >= loss_no_reg


class TestAIConvenience:
    """Tests for AIConvenience class."""

    def test_initialization(self):
        ai = AIConvenience()
        assert hasattr(ai, "logger")
        assert hasattr(ai, "_gradient_cache")

    def test_compute_gradient(self):
        ai = AIConvenience()

        def f(x):
            return float(np.sum(x**2))

        grad = ai.compute_gradient(f, np.array([2.0, 3.0]))
        assert abs(grad[0] - 4.0) < 1e-4
        assert abs(grad[1] - 6.0) < 1e-4

    def test_calculate_loss(self):
        ai = AIConvenience()
        loss = ai.calculate_loss(
            np.array([1.0, 2.0]), np.array([1.5, 2.5]), loss_type="mse"
        )
        assert abs(loss - 0.25) < 1e-10
