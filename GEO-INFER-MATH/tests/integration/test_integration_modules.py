"""Integration tests for expanded integration modules.

Tests the real mathematical implementations added to the
ACT, AI, and BAYES integration sub-packages.
"""

import numpy as np
import pytest


# ─── ACT Integration ────────────────────────────────────────────────


class TestFreeEnergyCalculator:
    """Tests for free energy calculations."""

    def test_variational_free_energy(self):
        from geo_infer_math.integration.act.free_energy import FreeEnergyCalculator

        calc = FreeEnergyCalculator()
        obs = np.array([0.8, 0.1, 0.1])
        beliefs = np.array([0.5, 0.3, 0.2])

        result = calc.calculate(obs, beliefs)
        assert "free_energy" in result
        assert "complexity" in result
        assert "accuracy" in result
        assert np.isfinite(result["free_energy"])
        assert result["complexity"] >= 0  # KL is non-negative

    def test_expected_free_energy(self):
        from geo_infer_math.integration.act.free_energy import FreeEnergyCalculator

        calc = FreeEnergyCalculator()
        beliefs = np.array([0.7, 0.2, 0.1])
        likelihood = np.eye(3) * 0.8 + 0.2 / 3

        G = calc.expected_free_energy(beliefs, likelihood)
        assert np.isfinite(G)

    def test_uniform_beliefs_zero_complexity(self):
        from geo_infer_math.integration.act.free_energy import FreeEnergyCalculator

        calc = FreeEnergyCalculator()
        obs = np.ones(4) / 4
        beliefs = np.ones(4) / 4
        prior = np.ones(4) / 4

        result = calc.calculate(obs, beliefs, prior=prior)
        assert abs(result["complexity"]) < 1e-10  # KL(uniform||uniform) ≈ 0


class TestBeliefUpdating:
    """Tests for belief update mechanisms."""

    def test_bayesian_update(self):
        from geo_infer_math.integration.act.belief_updating import BeliefUpdating

        updater = BeliefUpdating()
        prior = np.array([0.5, 0.3, 0.2])
        obs = np.array([0.9, 0.05, 0.05])
        likelihood = np.eye(3) * 0.9 + 0.1 / 3

        result = updater.update(prior, obs, likelihood=likelihood)
        assert "posterior" in result
        assert abs(result["posterior"].sum() - 1.0) < 1e-10
        assert result["kl_change"] >= 0

    def test_precision_weighted_update(self):
        from geo_infer_math.integration.act.belief_updating import BeliefUpdating

        updater = BeliefUpdating()
        beliefs = np.array([1.0, 2.0, 3.0])
        errors = np.array([0.5, -0.5, 0.0])

        updated = updater.precision_weighted_update(
            beliefs, errors, sensory_precision=2.0, prior_precision=1.0
        )
        assert len(updated) == 3
        # With 2:1 precision ratio, learning rate = 2/3 ≈ 0.667
        expected_lr = 2.0 / 3.0
        expected = beliefs + expected_lr * errors
        np.testing.assert_allclose(updated, expected, atol=1e-10)


class TestVariationalInference:
    """Tests for variational inference."""

    def test_vi_convergence(self):
        from geo_infer_math.integration.act.variational_inference import (
            VariationalInferenceHelpers,
        )

        vi = VariationalInferenceHelpers(max_iterations=50)
        obs = np.array([0.9, 0.05, 0.05])
        prior = np.ones(3) / 3
        likelihood = np.eye(3) * 0.9 + 0.1 / 3

        result = vi.perform_vi(obs, prior, likelihood=likelihood)
        assert "posterior" in result
        assert abs(result["posterior"].sum() - 1.0) < 1e-10
        assert len(result["elbo_history"]) > 0


class TestGenerativeModels:
    """Tests for generative model construction."""

    def test_categorical_model(self):
        from geo_infer_math.integration.act.generative_models import GenerativeModels

        builder = GenerativeModels()
        model = builder.create_generative_model(
            "categorical", {"n_states": 4, "n_obs": 4, "n_actions": 2}
        )
        assert "A" in model and "B" in model and "C" in model and "D" in model
        assert model["A"].shape == (4, 4)
        assert model["B"].shape == (4, 4, 2)
        # A columns should sum to 1
        np.testing.assert_allclose(model["A"].sum(axis=0), 1.0, atol=1e-10)

    def test_grid_world_model(self):
        from geo_infer_math.integration.act.generative_models import GenerativeModels

        builder = GenerativeModels()
        model = builder.create_generative_model("grid_world", {"grid_size": 3})
        assert model["A"].shape == (9, 9)  # 3×3 grid = 9 states
        assert model["B"].shape == (9, 9, 4)  # 4 actions


class TestPolicyOptimization:
    """Tests for policy optimization."""

    def test_policy_selection(self):
        from geo_infer_math.integration.act.policy_optimization import (
            PolicyOptimization,
        )

        po = PolicyOptimization(gamma=2.0)
        A = np.eye(3) * 0.8 + 0.2 / 3
        B = np.zeros((3, 3, 2))
        B[:, :, 0] = np.eye(3)  # Stay
        B[:, :, 1] = np.roll(np.eye(3), 1, axis=0)  # Shift

        policies = np.array([[0], [1]])  # One-step policies
        result = po.optimize_policy(policies, A, B)

        assert "selected_policy" in result
        assert "policy_probs" in result
        assert abs(result["policy_probs"].sum() - 1.0) < 1e-10


# ─── AI Integration ─────────────────────────────────────────────────


class TestSpatialLossFunctions:
    """Tests for spatial loss functions."""

    def test_spatial_mse(self):
        from geo_infer_math.integration.ai.loss_functions import SpatialLossFunctions

        loss_fn = SpatialLossFunctions()
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 2.9])

        result = loss_fn.calculate_loss(pred, target, loss_type="spatial_mse")
        assert result["loss"] > 0
        assert result["base_loss"] > 0

    def test_spatial_mse_with_coordinates(self):
        from geo_infer_math.integration.ai.loss_functions import SpatialLossFunctions

        loss_fn = SpatialLossFunctions()
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 2.9])
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)

        result = loss_fn.calculate_loss(
            pred, target, coordinates=coords, loss_type="spatial_mse"
        )
        assert result["spatial_penalty"] >= 0


class TestOptimizationBridges:
    """Tests for optimization bridges."""

    def test_gradient_descent(self):
        from geo_infer_math.integration.ai.optimization_bridges import (
            OptimizationBridges,
        )

        bridge = OptimizationBridges(learning_rate=0.1, max_iterations=200)

        def quadratic(x):
            return float(np.sum((x - 2.0) ** 2))

        result = bridge.bridge_optimize(quadratic, np.array([0.0, 0.0]))
        assert result["converged"]
        np.testing.assert_allclose(result["optimal_params"], [2.0, 2.0], atol=0.1)


class TestSpatialAttention:
    """Tests for spatial attention."""

    def test_basic_attention(self):
        from geo_infer_math.integration.ai.spatial_attention import SpatialAttention

        sa = SpatialAttention()
        n, d = 5, 8
        Q = np.random.randn(n, d)
        K = np.random.randn(n, d)
        V = np.random.randn(n, d)

        output, weights = sa.compute_attention_weights(Q, K, V)
        assert output.shape == (n, d)
        assert weights.shape == (n, n)
        # Attention weights should sum to 1 per row
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)

    def test_multi_head_attention(self):
        from geo_infer_math.integration.ai.spatial_attention import SpatialAttention

        sa = SpatialAttention()
        n, d = 6, 8
        Q = np.random.randn(n, d)
        K = np.random.randn(n, d)
        V = np.random.randn(n, d)

        output, weights = sa.multi_head_attention(Q, K, V, n_heads=4)
        assert output.shape == (n, d)
        assert weights.shape == (4, n, n)


class TestSpatialTensorOperations:
    """Tests for spatial tensor operations."""

    def test_distance_tensor(self):
        from geo_infer_math.integration.ai.tensor_operations import (
            SpatialTensorOperations,
        )

        ops = SpatialTensorOperations()
        coords = np.array([[0, 0], [3, 4], [1, 1]], dtype=float)

        distances = ops.compute_distance_tensor(coords)
        assert distances.shape == (3, 3)
        assert distances[0, 1] == pytest.approx(5.0)  # 3-4-5 triangle
        np.testing.assert_allclose(np.diag(distances), 0.0)

    def test_adjacency_tensor_threshold(self):
        from geo_infer_math.integration.ai.tensor_operations import (
            SpatialTensorOperations,
        )

        ops = SpatialTensorOperations()
        coords = np.array([[0, 0], [1, 0], [10, 0]], dtype=float)

        adj = ops.build_adjacency_tensor(coords, threshold=2.0)
        assert adj[0, 1] == 1.0  # Within threshold
        assert adj[0, 2] == 0.0  # Beyond threshold

    def test_convolution_kernel_gaussian(self):
        from geo_infer_math.integration.ai.tensor_operations import (
            SpatialTensorOperations,
        )

        ops = SpatialTensorOperations()
        kernel = ops.spatial_convolution_kernel(3, kernel_type="gaussian")
        assert kernel.shape == (3, 3)
        assert abs(kernel.sum() - 1.0) < 1e-10


# ─── BAYES Integration ──────────────────────────────────────────────


class TestPosteriorHelpers:
    """Tests for conjugate posterior computations."""

    def test_normal_normal(self):
        from geo_infer_math.integration.bayes.posterior_helpers import PosteriorHelpers

        ph = PosteriorHelpers()
        data = np.random.randn(100) + 5.0  # Mean ≈ 5
        result = ph.calculate_posterior(
            data,
            {"mu_0": 0.0, "sigma_0": 10.0, "sigma": 1.0},
            family="normal_normal",
        )
        assert abs(result["mu_n"] - 5.0) < 1.0  # Should be near true mean

    def test_beta_binomial(self):
        from geo_infer_math.integration.bayes.posterior_helpers import PosteriorHelpers

        ph = PosteriorHelpers()
        data = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        result = ph.calculate_posterior(
            data, {"alpha": 1.0, "beta": 1.0}, family="beta_binomial"
        )
        assert result["alpha_n"] == 8.0  # 1 + 7 successes
        assert result["beta_n"] == 4.0  # 1 + 3 failures


class TestPriorBuilders:
    """Tests for prior distribution builders."""

    def test_uniform_prior(self):
        from geo_infer_math.integration.bayes.prior_builders import PriorBuilders

        pb = PriorBuilders()
        prior = pb.build_prior("uniform", size=10)
        assert len(prior) == 10
        assert abs(prior.sum() - 1.0) < 1e-10
        np.testing.assert_allclose(prior, 0.1, atol=1e-10)

    def test_normal_prior(self):
        from geo_infer_math.integration.bayes.prior_builders import PriorBuilders

        pb = PriorBuilders()
        prior = pb.build_prior("normal", size=20, mean=10, std=2)
        assert len(prior) == 20
        assert abs(prior.sum() - 1.0) < 1e-10
        # Peak should be near index 10
        assert np.argmax(prior) == 10


class TestMCMCHelpers:
    """Tests for MCMC sampling."""

    def test_metropolis_hastings(self):
        from geo_infer_math.integration.bayes.mcmc_helpers import MCMCHelpers

        # Determinism pin: the 200-sample chain is too short for an unseeded
        # tolerance bound (CI 3.12 drew 1.177- and 1.035-off means). The
        # sampler draws from its own generator, so the legacy global
        # np.random.seed never reached it; seed the owning rng instead.
        mcmc = MCMCHelpers(
            n_samples=200, burn_in=50, proposal_std=0.5, rng=np.random.default_rng(0)
        )

        # Sample from N(3, 1)
        def log_posterior(x):
            return -0.5 * np.sum((x - 3.0) ** 2)

        result = mcmc.mcmc_sample(log_posterior, np.array([0.0]))
        assert result["samples"].shape[0] == 200
        assert 0 < result["acceptance_rate"] < 1
        assert abs(np.mean(result["samples"]) - 3.0) < 1.0


class TestModelSelection:
    """Tests for model selection criteria."""

    def test_bic_aic_comparison(self):
        from geo_infer_math.integration.bayes.model_selection import ModelSelection

        ms = ModelSelection()

        models = [
            {"name": "simple", "log_likelihood": -100, "n_params": 2, "n_obs": 50},
            {"name": "complex", "log_likelihood": -95, "n_params": 10, "n_obs": 50},
        ]

        for method in ["bic", "aic"]:
            result = ms.compare_models(models, method=method)
            assert result["best_model"] in ["simple", "complex"]
            assert len(result["rankings"]) == 2
            assert result["deltas"][result["best_model"]] == 0.0

    def test_bayes_factor(self):
        from geo_infer_math.integration.bayes.model_selection import ModelSelection

        ms = ModelSelection()

        result = ms.bayes_factor(log_evidence_1=-50.0, log_evidence_2=-55.0)
        assert result["log_bf"] == pytest.approx(5.0)
        assert result["bf"] > 1.0


class TestBayesianOptimization:
    """Tests for Bayesian optimization."""

    def test_simple_optimization(self):
        from geo_infer_math.integration.bayes.bayesian_optimization import (
            BayesianOptimization,
        )

        bo = BayesianOptimization(n_initial=3, max_iterations=5, length_scale=0.5)

        def sphere(x):
            return float(np.sum((x - 0.5) ** 2))

        result = bo.optimize(sphere, bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))
        assert result["best_y"] < 0.5  # Should find something below 0.5
        assert result["n_evaluations"] == 8  # 3 initial + 5 BO
