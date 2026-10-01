"""
Contract tests for distributional uncertainty quantification.

Covers calibrated prediction intervals, the epistemic / aleatoric uncertainty
decomposition, and posterior predictive sampling on the spatial GP and
hierarchical models, plus the predictive-interval and epistemic-uncertainty
views on :class:`PosteriorAnalysis` and the estimator telemetry that makes the
trajectory of a run auditable (ELBO history for VI, acceptance for MCMC/HMC).
"""

import numpy as np
import pytest
from typing import Any


from geo_infer_bayes.core.inference import BayesianInference
from geo_infer_bayes.core.evaluation import empirical_coverage
from geo_infer_bayes.core.posterior import PosteriorAnalysis
from geo_infer_bayes.core.variational import VariationalInference
from geo_infer_bayes.core.mcmc import MCMC
from geo_infer_bayes.core.hmc import HMC
from geo_infer_bayes.models.base import BayesianModel
from geo_infer_bayes.models.spatial_gp import SpatialGP
from geo_infer_bayes.models.hierarchical import HierarchicalBayesianModel


class _Posterior:
    """Minimal stand-in exposing only the ``samples`` mapping models read."""

    def __init__(self, samples: dict) -> None:
        self.samples = samples


@pytest.fixture
def gp_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(40, 2))
    sq_dist = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    cov = 1.5 * np.exp(-0.5 * sq_dist / 2.0**2) + 1e-8 * np.eye(40)
    y = rng.multivariate_normal(np.zeros(40), cov)
    return X, y + rng.normal(0, 0.1, size=40)


@pytest.fixture
def gp_posterior() -> _Posterior:
    rng = np.random.default_rng(1)
    return _Posterior(
        {
            "lengthscale": rng.uniform(1.5, 2.5, size=60),
            "variance": rng.uniform(1.0, 2.0, size=60),
            "noise": rng.uniform(0.005, 0.02, size=60),
        }
    )


class TestSpatialGpUncertainty:
    def test_predictive_interval_is_ordered_and_aligned(
        self, gp_data: tuple, gp_posterior: _Posterior
    ) -> None:
        X, y = gp_data
        model = SpatialGP(kernel="rbf", lengthscale=2.0, variance=1.5, noise=0.01)
        model.fit(X, y)
        mean, lower, upper = model.predictive_interval(
            gp_posterior, X=X[:6], level=0.95, samples=200, random_seed=0
        )
        assert mean.shape == (6,)
        assert lower.shape == (6,) and upper.shape == (6,)
        assert np.all(lower <= mean) and np.all(mean <= upper)
        assert np.all(lower < upper)

    def test_interval_widens_away_from_data(
        self, gp_data: tuple, gp_posterior: _Posterior
    ) -> None:
        X, y = gp_data
        model = SpatialGP(kernel="rbf", lengthscale=2.0, variance=1.5, noise=0.01)
        model.fit(X, y)
        near = model.predictive_interval(
            gp_posterior, X[:3], level=0.9, samples=200, random_seed=0
        )
        far = model.predictive_interval(
            gp_posterior,
            X=np.tile([[50.0, 50.0]], (50, 1)),
            level=0.9,
            samples=200,
            random_seed=0,
        )
        assert (far[2] - far[1]).mean() > (near[2] - near[1]).mean()

    def test_decomposition_sums_to_total(
        self, gp_data: tuple, gp_posterior: _Posterior
    ) -> None:
        X, y = gp_data
        model = SpatialGP(kernel="rbf", lengthscale=2.0, variance=1.5, noise=0.01)
        model.fit(X, y)
        dec = model.uncertainty_decomposition(gp_posterior, X[:4], samples=30)
        assert set(dec) == {"mean", "epistemic", "aleatoric", "total"}
        recombined = np.sqrt(dec["epistemic"] ** 2 + dec["aleatoric"] ** 2)
        np.testing.assert_allclose(dec["total"], recombined, atol=1e-6)
        assert np.all(dec["aleatoric"] > 0.0)

    def test_decomposition_epistemic_grows_away_from_data(
        self, gp_data: tuple, gp_posterior: _Posterior
    ) -> None:
        X, y = gp_data
        model = SpatialGP(kernel="rbf", lengthscale=2.0, variance=1.5, noise=0.01)
        model.fit(X, y)
        near = model.uncertainty_decomposition(gp_posterior, X[:3], samples=40)
        far = model.uncertainty_decomposition(
            gp_posterior, X=np.tile([[60.0, 60.0]], (8, 1)), samples=40
        )
        # Away from the data the predictive reverts to the prior, whose
        # conditional variance (aleatoric) dominates; the total must widen.
        assert far["aleatoric"].mean() > near["aleatoric"].mean()
        assert far["total"].mean() > near["total"].mean()

    def test_interval_is_calibrated_on_heldout(self, gp_posterior: _Posterior) -> None:
        rng = np.random.default_rng(5)
        X = rng.uniform(0, 10, size=(140, 2))
        sq_dist = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        cov = 1.5 * np.exp(-0.5 * sq_dist / 4.0) + 1e-8 * np.eye(140)
        latent = rng.multivariate_normal(np.zeros(140), cov)
        y = latent + rng.normal(0, 0.1, size=140)
        model = SpatialGP(kernel="rbf", lengthscale=2.0, variance=1.5, noise=0.01)
        model.fit(X[:100], y[:100])
        _, lower, upper = model.predictive_interval(
            gp_posterior, X[100:], level=0.95, samples=300, random_seed=3
        )
        cov = empirical_coverage(y[100:], lower, upper)
        assert 0.6 < cov <= 1.0

    def test_predictive_interval_rejects_bad_level(
        self, gp_data: tuple, gp_posterior: _Posterior
    ) -> None:
        X, y = gp_data
        model = SpatialGP(kernel="rbf")
        model.fit(X, y)
        with pytest.raises(ValueError, match="strictly between zero and one"):
            model.predictive_interval(gp_posterior, X[:2], level=1.0)


class TestHierarchicalUncertainty:
    @pytest.fixture
    def hierarchical(self) -> HierarchicalBayesianModel:
        return HierarchicalBayesianModel(n_levels=2)

    @pytest.fixture
    def hierarchical_posterior(self) -> _Posterior:
        rng = np.random.default_rng(2)
        return _Posterior(
            {
                "alpha_0": rng.normal(0, 1, size=80),
                "alpha_1": rng.normal(1, 1, size=80),
                "noise": rng.uniform(0.5, 1.5, size=80),
            }
        )

    def test_predictive_interval_respects_group_levels(
        self,
        hierarchical: HierarchicalBayesianModel,
        hierarchical_posterior: _Posterior,
    ) -> None:
        groups = [[0], [1], [0], [1]]
        mean, lower, upper = hierarchical.predictive_interval(
            hierarchical_posterior, X=groups, level=0.9, samples=60, random_seed=0
        )
        assert mean.shape == (4,)
        assert np.all(lower <= mean) and np.all(mean <= upper)
        # Group 0 has posterior mean ~0, group 1 ~1.
        assert mean[0] < mean[1]

    def test_decomposition_recovers_total(
        self,
        hierarchical: HierarchicalBayesianModel,
        hierarchical_posterior: _Posterior,
    ) -> None:
        groups = [[0], [1], [0], [1]]
        dec = hierarchical.uncertainty_decomposition(
            hierarchical_posterior, X=groups, samples=60
        )
        re = np.sqrt(dec["epistemic"] ** 2 + dec["aleatoric"] ** 2)
        np.testing.assert_allclose(dec["total"], re, atol=1e-6)

    def test_predictive_draws_reflect_group_uncertainty(
        self,
        hierarchical: HierarchicalBayesianModel,
        hierarchical_posterior: _Posterior,
    ) -> None:
        """The fixed posterior_predictive spreads with alpha · noise, not just noise."""
        groups = [[0], [1]]
        draws = hierarchical.posterior_predictive(
            hierarchical_posterior, X=groups, samples=200, random_seed=4
        )
        # alpha_0 has sd 1 and noise mean ~1, so the predictive sd exceeds noise alone.
        assert draws.std() > 0.7

    def test_posterior_predictive_is_replayable(
        self,
        hierarchical: HierarchicalBayesianModel,
        hierarchical_posterior: _Posterior,
    ) -> None:
        groups = [[0], [1], [0]]
        a = hierarchical.posterior_predictive(
            hierarchical_posterior, X=groups, samples=20, random_seed=7
        )
        b = hierarchical.posterior_predictive(
            hierarchical_posterior, X=groups, samples=20, random_seed=7
        )
        np.testing.assert_array_equal(a, b)


class TestPosteriorAnalysisUncertainty:
    def test_epistemic_uncertainty_is_the_posterior_sd(self) -> None:
        app_model = SpatialGP(kernel="rbf")
        samples = {"lengthscale": np.arange(0, 40, dtype=float)}
        posterior = PosteriorAnalysis(app_model, samples, None, "mcmc")
        assert posterior.epistemic_uncertainty("lengthscale") == pytest.approx(
            np.std(np.arange(0, 40, dtype=float))
        )

    def test_epistemic_uncertainty_missing_parameter(self) -> None:
        posterior = PosteriorAnalysis(
            SpatialGP(kernel="rbf"), {"a": np.zeros(4)}, None, "mcmc"
        )
        with pytest.raises(KeyError, match="does not contain"):
            posterior.epistemic_uncertainty("definitely_not_here")

    def test_coverage_on_heldout_gp(
        self, gp_data: tuple, gp_posterior: _Posterior
    ) -> None:
        X, y = gp_data
        model = SpatialGP(kernel="rbf", lengthscale=2.0, variance=1.5, noise=0.01)
        model.fit(X, y)
        # A posterior is normally built by inference; here the model holds the
        # training data and the fake posterior supplies draws.
        posterior = PosteriorAnalysis(model, gp_posterior.samples, None, "mcmc")
        cov = posterior.predictive_interval(
            X[:6], level=0.9, samples=100, random_seed=0
        )
        assert cov[0].shape == (6,)
        assert np.all(cov[1] <= cov[0]) and np.all(cov[0] <= cov[2])

    def test_coverage_function_is_a_fraction(self, gp_posterior: _Posterior) -> None:
        rng = np.random.default_rng(5)
        X = rng.uniform(0, 10, size=(60, 2))
        y = rng.normal(0, 1, size=60)
        model = SpatialGP(kernel="rbf", lengthscale=2.0, variance=1.5, noise=0.01)
        model.fit(X, y)
        posterior = PosteriorAnalysis(model, gp_posterior.samples, None, "mcmc")
        cov = posterior.coverage(X, y, level=0.95, samples=150, random_seed=1)
        assert 0.0 <= cov <= 1.0


class _FakeBayesianModel(BayesianModel):
    """Tiny Gaussian model shared by the estimator-telemetry checks."""

    def _setup_model(self, **kwargs: Any) -> None:
        self.parameters = {
            "mu": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 10.0}},
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        return float(-0.5 * np.sum((np.asarray(data) - theta["mu"]) ** 2))

    def log_prior(self, theta: dict[str, Any]) -> float:
        return float(-0.5 * (theta["mu"] / 10.0) ** 2)

    def predict(self, X_new, posterior=None, samples=100, return_std=False):
        values = np.zeros(len(X_new))
        return (values, np.ones(len(X_new))) if return_std else values

    def posterior_predictive(self, posterior, X=None, samples=100):
        return np.zeros((samples, 1))


class TestEstimatorTelemetry:
    def test_variational_records_elbo_and_posterior(self) -> None:
        model = _FakeBayesianModel(name="t")
        vi = VariationalInference(model, n_iterations=40, n_mc_samples=5, random_seed=0)
        out = vi.run(np.zeros(3), progress_bar=False, n_samples=30)
        assert out["mu"].shape == (30,)
        assert len(vi.elbo_history) == vi.n_total_iterations == 40
        assert np.isfinite(vi.best_elbo)
        summary = vi.estimate_posterior()
        assert set(summary["mu"]) == {"mean", "std"}
        assert summary["mu"]["std"] > 0.0

    def test_mcmc_records_acceptance(self) -> None:
        model = _FakeBayesianModel(name="t")
        sampler = MCMC(model, n_chains=2, random_seed=0, adapt_step_size=False)
        sampler.run(np.zeros(5), n_samples=20, n_warmup=5, progress_bar=False)
        assert sampler.acceptance_rates is not None
        assert sampler.acceptance_rates.shape == (2,)
        assert np.all(sampler.acceptance_rates >= 0.0)
        assert sampler.total_iterations == 25
        assert sampler.final_step_size is not None

    def test_hmc_records_acceptance(self) -> None:
        model = _FakeBayesianModel(name="t")
        sampler = HMC(model, n_chains=2, random_seed=0, adapt_step_size=False)
        sampler.run(np.zeros(5), n_samples=15, n_warmup=5, progress_bar=False)
        assert sampler.acceptance_rates is not None
        assert sampler.acceptance_rates.shape == (2,)
        assert len(sampler.final_step_sizes) == 2
        assert sampler.total_iterations == 20


class TestInferenceEndToEnd:
    def test_bayesian_inference_exposes_predictive_interval(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 10, size=(20, 2))
        y = rng.normal(0, 1, size=20)
        model = SpatialGP(kernel="rbf", lengthscale=2.0, variance=1.0, noise=0.1)
        inference = BayesianInference(
            model=model, method="mcmc", sampler_config={"n_chains": 2, "random_seed": 0}
        )
        posterior = inference.run(
            data={"X": X, "y": y}, n_samples=40, n_warmup=10, progress_bar=False
        )
        mean, lower, upper = posterior.predictive_interval(X[:3], level=0.9, samples=30)
        assert mean.shape == (3,)
        assert np.all(lower <= mean) and np.all(mean <= upper)
