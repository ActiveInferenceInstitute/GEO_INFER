"""
Unit tests for ModelComparison in core/model_comparison.py.

Tests cover AIC, BIC, Bayes factors, LOO, WAIC, DIC comparison methods,
model ranking, and the plot_comparison method.
"""

import numpy as np
import pytest


from geo_infer_bayes.core.model_comparison import ModelComparison


class _DummyModel:
    """Minimal model for testing comparison methods."""

    def __init__(self, name: str, bias: float = 0.0) -> None:
        self.name = name
        self.bias = bias
        self.parameters = {
            "mu": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 1.0}},
        }

    def log_likelihood(self, theta, data) -> float:
        observations = np.asarray(data)
        mu = theta.get("mu", 0.0) + self.bias
        return float(-0.5 * np.sum((observations - mu) ** 2))

    def log_prior(self, theta) -> float:
        mu = theta.get("mu", 0.0)
        return float(-0.5 * mu**2)


class TestModelComparisonInit:
    def test_default_construction(self) -> None:
        mc = ModelComparison()
        assert mc.models == []
        assert mc.comparison_results == {}

    def test_construction_with_models(self) -> None:
        models = [_DummyModel("A"), _DummyModel("B")]
        mc = ModelComparison(models=models)
        assert len(mc.models) == 2


class TestInformationCriteria:
    def test_aic_computation(self) -> None:
        mc = ModelComparison()
        aic = mc.compute_aic(log_likelihood=-50.0, n_params=3)
        expected = -2 * (-50.0) + 2 * 3
        np.testing.assert_allclose(aic, expected)

    def test_bic_computation(self) -> None:
        mc = ModelComparison()
        bic = mc.compute_bic(log_likelihood=-50.0, n_params=3, n_obs=100)
        expected = -2 * (-50.0) + 3 * np.log(100)
        np.testing.assert_allclose(bic, expected)

    def test_bic_penalises_more_than_aic_for_large_n(self) -> None:
        mc = ModelComparison()
        ll = -50.0
        k = 5
        n = 1000
        aic = mc.compute_aic(ll, k)
        bic = mc.compute_bic(ll, k, n)
        # BIC penalty = k * log(n) = 5 * log(1000) ~ 34.5
        # AIC penalty = 2 * k = 10
        assert bic > aic

    def test_bayes_factor_equal_evidence(self) -> None:
        mc = ModelComparison()
        bf = mc.compute_bayes_factor(log_evidence_1=-100.0, log_evidence_2=-100.0)
        np.testing.assert_allclose(bf, 1.0, atol=1e-10)

    def test_bayes_factor_favours_better_model(self) -> None:
        mc = ModelComparison()
        bf = mc.compute_bayes_factor(log_evidence_1=-80.0, log_evidence_2=-100.0)
        assert bf > 1.0


class TestCompareModels:
    @pytest.fixture
    def two_models(self):
        """Create two models with different biases -- model A should fit
        data centred at 0 better than model B which is biased."""
        return [_DummyModel("ModelA", bias=0.0), _DummyModel("ModelB", bias=5.0)]

    @pytest.fixture
    def centered_data(self):
        rng = np.random.default_rng(42)
        ll_matrix_A = rng.normal(-1.0, 0.1, size=(50, 20))
        ll_matrix_B = rng.normal(-3.0, 0.1, size=(50, 20))
        return {
            "observations": rng.standard_normal(20),
            "log_likelihood_matrix": ll_matrix_A,
        }, {
            "observations": rng.standard_normal(20),
            "log_likelihood_matrix": ll_matrix_B,
        }

    def test_compare_loo(self, two_models) -> None:
        mc = ModelComparison(models=two_models)
        rng = np.random.default_rng(0)
        data = {
            "observations": rng.standard_normal(15),
            "log_likelihood_matrix": rng.normal(-1, 0.2, size=(40, 15)),
        }
        results = mc.compare_models(data, method="loo")
        assert "ranking" in results
        assert "ModelA" in results
        assert "elpd_loo" in results["ModelA"]

    def test_compare_waic(self, two_models) -> None:
        mc = ModelComparison(models=two_models)
        rng = np.random.default_rng(1)
        data = {
            "observations": rng.standard_normal(10),
            "log_likelihood_matrix": rng.normal(-2, 0.3, size=(30, 10)),
        }
        results = mc.compare_models(data, method="waic")
        assert "ranking" in results
        assert "waic" in results["ModelA"]

    def test_compare_dic(self, two_models) -> None:
        mc = ModelComparison(models=two_models)
        rng = np.random.default_rng(2)
        data = {
            "observations": rng.standard_normal(10),
            "log_likelihood_matrix": rng.normal(-1.5, 0.2, size=(30, 10)),
        }
        results = mc.compare_models(data, method="dic")
        assert "ranking" in results
        assert "dic" in results["ModelA"]

    def test_invalid_method_raises(self, two_models) -> None:
        mc = ModelComparison(models=two_models)
        with pytest.raises(ValueError, match="Unknown comparison method"):
            mc.compare_models({}, method="unknown")

    def test_no_models_raises(self) -> None:
        mc = ModelComparison()
        with pytest.raises(ValueError, match="No models"):
            mc.compare_models({}, method="loo")

    def test_get_best_model_before_comparison_raises(self) -> None:
        mc = ModelComparison(models=[_DummyModel("A")])
        with pytest.raises(ValueError, match="No comparison results"):
            mc.get_best_model()

    def test_get_best_model_returns_model(self, two_models) -> None:
        mc = ModelComparison(models=two_models)
        rng = np.random.default_rng(5)
        data = {
            "observations": rng.standard_normal(10),
            "log_likelihood_matrix": rng.normal(-1, 0.1, size=(30, 10)),
        }
        mc.compare_models(data, method="loo")
        best = mc.get_best_model(criterion="loo")
        assert best is not None
        assert hasattr(best, "name")


class TestPlotComparison:
    def test_plot_comparison_returns_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")

        models = [_DummyModel("A"), _DummyModel("B")]
        mc = ModelComparison(models=models)
        rng = np.random.default_rng(10)
        data = {
            "observations": rng.standard_normal(10),
            "log_likelihood_matrix": rng.normal(-1, 0.2, size=(20, 10)),
        }
        mc.compare_models(data, method="loo")
        fig, ax = mc.plot_comparison()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_without_results_raises(self) -> None:
        mc = ModelComparison()
        with pytest.raises(ValueError):
            mc.plot_comparison()
