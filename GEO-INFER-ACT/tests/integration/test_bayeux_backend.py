"""Real installed Bayeux/NumPyro backend behavior with an analytical target."""

import numpy as np
import pytest

from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.utils.integration import ModernToolsIntegration


@pytest.mark.parametrize("surface", ["generative", "integration_hub"])
def test_real_numpyro_nuts_matches_scalar_gaussian_target(surface):
    import jax.numpy as jnp

    def density(x):
        return -0.5 * jnp.sum(((x - 1.5) / 0.7) ** 2)

    point = {"x": jnp.array([1.5])}
    if surface == "generative":
        result = GenerativeModel(
            "categorical", {"state_dim": 2, "random_seed": 7}
        ).integrate_bayeux(density, point, backend="bayeux", n_samples=100, warmup=100)
    else:
        result = ModernToolsIntegration({"random_seed": 7}).create_bayeux_model(
            lambda parameters: density(parameters["x"]),
            point,
            n_samples=100,
            warmup=100,
        )
    samples = np.asarray(result["posterior_samples"]["x"]).reshape(-1)
    assert samples.size == 100
    assert np.all(np.isfinite(samples))
    assert samples.mean() == pytest.approx(1.5, abs=0.35)
    assert samples.var() == pytest.approx(0.49, abs=0.30)
    assert result["backend"] == "bayeux"
    assert result["diagnostics"] == {"sampler": "numpyro_nuts", "chains": 1}
    assert result["log_marginal_likelihood"] is None
    assert "effective_sample_size" not in result["diagnostics"]
    np.testing.assert_array_equal(point["x"], [1.5])
