"""Real installed Bayeux/NumPyro backend behavior with an analytical target."""

import json
import os
import sys
from textwrap import dedent

import numpy as np
import pytest

from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.utils.integration import ModernToolsIntegration
from geo_infer_test.process import run_process


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


def test_cold_numpyro_backend_through_both_public_surfaces(tmp_path):
    """A new process recompiles dependencies under strict warning handling."""
    code = dedent(
        """
        import json,sys,warnings
        import numpy as np
        import jax.numpy as jnp
        from geo_infer_act.core.generative_model import GenerativeModel
        from geo_infer_act.utils.integration import ModernToolsIntegration
        assert not any(name in sys.modules for name in ('bayeux','blackjax','jaxopt'))
        def density(x): return -.5*jnp.sum(((x-1.5)/.7)**2)
        results = []
        for surface in ('generative','integration_hub'):
            before = list(warnings.filters)
            point = {'x':jnp.array([1.5])}
            if surface == 'generative':
                result = GenerativeModel('categorical', {'state_dim':2,'random_seed':7}).integrate_bayeux(density,point,backend='bayeux',n_samples=100,warmup=100)
            else:
                result = ModernToolsIntegration({'random_seed':7}).create_bayeux_model(lambda parameters:density(parameters['x']),point,n_samples=100,warmup=100)
            samples = np.asarray(result['posterior_samples']['x']).reshape(-1)
            assert samples.size == 100 and np.isfinite(samples).all()
            np.testing.assert_allclose(samples.mean(),1.5,atol=.2,rtol=0)
            np.testing.assert_allclose(samples.var(),.49,atol=.2,rtol=0)
            np.testing.assert_array_equal(point['x'],[1.5])
            assert result['backend'] == 'bayeux'
            assert result['diagnostics'] == {'sampler':'numpyro_nuts','chains':1}
            assert result['log_marginal_likelihood'] is None
            assert warnings.filters == before
            results.append({'surface':surface,'mean':float(samples.mean()),'variance':float(samples.var()),'draws':samples.size})
        print(json.dumps(results))
        """
    )
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    # A unique empty prefix prevents an accepted old .pyc from hiding compile
    # warnings. No dependency source/cache is modified and no new cache is saved.
    environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "fresh-bytecode-prefix")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = run_process(
        [sys.executable, "-W", "error", "-c", code],
        env=environment,
        cwd=tmp_path,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert [row["surface"] for row in observed] == ["generative", "integration_hub"]
    assert all(row["draws"] == 100 for row in observed)
