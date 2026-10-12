#!/usr/bin/env python3
"""GEO-INFER-BAYES module orchestrator.
Runs one documented end-to-end BAYES operation on synthetic data: fit a
Bayesian linear regression (y = intercept + slope * x, Gaussian likelihood
and priors) to 60 seeded synthetic observations with the module's ``MCMC``
sampler, then summarize the posterior. All work goes through the real
``geo_infer_bayes`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import numpy as np
    from geo_infer_bayes import MCMCSampler
    from geo_infer_bayes.models.base import BayesianModel

    class LinearGaussianModel(BayesianModel):
        """Bayesian linear regression with Gaussian priors on both weights."""

        def _setup_model(self, **kwargs: Any) -> None:
            self.parameters = {
                "intercept": {
                    "prior": "normal",
                    "hyperparams": {"mu": 0.0, "sigma": 5.0},
                },
                "slope": {
                    "prior": "normal",
                    "hyperparams": {"mu": 0.0, "sigma": 5.0},
                },
            }
            self.noise_sigma = float(kwargs.get("noise_sigma", 1.0))

        def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
            x = np.asarray(data["x"], dtype=float)
            y = np.asarray(data["y"], dtype=float)
            intercept = float(np.asarray(theta["intercept"]))
            slope = float(np.asarray(theta["slope"]))
            residual = y - (intercept + slope * x)
            sigma = self.noise_sigma
            return float(
                -0.5 * np.sum((residual / sigma) ** 2)
                - len(y) * np.log(sigma * np.sqrt(2.0 * np.pi))
            )

        def log_prior(self, theta: dict[str, Any]) -> float:
            total = 0.0
            for name, spec in self.parameters.items():
                mu = float(spec["hyperparams"]["mu"])
                sigma = float(spec["hyperparams"]["sigma"])
                value = float(np.asarray(theta[name]))
                total += -0.5 * ((value - mu) / sigma) ** 2
                total -= np.log(sigma * np.sqrt(2.0 * np.pi))
            return float(total)

        def predict(
            self,
            posterior: Any,
            X: Any = None,
            samples: int = 100,
        ) -> Any:
            del posterior, samples  # point prediction from the posterior mean
            return None

        def posterior_predictive(
            self,
            posterior: Any,
            X: Any = None,
            samples: int = 100,
        ) -> Any:
            del posterior, X, samples
            return np.empty(0)

    rng = np.random.default_rng(7)
    n_obs = 60
    true_intercept, true_slope, true_sigma = 2.0, 1.5, 0.8
    x = rng.uniform(0.0, 10.0, n_obs)
    y = true_intercept + true_slope * x + rng.normal(0.0, true_sigma, n_obs)
    data = {"x": x, "y": y}

    model = LinearGaussianModel("linear-gaussian", noise_sigma=true_sigma)
    sampler = MCMCSampler(
        model,
        n_chains=2,
        step_size=0.05,
        adapt_step_size=True,
        random_seed=42,
    )
    posterior_samples = sampler.run(
        data, n_samples=400, n_warmup=200, progress_bar=False
    )

    samples = {
        name: np.asarray(values, dtype=float).reshape(-1)
        for name, values in posterior_samples.items()
    }
    return {
        "operation": "mcmc_linear_regression_fit",
        "n_observations": n_obs,
        "true_intercept": true_intercept,
        "true_slope": true_slope,
        "posterior": {
            name: {
                "posterior_mean": float(np.mean(values)),
                "posterior_std": float(np.std(values)),
            }
            for name, values in samples.items()
        },
        "n_posterior_samples": int(len(next(iter(samples.values())))),
        "acceptance_rates": [float(r) for r in sampler.acceptance_rates],
        "final_step_size": float(sampler.final_step_size),
        "total_iterations": int(sampler.total_iterations),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("BAYES", _operation))
