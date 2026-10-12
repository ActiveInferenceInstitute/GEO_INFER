"""
Interface to Stan for Bayesian computation.
"""

import numpy as np
from typing import Any


class StanInterface:
    """
    Interface to Stan for Bayesian computation.

    This class provides a bridge between GEO-INFER-BAYES models
    and Stan's Bayesian computation capabilities.
    """

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        self.model_config = model_config or {}
        self.stan_model = None

    def create_spatial_gp_model(
        self, X: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> str:
        """
        Create a Stan model for spatial Gaussian Process.

        Parameters
        ----------
        X : array-like
            Spatial locations
        y : array-like
            Observations
        **kwargs : dict
            Additional parameters

        Returns
        -------
        str
            Stan model code
        """
        stan_code = """
        data {
            int<lower=1> N;
            int<lower=1> D;
            vector[N] y;
            matrix[N, D] X;
        }
        parameters {
            real<lower=0> lengthscale;
            real<lower=0> variance;
            real<lower=0> noise;
        }
        model {
            matrix[N, N] K;
            for (i in 1:N) {
                for (j in i:N) {
                    K[i,j] = variance * exp(-0.5 * sum(square(X[i] - X[j])) / (lengthscale * lengthscale));
                    K[j,i] = K[i,j];
                }
            }
            for (i in 1:N) {
                K[i,i] = K[i,i] + noise;
            }
            y ~ multi_normal(rep_vector(0, N), K);
        }
        """
        return stan_code

    def sample(
        self, n_samples: int = 1000, n_warmup: int = 500, **kwargs: Any
    ) -> dict[str, np.ndarray]:
        """
        Sample from the Stan model.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_warmup : int
            Number of warmup iterations
        **kwargs : dict
            Additional sampling parameters

        Returns
        -------
        dict
            Dictionary with parameter samples
        """
        raise RuntimeError(
            "Stan sampling requires a compiled Stan backend; use create_spatial_gp_model "
            "to obtain the model code and configure a Stan runtime before sampling"
        )
