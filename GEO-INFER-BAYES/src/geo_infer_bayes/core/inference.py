"""
Main inference engine for Bayesian analysis of geospatial data.
"""

import numpy as np
import xarray as xr
from typing import Any

from ..models.base import BayesianModel
from .posterior import PosteriorAnalysis


class BayesianInference:
    """
    Main class for performing Bayesian inference on geospatial data.

    This class serves as the primary interface for running different
    Bayesian inference methods on geospatial models.

    Parameters
    ----------
    model : BayesianModel
        The Bayesian model to use for inference
    method : str
        Inference method to use: 'mcmc', 'hmc', 'vi', 'smc', or 'abc'
    sampler_config : dict, optional
        Configuration parameters for the sampler
    """

    def __init__(
        self,
        model: "BayesianModel",
        method: str = "mcmc",
        sampler_config: dict[str, Any] | None = None,
    ):
        self.model = model
        self.method = method.lower()
        self.sampler_config = sampler_config or {}
        # _initialize_backend always assigns or raises, so the attribute is
        # never left as None; declaring it Any keeps the backend duck-typed.
        self.backend: Any = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the computational backend based on the method."""
        from .mcmc import MCMC
        from .hmc import HMC
        from .variational import VariationalInference
        from .smc import SequentialMonteCarlo
        from .abc import ApproximateBayesianComputation

        backends = {
            "mcmc": MCMC,
            "hmc": HMC,
            "vi": VariationalInference,
            "smc": SequentialMonteCarlo,
            "abc": ApproximateBayesianComputation,
        }

        if self.method not in backends:
            raise ValueError(
                f"Unsupported inference method: {self.method}. "
                f"Choose from: {', '.join(backends.keys())}"
            )

        self.backend = backends[self.method](self.model, **self.sampler_config)

    def run(
        self, data: np.ndarray | xr.Dataset | dict[str, Any], **kwargs: Any
    ) -> PosteriorAnalysis:
        """
        Run the inference algorithm on the provided data.

        Parameters
        ----------
        data : array-like or Dataset
            The geospatial data to use for inference
        **kwargs : dict
            Additional arguments to pass to the inference method

        Returns
        -------
        PosteriorAnalysis
            Object containing posterior samples and analysis tools
        """
        # Prepare data for the model
        prepared_data = self.model.prepare_data(data)

        # Run inference with the selected backend
        samples = self.backend.run(prepared_data, **kwargs)

        # A posterior is conditional on the data that produced it, so hand that
        # data back to the model. Without this, a conditional model such as a
        # Gaussian process holds sampled hyperparameters but no training set,
        # and posterior prediction fails.
        self.model.bind_training_data(prepared_data)

        # Create and return a PosteriorAnalysis object
        return PosteriorAnalysis(
            model=self.model,
            samples=samples,
            data=prepared_data,
            method=self.method,
            n_chains=self._backend_chain_count(),
        )

    def _backend_chain_count(self) -> int:
        """Report how many chains the active backend ran.

        Backends that are not chain-based -- variational inference, SMC, ABC --
        expose no ``n_chains``, and their draws are a single pooled sample, so
        1 is the correct answer for them.

        Returns
        -------
        int
            Number of chains, at least 1.
        """
        return max(1, int(getattr(self.backend, "n_chains", 1) or 1))

    def update(
        self,
        new_data: np.ndarray | xr.Dataset | dict[str, Any],
        previous_posterior: PosteriorAnalysis,
        **kwargs: Any,
    ) -> PosteriorAnalysis:
        """
        Update a previous posterior with new data (sequential inference).

        Parameters
        ----------
        new_data : array-like or Dataset
            New data to update the posterior
        previous_posterior : PosteriorAnalysis
            Previous posterior analysis result
        **kwargs : dict
            Additional arguments for the update process

        Returns
        -------
        PosteriorAnalysis
            Updated posterior analysis
        """
        if not hasattr(self.backend, "update"):
            raise RuntimeError(
                f"The {self.method} backend does not support sequential updates"
            )

        prepared_data = self.model.prepare_data(new_data)
        updated_samples = self.backend.update(
            prepared_data, previous_posterior.samples, **kwargs
        )
        self.model.bind_training_data(prepared_data)

        return PosteriorAnalysis(
            model=self.model,
            samples=updated_samples,
            data=prepared_data,
            method=self.method,
            n_chains=self._backend_chain_count(),
        )
