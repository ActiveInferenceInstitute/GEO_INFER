"""
Markov Chain Monte Carlo implementation for Bayesian inference.
"""

import logging
import numpy as np
import xarray as xr
from typing import Any
from tqdm import tqdm
from ..utils.rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)


class MCMC:
    """
    Markov Chain Monte Carlo (MCMC) for Bayesian inference.

    This class implements MCMC sampling for Bayesian models, with
    specific enhancements for spatial models.

    Parameters
    ----------
    model : BayesianModel
        The model to perform inference on
    n_chains : int, default=4
        Number of Markov chains to run
    step_size : float, default=0.1
        Initial step size for proposals
    adapt_step_size : bool, default=True
        Whether to adapt the step size during warmup
    max_steps : int, default=1000
        Maximum number of steps in one parameter update
    random_seed : int or numpy.random.Generator, optional
        Seed or generator for every draw this sampler makes. ``None`` (default)
        means a generator seeded from OS entropy, so the chain is not
        replayable; pass an int to replay it, or a ``Generator`` to thread one
        stream through several samplers. See
        :func:`geo_infer_bayes.utils.rng.resolve_rng`.
    """

    def __init__(
        self,
        model: Any,
        n_chains: int = 4,
        step_size: float = 0.1,
        adapt_step_size: bool = True,
        max_steps: int = 1000,
        random_seed: SeedLike = None,
    ):
        if not isinstance(n_chains, (int, np.integer)) or n_chains < 1:
            raise ValueError("n_chains must be a positive integer")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and strictly positive")
        if not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.model = model
        self.n_chains = int(n_chains)
        self.step_size = float(step_size)
        self.adapt_step_size = adapt_step_size
        self.max_steps = int(max_steps)
        self.random_seed = random_seed
        self.rng: np.random.Generator = resolve_rng(random_seed)

        # Acceptance and run telemetry, populated by :meth:`run`. The arrays
        # let a caller audit the adaptively-tuned proposals after the fact.
        self.acceptance_rates: np.ndarray | None = None
        self.final_step_size: float | None = None
        self.total_iterations: int | None = None

    def run(
        self,
        data: Any,
        n_samples: int = 1000,
        n_warmup: int = 500,
        thin: int = 1,
        init_strategy: str = "random",
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | xr.Dataset:
        """
        Run MCMC sampling for the model.

        Parameters
        ----------
        data : any
            Data for inference
        n_samples : int, default=1000
            Number of samples to generate
        n_warmup : int, default=500
            Number of warmup/burn-in steps
        thin : int, default=1
            Thinning rate for samples
        init_strategy : str, default='random'
            Initialization strategy: 'random', 'prior', or 'map'
        progress_bar : bool, default=True
            Whether to show a progress bar
        **kwargs : dict
            Additional arguments for sampling

        Returns
        -------
        dict or Dataset
            Posterior samples
        """
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(n_warmup, (int, np.integer)) or n_warmup < 0:
            raise ValueError("n_warmup must be a non-negative integer")
        if not isinstance(thin, (int, np.integer)) or thin < 1:
            raise ValueError("thin must be a positive integer")
        # Initialize chains
        chains = self._initialize_chains(data, init_strategy, **kwargs)
        self._set_parameter_layout(chains[0])

        # Prepare storage for samples
        n_params = self._parameter_dimension

        # Allocate sample storage - shape: (n_chains, n_samples, n_params)
        samples = np.zeros((self.n_chains, n_samples, n_params))

        # Acceptance tracking
        acceptance_rate = np.zeros(self.n_chains)

        # Current log probabilities for each chain
        current_log_prob = np.zeros(self.n_chains)
        for c in range(self.n_chains):
            current_log_prob[c] = self._log_posterior(chains[c], data)

        # Run sampling
        total_iterations = n_warmup + n_samples * thin
        iterator = range(total_iterations)
        if progress_bar:
            iterator = tqdm(iterator, desc="MCMC sampling")

        for i in iterator:
            # Adapt step size during warmup
            if i < n_warmup and self.adapt_step_size and i % 50 == 0 and i > 0:
                for c in range(self.n_chains):
                    if acceptance_rate[c] / (i + 1) < 0.2:
                        self.step_size *= 0.8
                    elif acceptance_rate[c] / (i + 1) > 0.5:
                        self.step_size *= 1.2

            # Update each chain
            for c in range(self.n_chains):
                # Propose new state
                proposed_theta, log_proposal_ratio = self._propose(chains[c])

                # Compute acceptance probability
                proposed_log_prob = self._log_posterior(proposed_theta, data)
                log_accept_prob = (
                    proposed_log_prob - current_log_prob[c] + log_proposal_ratio
                )

                # Accept or reject
                if np.log(self.rng.random()) < log_accept_prob:
                    chains[c] = proposed_theta
                    current_log_prob[c] = proposed_log_prob
                    acceptance_rate[c] += 1

            # Store samples after warmup, respecting thinning
            if i >= n_warmup and (i - n_warmup) % thin == 0:
                sample_idx = (i - n_warmup) // thin
                for c in range(self.n_chains):
                    samples[c, sample_idx, :] = self._flatten_theta(chains[c])

        # Combine chains and convert to dictionary
        combined_samples = {}
        for param, start, end, shape in self._parameter_layout:
            values = samples[:, :, start:end].reshape((-1,) + shape)
            combined_samples[param] = values.reshape(-1) if shape == () else values

        # Persist run telemetry for post-hoc inspection.
        self.acceptance_rates = np.asarray(acceptance_rate, dtype=float)
        self.final_step_size = self.step_size
        self.total_iterations = total_iterations

        # Report diagnostics
        if progress_bar:
            for c in range(self.n_chains):
                logger.info(
                    "Chain %d acceptance rate: %.2f",
                    c + 1,
                    acceptance_rate[c] / total_iterations,
                )

        # Convert samples to desired format (add coordinates for xarray if needed)
        return combined_samples

    def update(
        self,
        new_data: Any,
        previous_samples: dict[str, np.ndarray] | xr.Dataset,
        n_samples: int = 500,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | xr.Dataset:
        """
        Update previous samples with new data.

        Parameters
        ----------
        new_data : any
            New data for updating
        previous_samples : dict or Dataset
            Previous posterior samples
        n_samples : int, default=500
            Number of new samples to generate
        **kwargs : dict
            Additional arguments for sampling

        Returns
        -------
        dict or Dataset
            Updated posterior samples
        """
        # Convert previous samples to initialization points for chains
        param_names = list(self.model.parameters.keys())

        # Get a random subset of previous samples to initialize chains
        if isinstance(previous_samples, dict):
            n_prev = len(previous_samples[param_names[0]])
            if n_prev < 1:
                raise ValueError("previous_samples must contain at least one sample")
            indices = self.rng.choice(
                n_prev, self.n_chains, replace=n_prev < self.n_chains
            )
            chains = []
            for idx in indices:
                chain = {}
                for param in param_names:
                    chain[param] = previous_samples[param][idx]
                chains.append(chain)
        else:
            # Handle xarray Dataset
            n_prev = len(previous_samples[param_names[0]])
            if n_prev < 1:
                raise ValueError("previous_samples must contain at least one sample")
            indices = self.rng.choice(
                n_prev, self.n_chains, replace=n_prev < self.n_chains
            )
            chains = []
            for idx in indices:
                chain = {}
                for param in param_names:
                    chain[param] = previous_samples[param].values[idx]
                chains.append(chain)

        # Run sampling with new data, using previous samples as initialization
        return self.run(
            data=new_data,
            n_samples=n_samples,
            n_warmup=n_samples // 2,  # Shorter warmup for updates
            init_strategy="custom",
            custom_init=chains,
            **kwargs,
        )

    def _initialize_chains(
        self, data: Any, init_strategy: str, **kwargs: Any
    ) -> list[dict[str, float]]:
        """Initialize the Markov chains."""
        param_names = list(self.model.parameters.keys())
        chains = []

        if init_strategy == "custom" and "custom_init" in kwargs:
            custom_init = kwargs["custom_init"]
            if len(custom_init) != self.n_chains:
                raise ValueError("custom_init must contain one state per chain")
            return [dict(chain) for chain in custom_init]

        if init_strategy not in {"random", "prior", "map"}:
            raise ValueError(
                "init_strategy must be 'random', 'prior', 'map', or 'custom'"
            )

        for _ in range(self.n_chains):
            chain = {}
            for param in param_names:
                param_info = self.model.parameters[param]

                if init_strategy == "random":
                    # Random initialization based on prior type
                    if param_info["prior"] == "log_normal":
                        mu = param_info["hyperparams"]["mu"]
                        sigma = param_info["hyperparams"]["sigma"]
                        chain[param] = np.exp(self.rng.normal(mu, sigma))
                    elif param_info["prior"] == "normal":
                        mu = param_info["hyperparams"]["mu"]
                        sigma = param_info["hyperparams"]["sigma"]
                        chain[param] = self.rng.normal(mu, sigma)
                    elif param_info["prior"] == "uniform":
                        low = param_info["hyperparams"]["low"]
                        high = param_info["hyperparams"]["high"]
                        chain[param] = self.rng.uniform(low, high)
                    else:
                        # Default to standard normal
                        chain[param] = self.rng.normal(0, 1)

                elif init_strategy == "prior":
                    # Sample directly from prior
                    if param_info["prior"] == "log_normal":
                        mu = param_info["hyperparams"]["mu"]
                        sigma = param_info["hyperparams"]["sigma"]
                        chain[param] = np.exp(self.rng.normal(mu, sigma))
                    elif param_info["prior"] == "normal":
                        mu = param_info["hyperparams"]["mu"]
                        sigma = param_info["hyperparams"]["sigma"]
                        chain[param] = self.rng.normal(mu, sigma)
                    elif param_info["prior"] == "uniform":
                        low = param_info["hyperparams"]["low"]
                        high = param_info["hyperparams"]["high"]
                        chain[param] = self.rng.uniform(low, high)

                elif init_strategy == "map":
                    # Use the analytical prior mode as a stable initial point.
                    if param_info["prior"] == "log_normal":
                        mu = param_info["hyperparams"]["mu"]
                        sigma = param_info["hyperparams"]["sigma"]
                        chain[param] = np.exp(mu)  # Mode of log-normal
                    elif param_info["prior"] == "normal":
                        mu = param_info["hyperparams"]["mu"]
                        chain[param] = mu  # Mode of normal
                    elif param_info["prior"] == "uniform":
                        low = param_info["hyperparams"]["low"]
                        high = param_info["hyperparams"]["high"]
                        chain[param] = (low + high) / 2  # Center of uniform

            chains.append(chain)

        return chains

    def _propose(
        self, current_theta: dict[str, float]
    ) -> tuple[dict[str, float], float]:
        """
        Generate a proposal for MCMC.

        Parameters
        ----------
        current_theta : dict
            Current parameter values

        Returns
        -------
        proposed_theta : dict
            Proposed parameter values
        log_proposal_ratio : float
            Log of the proposal ratio for asymmetric proposals
        """
        proposed_theta = {}
        log_proposal_ratio = 0.0

        for param, value in current_theta.items():
            param_info = self.model.parameters[param]

            # Adjust proposal based on prior
            if param_info["prior"] == "log_normal":
                # For log-normal, propose in log space
                value_array = np.asarray(value, dtype=float)
                if not np.all(np.isfinite(value_array)) or np.any(value_array <= 0):
                    raise ValueError(f"log-normal parameter {param} must be positive")
                proposed_log = np.log(value_array) + self.rng.normal(
                    0, self.step_size, size=value_array.shape
                )
                proposed_value = np.exp(proposed_log)
                proposed_theta[param] = (
                    float(proposed_value) if value_array.shape == () else proposed_value
                )

                # Proposal ratio is 1.0 (symmetric in log space)

            elif param_info["prior"] == "uniform":
                # For uniform, use truncated normal proposals
                low = param_info["hyperparams"]["low"]
                high = param_info["hyperparams"]["high"]

                value_array = np.asarray(value, dtype=float)
                proposed_value = value_array + self.rng.normal(
                    0, self.step_size, size=value_array.shape
                )
                width = high - low
                reflected = low + np.abs((proposed_value - low) % (2 * width))
                reflected = np.where(reflected > high, 2 * high - reflected, reflected)
                proposed_theta[param] = (
                    float(reflected) if value_array.shape == () else reflected
                )

                # Proposal ratio is 1.0 (symmetric)

            else:
                # Default to normal proposal
                value_array = np.asarray(value, dtype=float)
                proposed_value = value_array + self.rng.normal(
                    0, self.step_size, size=value_array.shape
                )
                proposed_theta[param] = (
                    float(proposed_value) if value_array.shape == () else proposed_value
                )
                # Proposal ratio is 1.0 (symmetric)

        return proposed_theta, log_proposal_ratio

    def _set_parameter_layout(self, theta: dict[str, Any]) -> None:
        """Record flattening slices for scalar and array parameters."""
        layout = []
        offset = 0
        for parameter in self.model.parameters:
            value = np.asarray(theta[parameter], dtype=float)
            shape = value.shape
            size = int(value.size)
            layout.append((parameter, offset, offset + size, shape))
            offset += size
        self._parameter_layout = layout
        self._parameter_dimension = offset

    def _flatten_theta(self, theta: dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(theta[param], dtype=float).reshape(-1)
                for param, _, _, _ in self._parameter_layout
            ]
        )

    def _log_posterior(self, theta: dict[str, float], data: Any) -> float:
        """Compute log posterior for a set of parameters."""
        try:
            return float(self.model.log_posterior(theta, data))
        except Exception:
            # Return negative infinity for invalid parameters
            return -np.inf
