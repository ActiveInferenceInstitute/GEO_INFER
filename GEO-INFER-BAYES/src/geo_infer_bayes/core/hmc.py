"""
Hamiltonian Monte Carlo implementation for Bayesian inference.
"""

from collections import deque
import logging
import numpy as np
import xarray as xr
from typing import Any
from tqdm import tqdm
from ..utils.rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)


class HMC:
    """
    Hamiltonian Monte Carlo (HMC) for Bayesian inference.

    This class implements HMC sampling for Bayesian models with enhanced
    efficiency for high-dimensional parameter spaces.

    Parameters
    ----------
    model : BayesianModel
        The model to perform inference on
    n_chains : int, default=4
        Number of Markov chains to run
    step_size : float, default=0.01
        Initial step size for leapfrog integration
    n_steps : int, default=50
        Number of steps in leapfrog integration
    adapt_step_size : bool, default=True
        Whether to adapt the step size during warmup
    max_tree_depth : int, default=10
        Maximum depth for NUTS (No-U-Turn Sampler)
    target_accept : float, default=0.8
        Target acceptance rate
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
        step_size: float = 0.01,
        n_steps: int = 50,
        adapt_step_size: bool = True,
        max_tree_depth: int = 10,
        target_accept: float = 0.8,
        random_seed: SeedLike = None,
    ) -> None:
        if not isinstance(n_chains, (int, np.integer)) or n_chains < 1:
            raise ValueError("n_chains must be a positive integer")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and strictly positive")
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if not isinstance(max_tree_depth, (int, np.integer)) or max_tree_depth < 1:
            raise ValueError("max_tree_depth must be a positive integer")
        if not np.isfinite(target_accept) or not 0 < target_accept < 1:
            raise ValueError("target_accept must be finite and in (0, 1)")
        self.model = model
        self.n_chains = int(n_chains)
        self.step_size = float(step_size)
        self.n_steps = int(n_steps)
        self.adapt_step_size = adapt_step_size
        self.max_tree_depth = int(max_tree_depth)
        self.target_accept = float(target_accept)
        self.random_seed = random_seed
        self.rng: np.random.Generator = resolve_rng(random_seed)
        self._parameter_layout: list[tuple[str, int, int, tuple[int, ...]]] | None = (
            None
        )
        self._parameter_dimension: int = 0

        # Acceptance and dual-averaging telemetry, populated by :meth:`run`.
        self.acceptance_rates: np.ndarray | None = None
        self.final_step_sizes: list[float] | None = None
        self.total_iterations: int | None = None
        # Post-warmup (sampling-phase) acceptance per chain.
        self.sampling_acceptance_rates: np.ndarray | None = None

    def run(
        self,
        data: Any,
        n_samples: int = 1000,
        n_warmup: int = 500,
        thin: int = 1,
        init_strategy: str = "random",
        use_nuts: bool = True,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | xr.Dataset:
        """
        Run HMC sampling for the model.

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
        use_nuts : bool, default=True
            Whether to use No-U-Turn Sampler (NUTS)
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

        # Current log probabilities and parameters for each chain
        current_params = chains
        current_log_prob = np.zeros(self.n_chains)
        current_grad: list[np.ndarray] = [
            np.zeros(n_params) for _ in range(self.n_chains)
        ]

        for c in range(self.n_chains):
            theta = current_params[c]
            current_log_prob[c], current_grad[c] = self._compute_log_posterior_grad(
                theta, data
            )

        # Momentum distribution - standard normal
        def momentum_dist(size: int) -> np.ndarray:
            return self.rng.normal(0, 1, size=size)

        # Step-size adaptation state: dual-averaging-lite on the recent
        # acceptance window, driven toward ``target_accept``.
        adapt_window = 50
        accepted_window: list[deque] = [
            deque(maxlen=adapt_window) for _ in range(self.n_chains)
        ]
        log_step_sizes = np.full(self.n_chains, np.log(self.step_size))
        step_sizes = [self.step_size] * self.n_chains
        log_step_min = float(np.log(1e-6))
        log_step_max = float(np.log(10.0))

        # Sampling-phase acceptance tracking (post-warmup only).
        sampling_accepts = np.zeros(self.n_chains)
        sampling_iters = np.zeros(self.n_chains)

        # Run sampling
        total_iterations = n_warmup + n_samples * thin
        iterator = range(total_iterations)
        if progress_bar:
            iterator = tqdm(iterator, desc="HMC sampling")

        for i in iterator:
            # Adapt step size during warmup: dual-averaging-lite nudges
            # log(step) toward the target using the mean acceptance rate of
            # the last ``adapt_window`` iterations, with eta = 1/sqrt(k)
            # decaying so early large corrections taper off.
            if i < n_warmup and self.adapt_step_size and i > 0:
                eta = 1.0 / np.sqrt(i)
                for c in range(self.n_chains):
                    if accepted_window[c]:
                        window_rate = float(np.mean(accepted_window[c]))
                        log_step_sizes[c] += eta * (window_rate - self.target_accept)
                        log_step_sizes[c] = float(
                            np.clip(log_step_sizes[c], log_step_min, log_step_max)
                        )
                    step_sizes[c] = float(np.exp(log_step_sizes[c]))

            # Update each chain
            for c in range(self.n_chains):
                # Get current state
                theta = current_params[c]
                log_prob = current_log_prob[c]
                grad = current_grad[c]

                # Initialize momentum
                momentum = momentum_dist(n_params)
                # Leapfrog integration
                if use_nuts:
                    # No-U-Turn Sampler (NUTS)
                    new_theta, new_momentum, new_log_prob, new_grad, accepted = (
                        self._nuts_step(
                            theta, momentum, log_prob, grad, step_sizes[c], data
                        )
                    )
                else:
                    # Standard HMC with fixed trajectory
                    new_theta, new_momentum, new_log_prob, new_grad, accepted = (
                        self._hmc_step(
                            theta,
                            momentum,
                            log_prob,
                            grad,
                            step_sizes[c],
                            self.n_steps,
                            data,
                        )
                    )

                # Update state if accepted
                if accepted:
                    current_params[c] = new_theta
                    current_log_prob[c] = new_log_prob
                    current_grad[c] = new_grad
                    acceptance_rate[c] += 1
                accepted_window[c].append(1.0 if accepted else 0.0)
                if i >= n_warmup:
                    sampling_iters[c] += 1
                    if accepted:
                        sampling_accepts[c] += 1
            # Store samples after warmup, respecting thinning
            if i >= n_warmup and (i - n_warmup) % thin == 0:
                sample_idx = (i - n_warmup) // thin
                for c in range(self.n_chains):
                    samples[c, sample_idx, :] = self._flatten_theta(current_params[c])

        # Combine chains and convert to dictionary
        combined_samples = {}
        assert self._parameter_layout is not None
        for param, start, end, shape in self._parameter_layout:
            values = samples[:, :, start:end].reshape((-1,) + shape)
            combined_samples[param] = values.reshape(-1) if shape == () else values

        self.acceptance_rates = np.asarray(acceptance_rate, dtype=float)
        self.final_step_sizes = list(step_sizes)
        self.total_iterations = total_iterations
        if sampling_iters.any():
            self.sampling_acceptance_rates = sampling_accepts / sampling_iters
        else:
            self.sampling_acceptance_rates = np.zeros(self.n_chains)

        # Report diagnostics
        if progress_bar:
            for c in range(self.n_chains):
                logger.info(
                    "Chain %d acceptance rate: %.2f",
                    c + 1,
                    acceptance_rate[c] / total_iterations,
                )

        return combined_samples

    def _hmc_step(
        self,
        theta: dict[str, Any],
        momentum: np.ndarray,
        log_prob: float,
        grad: np.ndarray,
        step_size: float,
        n_steps: int,
        data: Any,
    ) -> tuple[dict[str, Any], np.ndarray, float, np.ndarray, bool]:
        """Perform a single HMC step with leapfrog integration."""
        # Make a copy of the initial state
        current_theta = theta.copy()
        current_momentum = momentum.copy()
        current_log_prob = log_prob
        current_grad = grad.copy()

        # Half step for momentum
        current_momentum += step_size * current_grad / 2

        # Full steps for position and momentum
        for _ in range(n_steps):
            # Full step for position
            self._update_position(current_theta, current_momentum, step_size)

            # Recompute gradient at new position
            current_log_prob, current_grad = self._compute_log_posterior_grad(
                current_theta, data
            )

            # Full step for momentum
            current_momentum += step_size * current_grad

        # Half step for momentum
        current_momentum += step_size * current_grad / 2

        # Negate momentum for reversibility
        current_momentum = -current_momentum

        # Compute Hamiltonian (energy)
        current_K = 0.5 * np.sum(current_momentum**2)
        initial_K = 0.5 * np.sum(momentum**2)

        initial_U = -log_prob
        current_U = -current_log_prob

        # Compute acceptance probability
        delta_H = current_U + current_K - (initial_U + initial_K)
        accept_prob = min(1.0, float(np.exp(min(0.0, -delta_H))))

        # Accept or reject
        if self.rng.random() < accept_prob:
            return current_theta, current_momentum, current_log_prob, current_grad, True
        else:
            return theta, momentum, log_prob, grad, False

    def _nuts_step(
        self,
        theta: dict[str, Any],
        momentum: np.ndarray,
        log_prob: float,
        grad: np.ndarray,
        step_size: float,
        data: Any,
    ) -> tuple[dict[str, Any], np.ndarray, float, np.ndarray, bool]:
        """Perform one slice-sampled No-U-Turn transition.

        This follows the recursive tree construction from Algorithm 3 of
        Hoffman and Gelman (2014). Scalar and array-valued model parameters
        are flattened into the position and momentum vectors and restored
        before model evaluations.
        """
        initial_joint = float(log_prob - 0.5 * np.sum(momentum**2))
        log_slice = initial_joint - self.rng.exponential()
        left_theta = theta.copy()
        right_theta = theta.copy()
        left_momentum = momentum.copy()
        right_momentum = momentum.copy()
        left_grad = grad.copy()
        right_grad = grad.copy()
        proposal = theta.copy()
        proposal_log_prob = log_prob
        proposal_grad = grad.copy()
        proposal_count = 1
        tree_valid = True

        for depth in range(self.max_tree_depth):
            direction = -1 if self.rng.random() < 0.5 else 1
            if direction == -1:
                (
                    left_theta,
                    left_momentum,
                    left_grad,
                    _,
                    _,
                    _,
                    _,
                    _,
                    candidate_theta,
                    candidate_log_prob,
                    candidate_grad,
                    candidate_count,
                    candidate_valid,
                ) = self._build_tree(
                    left_theta,
                    left_momentum,
                    left_grad,
                    log_slice,
                    direction,
                    depth,
                    step_size,
                    initial_joint,
                    data,
                )
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    right_theta,
                    right_momentum,
                    right_grad,
                    _,
                    candidate_theta,
                    candidate_log_prob,
                    candidate_grad,
                    candidate_count,
                    candidate_valid,
                ) = self._build_tree(
                    right_theta,
                    right_momentum,
                    right_grad,
                    log_slice,
                    direction,
                    depth,
                    step_size,
                    initial_joint,
                    data,
                )

            if candidate_valid and candidate_count > 0:
                total_count = proposal_count + candidate_count
                if self.rng.random() < candidate_count / total_count:
                    proposal = candidate_theta
                    proposal_log_prob = candidate_log_prob
                    proposal_grad = candidate_grad
                proposal_count = total_count

            tree_valid = candidate_valid and self._no_u_turn(
                left_theta, right_theta, left_momentum, right_momentum
            )
            if not tree_valid:
                break

        return (
            proposal,
            momentum,
            proposal_log_prob,
            proposal_grad,
            tree_valid,
        )

    def _build_tree(
        self,
        theta: dict[str, Any],
        momentum: np.ndarray,
        grad: np.ndarray,
        log_slice: float,
        direction: int,
        depth: int,
        step_size: float,
        initial_joint: float,
        data: Any,
    ) -> tuple[Any, ...]:
        """Build one NUTS subtree and return its valid proposal count."""
        if depth == 0:
            new_theta, new_momentum, new_log_prob, new_grad = self._leapfrog(
                theta, momentum, grad, direction * step_size, data
            )
            joint = float(new_log_prob - 0.5 * np.sum(new_momentum**2))
            valid = np.isfinite(joint) and joint > initial_joint - 1000.0
            count = int(valid and log_slice <= joint)
            return (
                new_theta,
                new_momentum,
                new_grad,
                new_log_prob,
                new_theta,
                new_momentum,
                new_grad,
                new_log_prob,
                new_theta,
                new_log_prob,
                new_grad,
                count,
                valid,
            )

        (
            left_theta,
            left_momentum,
            left_grad,
            left_log_prob,
            right_theta,
            right_momentum,
            right_grad,
            right_log_prob,
            candidate_theta,
            candidate_log_prob,
            candidate_grad,
            count,
            valid,
        ) = self._build_tree(
            theta,
            momentum,
            grad,
            log_slice,
            direction,
            depth - 1,
            step_size,
            initial_joint,
            data,
        )

        if valid:
            if direction == -1:
                (
                    left_theta,
                    left_momentum,
                    left_grad,
                    left_log_prob,
                    _,
                    _,
                    _,
                    _,
                    second_candidate,
                    second_log_prob,
                    second_grad,
                    second_count,
                    second_valid,
                ) = self._build_tree(
                    left_theta,
                    left_momentum,
                    left_grad,
                    log_slice,
                    direction,
                    depth - 1,
                    step_size,
                    initial_joint,
                    data,
                )
                right_theta, right_momentum, right_grad, right_log_prob = (
                    right_theta,
                    right_momentum,
                    right_grad,
                    right_log_prob,
                )
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    right_theta,
                    right_momentum,
                    right_grad,
                    right_log_prob,
                    second_candidate,
                    second_log_prob,
                    second_grad,
                    second_count,
                    second_valid,
                ) = self._build_tree(
                    right_theta,
                    right_momentum,
                    right_grad,
                    log_slice,
                    direction,
                    depth - 1,
                    step_size,
                    initial_joint,
                    data,
                )

            combined_count = count + second_count
            if (
                second_valid
                and second_count > 0
                and self.rng.random() < second_count / max(combined_count, 1)
            ):
                candidate_theta = second_candidate
                candidate_log_prob = second_log_prob
                candidate_grad = second_grad
            valid = second_valid and self._no_u_turn(
                left_theta, right_theta, left_momentum, right_momentum
            )
            count = combined_count

        return (
            left_theta,
            left_momentum,
            left_grad,
            left_log_prob,
            right_theta,
            right_momentum,
            right_grad,
            right_log_prob,
            candidate_theta,
            candidate_log_prob,
            candidate_grad,
            count,
            valid,
        )

    def _leapfrog(
        self,
        theta: dict[str, Any],
        momentum: np.ndarray,
        grad: np.ndarray,
        step_size: float,
        data: Any,
    ) -> tuple[dict[str, Any], np.ndarray, float, np.ndarray]:
        """Take one reversible leapfrog step."""
        new_momentum = momentum + 0.5 * step_size * grad
        new_theta = theta.copy()
        self._update_position(new_theta, new_momentum, step_size)
        new_log_prob, new_grad = self._compute_log_posterior_grad(new_theta, data)
        new_momentum = new_momentum + 0.5 * step_size * new_grad
        return new_theta, new_momentum, new_log_prob, new_grad

    def _no_u_turn(
        self,
        left_theta: dict[str, Any],
        right_theta: dict[str, Any],
        left_momentum: np.ndarray,
        right_momentum: np.ndarray,
    ) -> bool:
        displacement = self._flatten_theta(right_theta) - self._flatten_theta(
            left_theta
        )
        return bool(
            np.dot(displacement, left_momentum) >= 0
            and np.dot(displacement, right_momentum) >= 0
        )

    def _update_position(
        self, theta: dict[str, Any], momentum: np.ndarray, step_size: float
    ) -> None:
        """Update position (parameters) using momentum."""
        self._ensure_parameter_layout(theta)
        assert self._parameter_layout is not None
        for param, start, end, shape in self._parameter_layout:
            value = np.asarray(theta[param], dtype=float).reshape(-1)
            updated = value + step_size * momentum[start:end]
            theta[param] = updated.item() if shape == () else updated.reshape(shape)

    def _set_parameter_layout(self, theta: dict[str, Any]) -> None:
        """Record flatten/unflatten slices for scalar and array parameters."""
        layout: list[tuple[str, int, int, tuple[int, ...]]] = []
        offset = 0
        for parameter in self.model.parameters:
            value = np.asarray(theta[parameter], dtype=float)
            shape = value.shape
            size = int(value.size)
            layout.append((parameter, offset, offset + size, shape))
            offset += size
        self._parameter_layout = layout
        self._parameter_dimension = offset

    def _ensure_parameter_layout(self, theta: dict[str, Any]) -> None:
        if self._parameter_layout is None:
            self._set_parameter_layout(theta)

    def _flatten_theta(self, theta: dict[str, Any]) -> np.ndarray:
        """Flatten a parameter dictionary according to the sampler layout."""
        self._ensure_parameter_layout(theta)
        assert self._parameter_layout is not None
        return np.concatenate(
            [
                np.asarray(theta[param], dtype=float).reshape(-1)
                for param, _, _, _ in self._parameter_layout
            ]
        )

    def _compute_log_posterior_grad(
        self, theta: dict[str, Any], data: Any
    ) -> tuple[float, np.ndarray]:
        """
        Compute log posterior and its gradient.

        For simplicity, we'll use numerical differentiation.
        In practice, analytical gradients should be provided by the model
        for efficiency.
        """
        self._ensure_parameter_layout(theta)
        assert self._parameter_layout is not None
        log_posterior = float(self.model.log_posterior(theta, data))
        grad = np.zeros(self._parameter_dimension)

        h = 1e-6  # Step size for central finite differences
        for param, start, end, shape in self._parameter_layout:
            value = np.asarray(theta[param], dtype=float).reshape(-1)
            for index in range(value.size):
                plus_value = value.copy()
                minus_value = value.copy()
                plus_value[index] += h
                minus_value[index] -= h
                theta_plus = theta.copy()
                theta_minus = theta.copy()
                theta_plus[param] = (
                    plus_value.item() if shape == () else plus_value.reshape(shape)
                )
                theta_minus[param] = (
                    minus_value.item() if shape == () else minus_value.reshape(shape)
                )
                log_posterior_plus = float(self.model.log_posterior(theta_plus, data))
                log_posterior_minus = float(self.model.log_posterior(theta_minus, data))
                grad[start + index] = (log_posterior_plus - log_posterior_minus) / (
                    2 * h
                )

        return log_posterior, grad

    def _initialize_chains(
        self, data: Any, init_strategy: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
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

        for c in range(self.n_chains):
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
