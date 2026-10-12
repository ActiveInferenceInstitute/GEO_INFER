"""
Sequential Monte Carlo implementation for Bayesian inference.

This module provides Sequential Monte Carlo (SMC) methods for
Bayesian inference, particularly useful for sequential data.
"""

import numpy as np
from typing import Any
from ..utils.rng import SeedLike, resolve_rng


class SequentialMonteCarlo:
    """
    Sequential Monte Carlo (SMC) for Bayesian inference.

    This class implements SMC sampling for Bayesian models,
    particularly useful for sequential data and online learning.
    """

    def __init__(
        self,
        model: Any,
        n_particles: int = 1000,
        resampling_threshold: float = 0.5,
        random_seed: SeedLike = None,
    ):
        """
        Initialize the SMC sampler.

        Args:
            model: The model to perform inference on
            n_particles: Number of particles for SMC
            resampling_threshold: ESS threshold for resampling
            random_seed: Seed or generator for every draw this sampler
                makes. ``None`` (default) means a generator seeded from OS
                entropy, so the run is not replayable; pass an int to replay
                it. See :func:`geo_infer_bayes.utils.rng.resolve_rng`.
        """
        if not isinstance(n_particles, (int, np.integer)) or n_particles < 1:
            raise ValueError("n_particles must be a positive integer")
        if not np.isfinite(resampling_threshold) or not 0 < resampling_threshold <= 1:
            raise ValueError("resampling_threshold must be in (0, 1]")
        self.model = model
        self.n_particles = int(n_particles)
        self.resampling_threshold = float(resampling_threshold)
        self.random_seed = random_seed
        self.rng: np.random.Generator = resolve_rng(random_seed)

    def run(
        self,
        data: Any,
        n_steps: int = 100,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | Any:
        """
        Run SMC sampling for the model.

        Args:
            data: Data for inference
            n_steps: Number of SMC steps
            progress_bar: Whether to show progress
            **kwargs: Additional arguments

        Returns:
            Posterior samples or SMC results
        """
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        # Initialize particles
        particles = kwargs.pop("initial_particles", None)
        if particles is None:
            particles = self._initialize_particles()
        if len(particles) != self.n_particles:
            raise ValueError("initial_particles must match n_particles")

        # Run SMC steps
        for step in range(n_steps):
            # Resample if effective sample size is too low
            if (
                self._effective_sample_size(particles)
                < self.resampling_threshold * self.n_particles
            ):
                particles = self._resample_particles(particles)

            # Move particles
            particles = self._move_particles(particles, data, step)

        # Extract samples from final particles
        samples = self._extract_samples(particles)

        return samples

    def _initialize_particles(self) -> list[dict[str, float]]:
        """Initialize SMC particles."""
        particles = []

        for _ in range(self.n_particles):
            particle = {}

            for param, param_info in self.model.parameters.items():
                if param_info["prior"] == "log_normal":
                    mu = param_info["hyperparams"]["mu"]
                    sigma = param_info["hyperparams"]["sigma"]
                    particle[param] = np.exp(self.rng.normal(mu, sigma))
                elif param_info["prior"] == "normal":
                    mu = param_info["hyperparams"]["mu"]
                    sigma = param_info["hyperparams"]["sigma"]
                    particle[param] = self.rng.normal(mu, sigma)
                elif param_info["prior"] == "uniform":
                    low = param_info["hyperparams"]["low"]
                    high = param_info["hyperparams"]["high"]
                    particle[param] = self.rng.uniform(low, high)
                else:
                    particle[param] = self.rng.normal(0, 1)

            particles.append(particle)

        return particles

    def _effective_sample_size(self, particles: list[dict[str, float]]) -> float:
        """Compute effective sample size."""
        # Importance-sampling ESS from normalized particle weights.
        weights = np.asarray([p.get("weight", 1.0) for p in particles], dtype=float)
        if (
            weights.size == 0
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or np.sum(weights) <= 0
        ):
            raise ValueError("particle weights must be finite and non-negative")
        weights = weights / np.sum(weights)

        return float(1.0 / np.sum(weights**2))

    def _resample_particles(
        self, particles: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """Resample particles based on their weights."""
        weights = np.asarray([p.get("weight", 1.0) for p in particles], dtype=float)
        if (
            weights.size == 0
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or np.sum(weights) <= 0
        ):
            raise ValueError("particle weights must be finite and non-negative")
        weights = weights / np.sum(weights)

        # Resample indices
        indices = self.rng.choice(
            len(particles), size=len(particles), p=weights, replace=True
        )

        # Create new particles
        new_particles = []
        for idx in indices:
            new_particle = particles[idx].copy()
            new_particle["weight"] = 1.0 / len(
                particles
            )  # Equal weights after resampling
            new_particles.append(new_particle)

        return new_particles

    def _move_particles(
        self, particles: list[dict[str, float]], data: Any, step: int
    ) -> list[dict[str, float]]:
        """Move particles using MCMC steps."""
        new_particles = []

        for particle in particles:
            # Propose new particle
            proposed = self._propose_particle(particle)

            # Compute acceptance probability
            log_accept_prob = self._compute_acceptance_probability(
                particle, proposed, data
            )

            # Accept or reject
            if np.log(self.rng.random()) < log_accept_prob:
                new_particles.append(proposed)
            else:
                new_particles.append(particle)

        return new_particles

    def _propose_particle(self, particle: dict[str, float]) -> dict[str, float]:
        """Propose a new particle."""
        proposed = particle.copy()

        for param, param_info in self.model.parameters.items():
            value = np.asarray(particle[param], dtype=float)
            if param_info["prior"] == "log_normal":
                proposal = value * np.exp(self.rng.normal(0, 0.1, size=value.shape))
            else:
                proposal = value + self.rng.normal(0, 0.1, size=value.shape)
                if param_info["prior"] == "uniform":
                    low = param_info["hyperparams"]["low"]
                    high = param_info["hyperparams"]["high"]
                    width = high - low
                    proposal = low + np.abs((proposal - low) % (2 * width))
                    proposal = np.where(proposal > high, 2 * high - proposal, proposal)
            proposed[param] = float(proposal) if value.shape == () else proposal

        return proposed

    def _compute_acceptance_probability(
        self, current: dict[str, float], proposed: dict[str, float], data: Any
    ) -> float:
        """Compute acceptance probability for particle move."""
        # Compute log posterior for current and proposed
        current_log_post = float(self.model.log_posterior(current, data))
        proposed_log_post = float(self.model.log_posterior(proposed, data))
        if not np.isfinite(current_log_post) or not np.isfinite(proposed_log_post):
            return -np.inf

        # Proposal ratio (symmetric)
        log_proposal_ratio = 0.0

        # Acceptance probability
        log_accept_prob = proposed_log_post - current_log_post + log_proposal_ratio

        return float(min(0, log_accept_prob))  # Metropolis acceptance

    def _extract_samples(
        self, particles: list[dict[str, float]]
    ) -> dict[str, np.ndarray]:
        """Extract samples from final particles."""
        samples = {}

        # Preserve the model's declared parameter order for reproducible output.
        for param in self.model.parameters:
            if any(param not in particle for particle in particles):
                raise ValueError(f"particle set is missing parameter {param!r}")
            samples[param] = np.array([p[param] for p in particles])

        return samples

    def update(
        self,
        new_data: Any,
        previous_samples: dict[str, np.ndarray] | Any,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | Any:
        """
        Update particles with new data.

        Args:
            new_data: New data for updating
            previous_samples: Previous SMC samples
            **kwargs: Additional arguments

        Returns:
            Updated samples
        """
        if not isinstance(previous_samples, dict) or not previous_samples:
            raise ValueError("previous_samples must be a non-empty mapping")
        parameter_samples = {
            name: np.asarray(previous_samples[name]) for name in self.model.parameters
        }
        if any(values.ndim == 0 for values in parameter_samples.values()):
            raise ValueError("previous_samples values must contain draws")
        lengths = {len(values) for values in parameter_samples.values()}
        if len(lengths) != 1:
            raise ValueError("previous_samples parameters must have equal lengths")
        n_previous = next(iter(lengths))
        if n_previous < 1:
            raise ValueError("previous_samples must contain at least one draw")

        # Initialize particles from previous samples, resampling if necessary.
        indices = self.rng.integers(n_previous, size=self.n_particles)
        particles = []
        for index in indices:
            particle = {}
            for param, values in parameter_samples.items():
                particle[param] = np.asarray(values)[index]
            particle["weight"] = 1.0 / self.n_particles
            particles.append(particle)

        # Run SMC on new data
        return self.run(new_data, initial_particles=particles, **kwargs)
