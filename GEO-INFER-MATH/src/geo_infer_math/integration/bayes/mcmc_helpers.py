"""MCMC sampling helpers for Bayesian inference.

Implements Metropolis-Hastings with Gaussian proposals, acceptance
rate tracking, chain diagnostics, and thinning.
"""

import numpy as np
from typing import Any
from collections.abc import Callable
import logging

from geo_infer_math.utils.rng import resolve_rng


logger = logging.getLogger(__name__)


class MCMCHelpers:
    """MCMC sampling via Metropolis-Hastings.

    Provides configurable MCMC sampling with proposal tuning,
    burn-in, thinning, and convergence diagnostics.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        burn_in: int = 200,
        thin: int = 1,
        proposal_std: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize MCMC sampler.

        Args:
            n_samples: Number of post-burn-in samples.
            burn_in: Number of burn-in samples to discard.
            thin: Thinning interval (keep every n-th sample).
            proposal_std: Standard deviation of Gaussian proposal distribution.
            rng: Optional seed or np.random.Generator resolved via
                ``resolve_rng``; pass a seeded generator for reproducible chains.
        """
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.thin = thin
        self.proposal_std = proposal_std
        self._rng = resolve_rng(rng)
        logger.debug(
            "MCMCHelpers initialized (n_samples=%d, burn_in=%d, thin=%d)",
            n_samples,
            burn_in,
            thin,
        )

    def mcmc_sample(
        self,
        log_posterior: Callable,
        initial_state: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run Metropolis-Hastings sampling.

        Args:
            log_posterior: Function computing log-posterior density
                log p(θ|data). Must accept a 1-D numpy array.
            initial_state: Starting parameter vector, shape (d,).
            **kwargs: Additional parameters:
                - n_samples: Override instance default.
                - burn_in: Override instance default.
                - proposal_std: Override instance default.
                - rng: Override the seeded random generator.

        Returns:
            Dictionary with 'samples', 'acceptance_rate', 'log_posteriors',
            'diagnostics'.
        """
        n_samples = kwargs.get("n_samples", self.n_samples)
        burn_in = kwargs.get("burn_in", self.burn_in)
        thin = kwargs.get("thin", self.thin)
        proposal_std = kwargs.get("proposal_std", self.proposal_std)
        rng_override = kwargs.get("rng", self._rng)
        random_gen = resolve_rng(rng_override)

        total = burn_in + n_samples * thin
        state = np.asarray(initial_state, dtype=np.float64).copy()
        d = len(state)

        current_lp = float(log_posterior(state))

        chain: list[np.ndarray] = []
        log_posteriors: list[float] = []
        n_accepted = 0

        for i in range(total):
            # Gaussian proposal
            proposal = state + random_gen.standard_normal(d) * proposal_std
            proposal_lp = float(log_posterior(proposal))

            # Acceptance ratio (in log space)
            log_alpha = proposal_lp - current_lp

            if np.log(random_gen.random()) < log_alpha:
                state = proposal
                current_lp = proposal_lp
                n_accepted += 1

            # Record (post-burn-in, after thinning)
            if i >= burn_in and (i - burn_in) % thin == 0:
                chain.append(state.copy())
                log_posteriors.append(current_lp)

        samples = np.array(chain)
        acceptance_rate = n_accepted / total

        # Basic diagnostics
        diagnostics = self._compute_diagnostics(samples)

        logger.debug(
            "MCMC complete: %d samples, acceptance=%.2f%%",
            len(samples),
            acceptance_rate * 100,
        )

        return {
            "samples": samples,
            "acceptance_rate": float(acceptance_rate),
            "log_posteriors": log_posteriors,
            "diagnostics": diagnostics,
        }

    def _compute_diagnostics(self, samples: np.ndarray) -> dict[str, Any]:
        """Compute basic chain diagnostics."""
        n, d = samples.shape

        # Per-dimension statistics
        means = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)

        # Effective sample size (simple autocorrelation estimate)
        ess = np.zeros(d)
        for j in range(d):
            chain_j = samples[:, j]
            # Lag-1 autocorrelation
            if n > 1:
                auto = np.corrcoef(chain_j[:-1], chain_j[1:])[0, 1]
                ess[j] = n / max(1.0, 1.0 + 2.0 * abs(auto))
            else:
                ess[j] = 1.0

        return {
            "means": means.tolist(),
            "stds": stds.tolist(),
            "effective_sample_size": ess.tolist(),
            "n_samples": n,
            "n_params": d,
        }
