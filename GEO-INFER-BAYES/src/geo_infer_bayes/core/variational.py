"""
Variational Inference implementation for Bayesian inference.
"""

import logging
import numpy as np
import xarray as xr
import copy
from typing import Any
from tqdm import tqdm
from ..utils.rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)


class VariationalInference:
    """
    Variational Inference (VI) for scalable Bayesian approximation.

    This class implements variational inference methods to approximate
    posterior distributions by optimizing a simpler distribution.

    Parameters
    ----------
    model : BayesianModel
        The model to perform inference on
    learning_rate : float, default=0.01
        Learning rate for optimization
    n_iterations : int, default=10000
        Maximum number of optimization iterations
    convergence_tol : float, default=1e-6
        Convergence tolerance for ELBO
    n_mc_samples : int, default=10
        Number of Monte Carlo samples for gradient estimation
    vi_method : str, default='meanfield'
        Variational inference method: 'meanfield' or 'fullrank'
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
        learning_rate: float = 0.01,
        n_iterations: int = 10000,
        convergence_tol: float = 1e-6,
        n_mc_samples: int = 10,
        vi_method: str = "meanfield",
        random_seed: SeedLike = None,
    ):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and strictly positive")
        if not isinstance(n_iterations, (int, np.integer)) or n_iterations < 1:
            raise ValueError("n_iterations must be a positive integer")
        if not np.isfinite(convergence_tol) or convergence_tol < 0:
            raise ValueError("convergence_tol must be finite and non-negative")
        if not isinstance(n_mc_samples, (int, np.integer)) or n_mc_samples < 1:
            raise ValueError("n_mc_samples must be a positive integer")
        self.model = model
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.convergence_tol = float(convergence_tol)
        self.n_mc_samples = int(n_mc_samples)
        self.vi_method = vi_method.lower()
        self.random_seed = random_seed
        self.rng: np.random.Generator = resolve_rng(random_seed)

        # Convergence telemetry, populated by :meth:`run`. ``elbo_history``
        # records the ELBO in optimization order so a caller can inspect the
        # trace, and ``best_elbo`` / ``best_var_params`` pin the highest value
        # seen (the returned samples are drawn from the best state).
        self.elbo_history: list[float] = []
        self.best_elbo: float = -np.inf
        self.best_var_params: dict[str, dict[str, np.ndarray]] | None = None
        self.converged_at: int | None = None
        self.n_total_iterations: int = 0

        if self.vi_method not in ["meanfield", "fullrank"]:
            raise ValueError(
                f"Unsupported VI method: {self.vi_method}. "
                f"Choose from: 'meanfield', 'fullrank'"
            )

    def run(
        self,
        data: Any,
        progress_bar: bool = True,
        *,
        initial_var_params: dict[str, dict[str, np.ndarray]] | None = None,
        n_samples: int = 1000,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | xr.Dataset:
        """
        Run variational inference for the model.

        Parameters
        ----------
        data : any
            Data for inference
        progress_bar : bool, default=True
            Whether to show a progress bar
        **kwargs : dict
            Additional arguments for inference

        Returns
        -------
        dict or Dataset
            Approximate posterior samples
        """
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        param_names = list(self.model.parameters.keys())

        # Initialize variational parameters
        if initial_var_params is None:
            var_params = self._initialize_variational_parameters(param_names)
        else:
            if set(initial_var_params) != set(param_names):
                raise ValueError("initial_var_params must match model parameters")
            var_params = copy.deepcopy(initial_var_params)

        # Set up progress bar
        iterator = range(self.n_iterations)
        if progress_bar:
            iterator = tqdm(iterator, desc="Variational inference")

        # Track ELBO for convergence monitoring
        elbo_history = []
        best_elbo = -np.inf
        best_params = copy.deepcopy(var_params)

        # Optimization loop
        for i in iterator:
            # Compute ELBO and gradients
            elbo, grads = self._compute_elbo_and_gradients(var_params, data)

            # Update variational parameters using gradients
            self._update_variational_parameters(var_params, grads)

            # Track progress
            elbo_history.append(elbo)

            # Check if this is the best seen so far
            if elbo > best_elbo:
                best_elbo = elbo
                best_params = copy.deepcopy(var_params)

            # Check for convergence
            if (
                i > 100
                and abs(elbo_history[-1] - elbo_history[-100]) < self.convergence_tol
            ):
                if progress_bar:
                    logger.info("Converged after %d iterations", i)
                self.converged_at = i
                break

            # Update progress bar. tqdm wraps range() only when enabled, so
            # the attribute is checked rather than assumed.
            if progress_bar and i % 100 == 0 and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(ELBO=elbo)

        # Persist convergence telemetry for post-hoc inspection.
        self.elbo_history = list(elbo_history)
        self.best_elbo = best_elbo
        self.best_var_params = best_params
        self.n_total_iterations = len(elbo_history)

        # Generate samples from the approximate posterior (best state found).
        samples = self._generate_samples(best_params, n_samples=n_samples)

        return samples

    def _initialize_variational_parameters(
        self, param_names: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Initialize variational distribution parameters.

        For mean-field Gaussian approximation, we need a mean and
        log-standard deviation for each parameter.
        For full-rank, we would need additional covariance terms.
        """
        var_params: dict[str, dict[str, np.ndarray]] = {}

        for param in param_names:
            param_info = self.model.parameters[param]
            var_params[param] = {}

            # Initialize mean based on prior
            if param_info["prior"] == "log_normal":
                mu = param_info["hyperparams"]["mu"]
                sigma = param_info["hyperparams"]["sigma"]
                # Initialize in log space
                var_params[param]["mean"] = np.asarray(mu, dtype=float)
            elif param_info["prior"] == "normal":
                mu = param_info["hyperparams"]["mu"]
                var_params[param]["mean"] = np.asarray(mu, dtype=float)
            elif param_info["prior"] == "uniform":
                low = param_info["hyperparams"]["low"]
                high = param_info["hyperparams"]["high"]
                var_params[param]["mean"] = np.asarray((low + high) / 2, dtype=float)
            else:
                var_params[param]["mean"] = np.asarray(0.0, dtype=float)

            # Initialize log-std based on prior
            if param_info["prior"] == "log_normal" or param_info["prior"] == "normal":
                sigma = param_info["hyperparams"].get("sigma", 1.0)
                var_params[param]["log_std"] = np.log(np.asarray(sigma, dtype=float))
            else:
                var_params[param]["log_std"] = np.asarray(0.0, dtype=float)

            # Full-rank approximation: initialise a lower-triangular Cholesky
            # factor L such that Sigma = L @ L.T approximates the prior covariance.
            # Use a small near-identity initialisation for numerical stability.
            sigma_init = param_info["hyperparams"].get("sigma", 1.0)
            var_params[param]["cov_factor"] = np.eye(1) * float(
                np.asarray(sigma_init).reshape(-1)[0]
            )

        return var_params

    def _compute_elbo_and_gradients(
        self, var_params: dict[str, dict[str, np.ndarray]], data: Any
    ) -> tuple[float, dict[str, dict[str, np.ndarray]]]:
        """
        Compute the Evidence Lower Bound (ELBO) and its gradients.

        The ELBO is the objective function in variational inference,
        which we want to maximize. It's a lower bound on the model evidence.
        """
        param_names = list(var_params.keys())

        # Initialize gradients
        grads = {
            param: {
                "mean": np.zeros_like(var_params[param]["mean"], dtype=float),
                "log_std": np.zeros_like(var_params[param]["log_std"], dtype=float),
            }
            for param in param_names
        }
        if self.vi_method == "fullrank":
            for param in param_names:
                grads[param]["cov_factor"] = np.zeros_like(
                    var_params[param]["cov_factor"]
                )

        # Compute ELBO via Monte Carlo sampling
        elbo = 0.0
        for _ in range(self.n_mc_samples):
            # Sample from the variational distribution
            sample = self._sample_variational_distribution(var_params)

            # Compute log probability of the model
            log_prob_model = self.model.log_posterior(sample, data)

            # Compute log probability of the variational distribution
            log_prob_q = self._log_prob_variational(sample, var_params)

            # Accumulate ELBO
            elbo += float(log_prob_model - log_prob_q)

            # Compute gradients using score function estimator (REINFORCE)
            # or reparameterization trick (preferred for continuous parameters)
            for param in param_names:
                # Compute gradient for mean
                grads[param]["mean"] += (
                    log_prob_model - log_prob_q
                ) * self._compute_mean_gradient(sample, var_params, param)

                # Compute gradient for log-std
                grads[param]["log_std"] += (
                    log_prob_model - log_prob_q
                ) * self._compute_log_std_gradient(sample, var_params, param)

                if self.vi_method == "fullrank":
                    grads[param]["cov_factor"] += (
                        log_prob_model - log_prob_q
                    ) * self._compute_cov_factor_gradient(sample, var_params, param)

        # Average over Monte Carlo samples
        elbo /= self.n_mc_samples
        for param in param_names:
            grads[param]["mean"] /= self.n_mc_samples
            grads[param]["log_std"] /= self.n_mc_samples
            if self.vi_method == "fullrank":
                grads[param]["cov_factor"] /= self.n_mc_samples

        return elbo, grads

    def _sample_variational_distribution(
        self, var_params: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, Any]:
        """
        Sample from the variational distribution.

        For a mean-field Gaussian approximation, we sample each parameter
        independently from its variational distribution.
        """
        sample = {}

        for param, param_dist in var_params.items():
            mean = param_dist["mean"]
            std = self._effective_std(param_dist)

            # Sample from a standard normal and then transform
            z = self.rng.normal(0, 1, size=np.shape(mean))
            sample[param] = mean + z * std

            # Handle constraints for different parameter types
            param_info = self.model.parameters[param]

            # For log-normal parameters, work in log space
            if param_info["prior"] == "log_normal":
                sample[param] = np.exp(sample[param])

            # For uniform parameters, clip to the bounds
            elif param_info["prior"] == "uniform":
                low = param_info["hyperparams"]["low"]
                high = param_info["hyperparams"]["high"]
                sample[param] = np.clip(sample[param], low, high)

        return sample

    def _log_prob_variational(
        self, sample: dict[str, float], var_params: dict[str, dict[str, np.ndarray]]
    ) -> float:
        """
        Compute the log probability of a sample under the variational distribution.
        """
        log_prob = 0.0

        for param, value in sample.items():
            mean = var_params[param]["mean"]
            std = self._effective_std(var_params[param])

            # Handle log-normal parameters
            if self.model.parameters[param]["prior"] == "log_normal":
                # Convert to log space
                log_value = np.log(value)
                # Gaussian log-pdf
                log_prob += float(
                    np.sum(
                        -0.5 * ((log_value - mean) / std) ** 2
                        - np.log(std)
                        - 0.5 * np.log(2 * np.pi)
                    )
                )
                # Jacobian adjustment for log transform
                log_prob += float(-np.sum(np.log(value)))
            else:
                # Regular Gaussian log-pdf
                log_prob += float(
                    np.sum(
                        -0.5 * ((value - mean) / std) ** 2
                        - np.log(std)
                        - 0.5 * np.log(2 * np.pi)
                    )
                )

        return log_prob

    def _compute_mean_gradient(
        self,
        sample: dict[str, float],
        var_params: dict[str, dict[str, np.ndarray]],
        param: str,
    ) -> np.ndarray:
        """
        Compute the gradient of the log density with respect to the mean parameter.
        """
        std = self._effective_std(var_params[param])

        # For log-normal parameters, handle in log space
        if self.model.parameters[param]["prior"] == "log_normal":
            log_value = np.log(sample[param])
            gradient = (log_value - var_params[param]["mean"]) / (std**2)
        else:
            gradient = (sample[param] - var_params[param]["mean"]) / (std**2)
        return np.asarray(gradient, dtype=float)

    def _compute_log_std_gradient(
        self,
        sample: dict[str, float],
        var_params: dict[str, dict[str, np.ndarray]],
        param: str,
    ) -> np.ndarray:
        """
        Compute the gradient of the log density with respect to log standard deviation.
        """
        mean = var_params[param]["mean"]
        if self.vi_method == "fullrank":
            return np.zeros_like(var_params[param]["log_std"], dtype=float)
        std = self._effective_std(var_params[param])

        # For log-normal parameters, handle in log space
        if self.model.parameters[param]["prior"] == "log_normal":
            log_value = np.log(sample[param])
            gradient = (log_value - mean) ** 2 / std**2 - 1.0
        else:
            gradient = (sample[param] - mean) ** 2 / std**2 - 1.0
        return np.asarray(gradient, dtype=float)

    def _compute_cov_factor_gradient(
        self,
        sample: dict[str, float],
        var_params: dict[str, dict[str, np.ndarray]],
        param: str,
    ) -> np.ndarray:
        """
        Compute the score gradient for the Cholesky covariance factor.

        Current model parameters in this module are scalar, so the full-rank
        representation is a 1x1 Cholesky factor. The method is written in
        matrix form at the boundary so multidimensional parameters can extend
        the same contract later without changing callers.
        """
        cov_factor = np.asarray(var_params[param]["cov_factor"], dtype=float)
        std = self._effective_std(var_params[param])
        mean = var_params[param]["mean"]

        if self.model.parameters[param]["prior"] == "log_normal":
            transformed_value = np.log(sample[param])
        else:
            transformed_value = sample[param]

        residual = float(transformed_value - mean)
        scalar_grad = -1.0 / std + (residual**2) / (std**3)
        return np.array([[scalar_grad]], dtype=float).reshape(cov_factor.shape)

    def _update_variational_parameters(
        self,
        var_params: dict[str, dict[str, np.ndarray]],
        grads: dict[str, dict[str, np.ndarray]],
    ) -> None:
        """
        Update variational parameters using computed gradients.

        We use simple gradient ascent here, but more sophisticated optimizers
        like Adam or RMSprop could be implemented for better performance.
        """
        for param in var_params:
            # Update mean
            var_params[param]["mean"] += self.learning_rate * grads[param]["mean"]

            if self.vi_method == "fullrank" and "cov_factor" in var_params[param]:
                var_params[param]["cov_factor"] += (
                    self.learning_rate * grads[param]["cov_factor"]
                )
                cov_factor = np.asarray(var_params[param]["cov_factor"], dtype=float)
                cov_factor = np.nan_to_num(
                    cov_factor, nan=1e-3, posinf=1.0, neginf=-1.0
                )
                diag = np.diag(cov_factor).copy()
                min_diag = 1e-6
                for idx, value in enumerate(diag):
                    if abs(value) < min_diag:
                        cov_factor[idx, idx] = min_diag
                    else:
                        cov_factor[idx, idx] = np.sign(value) * min(
                            abs(value), np.exp(2.0)
                        )
                var_params[param]["cov_factor"] = np.tril(cov_factor)
                var_params[param]["log_std"] = np.log(
                    self._effective_std(var_params[param])
                )
                continue

            # Update log_std
            var_params[param]["log_std"] += self.learning_rate * grads[param]["log_std"]

            # Constrain log_std for numerical stability
            var_params[param]["log_std"] = np.clip(var_params[param]["log_std"], -10, 2)

    def _effective_std(self, param_dist: dict[str, np.ndarray]) -> np.ndarray:
        """Return the positive scalar standard deviation for the VI family."""
        if self.vi_method == "fullrank" and "cov_factor" in param_dist:
            cov_factor = np.asarray(param_dist["cov_factor"], dtype=float)
            if cov_factor.size:
                if cov_factor.size != 1:
                    raise ValueError(
                        "fullrank variational covariance currently supports scalar parameters only"
                    )
                return np.asarray(max(abs(cov_factor.reshape(-1)[0]), 1e-6))
        return np.asarray(
            np.maximum(np.exp(np.asarray(param_dist["log_std"], dtype=float)), 1e-6),
            dtype=float,
        )

    def _generate_samples(
        self, var_params: dict[str, dict[str, np.ndarray]], n_samples: int = 1000
    ) -> dict[str, np.ndarray]:
        """
        Generate samples from the approximate posterior for inference.
        """
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        samples = {
            param: np.zeros((n_samples,) + np.shape(dist["mean"]))
            for param, dist in var_params.items()
        }

        for i in range(n_samples):
            sample = self._sample_variational_distribution(var_params)
            for param, value in sample.items():
                samples[param][i] = value

        return samples

    def estimate_posterior(self) -> dict[str, dict[str, float]]:
        """
        Summarize the converged variational posterior per parameter.

        Returns the mean and marginal standard deviation of the best
        variational distribution found by the last :meth:`run` call. This is
        the parametric counterpart to a prediction interval: the ``std`` is the
        parameter-level posterior uncertainty expressed by the chosen family.

        Returns
        -------
        dict of str to dict
            One ``{"mean": float, "std": float}`` per model parameter.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not populated the best variational state.
        """
        if self.best_var_params is None:
            raise RuntimeError(
                "No variational posterior to summarize; call run() first."
            )
        summary: dict[str, dict[str, float]] = {}
        for param, dist in self.best_var_params.items():
            summary[param] = {
                "mean": float(np.mean(np.asarray(dist["mean"], dtype=float))),
                "std": float(np.mean(self._effective_std(dist))),
            }
        return summary

    def update(
        self,
        new_data: Any,
        previous_samples: dict[str, np.ndarray] | xr.Dataset,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | xr.Dataset:
        """
        Update the approximate posterior with new data.

        Parameters
        ----------
        new_data : any
            New data for updating
        previous_samples : dict or Dataset
            Previous posterior samples
        **kwargs : dict
            Additional arguments for inference

        Returns
        -------
        dict or Dataset
            Updated posterior samples
        """
        # Initialize variational parameters from previous posterior
        param_names = list(self.model.parameters.keys())
        var_params = self._initialize_variational_parameters(param_names)

        # Compute mean and std from previous samples
        for param in param_names:
            if isinstance(previous_samples, dict):
                samples = previous_samples[param]
            else:
                samples = previous_samples[param].values

            # Use samples to initialize variational parameters
            if self.model.parameters[param]["prior"] == "log_normal":
                # For log-normal, work in log space
                samples = np.asarray(samples, dtype=float)
                if not np.all(np.isfinite(samples)) or np.any(samples <= 0):
                    raise ValueError(
                        f"previous samples for log_normal parameter {param} must be positive and finite"
                    )
                log_samples = np.log(samples)
                var_params[param]["mean"] = np.mean(log_samples)
                var_params[param]["log_std"] = np.log(np.std(log_samples) + 1e-10)
            else:
                samples = np.asarray(samples, dtype=float)
                if samples.ndim == 0:
                    raise ValueError(
                        f"previous samples for parameter {param} must include a sample axis"
                    )
                if samples.shape[0] < 1 or not np.all(np.isfinite(samples)):
                    raise ValueError(
                        f"previous samples for parameter {param} must be non-empty and finite"
                    )
                var_params[param]["mean"] = np.mean(samples, axis=0)
                var_params[param]["log_std"] = np.log(np.std(samples, axis=0) + 1e-10)

        # Run inference with the new data and warm-started parameters.
        return self.run(
            data=new_data,
            initial_var_params=var_params,
            **kwargs,
        )
