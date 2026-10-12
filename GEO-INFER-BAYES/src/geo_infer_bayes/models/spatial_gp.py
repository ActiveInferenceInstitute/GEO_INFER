"""
Gaussian Process model for spatial data.
"""

from contextlib import contextmanager
from typing import Any
from collections.abc import Callable, Iterator

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial.distance import cdist

from .base import BayesianModel
from ._model_utils import posterior_draw_indices
from ..utils.rng import SeedLike, resolve_rng


class SpatialGP(BayesianModel):
    """
    Gaussian Process model for spatial data.

    This class implements a Gaussian Process regression model
    for spatial interpolation and prediction.

    Parameters
    ----------
    kernel : str, default='matern'
        Covariance kernel: 'matern', 'rbf', 'exponential'
    lengthscale : float, default=1.0
        Length scale parameter for the kernel
    variance : float, default=1.0
        Variance parameter for the kernel
    noise : float, default=0.1
        Observation noise variance
    degree : float, default=1.5
        Degree parameter for the Matern kernel
    mean_function : callable, optional
        Mean function for the GP
    jitter : float, default=1e-6
        Small value added to the diagonal for numerical stability
    """

    def __init__(
        self,
        kernel: str = "matern",
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise: float = 0.1,
        degree: float = 1.5,
        mean_function: Callable[[np.ndarray], np.ndarray] | None = None,
        jitter: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.kernel_type = kernel.lower()
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.degree = degree
        self.mean_function = mean_function or (lambda x: np.zeros(len(x)))
        self.jitter = jitter
        # None until fit(); every method that needs them checks first.
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.L: np.ndarray | None = None  # Cholesky factor of the covariance

        super().__init__(name="SpatialGP", **kwargs)

    def _setup_model(self, **kwargs: Any) -> None:
        """Set up the Gaussian Process model."""
        # Define parameter distributions for inference
        self.parameters = {
            "lengthscale": {
                "prior": "log_normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "variance": {
                "prior": "log_normal",
                "hyperparams": {"mu": 0.0, "sigma": 1.0},
            },
            "noise": {"prior": "log_normal", "hyperparams": {"mu": -2.0, "sigma": 1.0}},
        }

        if self.kernel_type == "matern":
            self.parameters["degree"] = {
                "prior": "uniform",
                "hyperparams": {"low": 0.5, "high": 3.0},
            }

        # Initialize kernels based on type
        self.kernel_fn = self._get_kernel_function()

    def _get_kernel_function(self) -> Callable:
        """Get the appropriate kernel function based on the kernel type."""
        if self.kernel_type == "rbf":
            return self._rbf_kernel
        elif self.kernel_type == "matern":
            return self._matern_kernel
        elif self.kernel_type == "exponential":
            return self._exponential_kernel
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        dist = cdist(X1, X2)
        return np.asarray(
            self.variance * np.exp(-0.5 * (dist / self.lengthscale) ** 2), dtype=float
        )

    def _matern_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern kernel with adjustable degree."""
        dist = cdist(X1, X2)

        if self.degree == 0.5:
            # Exponential kernel
            kernel = self.variance * np.exp(-dist / self.lengthscale)
        elif self.degree == 1.5:
            # Matern 3/2
            scaled_dist = np.sqrt(3) * dist / self.lengthscale
            kernel = self.variance * (1 + scaled_dist) * np.exp(-scaled_dist)
        elif self.degree == 2.5:
            # Matern 5/2
            scaled_dist = np.sqrt(5) * dist / self.lengthscale
            kernel = (
                self.variance
                * (1 + scaled_dist + scaled_dist**2 / 3)
                * np.exp(-scaled_dist)
            )
        else:
            # Other degrees fall back to a stretched-exponential form, which is
            # not a Matern kernel; only the three cases above are exact.
            scaled_dist = dist / self.lengthscale
            kernel = self.variance * np.exp(-(scaled_dist**self.degree))
        return np.asarray(kernel, dtype=float)

    def _exponential_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Exponential kernel."""
        dist = cdist(X1, X2)
        return np.asarray(self.variance * np.exp(-dist / self.lengthscale), dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpatialGP":
        """
        Fit the GP to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns self
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        # Compute covariance matrix
        K = self.kernel_fn(self.X_train, self.X_train)
        K += np.eye(len(self.X_train)) * (self.noise + self.jitter)

        # Cache Cholesky factor for predictions
        self.L = cholesky(K, lower=True)

        return self

    def _posterior_draws(self, posterior: Any, samples: int) -> list[dict[str, float]]:
        """Return hyperparameter dicts for the draws a prediction averages over.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior holding the sampled hyperparameters.
        samples : int
            Maximum number of draws to use.

        Returns
        -------
        list of dict
            One parameter dict per selected draw, spread evenly across the
            chain.
        """
        names = ["lengthscale", "variance", "noise"]
        if self.kernel_type == "matern":
            names.append("degree")
        indices = posterior_draw_indices(posterior, samples, names)
        return [
            {name: float(np.asarray(posterior.samples[name])[i]) for name in names}
            for i in indices
        ]

    @contextmanager
    def _parameters_from(self, theta: dict[str, Any]) -> Iterator[None]:
        """Adopt hyperparameters from ``theta`` without touching cached state.

        Unlike :meth:`_temporary_parameters` this does not refactorize the
        training covariance, so it is the right tool for evaluating a likelihood
        on data other than the training set.

        Parameters
        ----------
        theta : dict
            Hyperparameter values to adopt; unknown keys are ignored.

        Yields
        ------
        None
        """
        tracked = ("lengthscale", "variance", "noise", "degree", "kernel_type")
        saved = {name: getattr(self, name) for name in tracked}
        saved_kernel = self.kernel_fn
        try:
            for name in tracked:
                if name in theta:
                    setattr(self, name, theta[name])
            self.kernel_fn = self._get_kernel_function()
            yield
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
            self.kernel_fn = saved_kernel

    @contextmanager
    def _temporary_parameters(self, theta: dict[str, float]) -> Iterator[None]:
        """Adopt one hyperparameter draw, then restore the fitted state.

        Refactorizing the training covariance is what makes the draw usable:
        the Cholesky factor cached by :meth:`fit` belongs to the fitted
        hyperparameters, not to ``theta``. The original factor and parameters
        are restored even if the body raises, so a failed draw cannot leave the
        model describing a covariance it no longer holds.

        Parameters
        ----------
        theta : dict
            Hyperparameter values to adopt; unknown keys are ignored.

        Yields
        ------
        None
        """
        X_train, _, _ = self._fitted_state()
        tracked = ("lengthscale", "variance", "noise", "degree")
        saved = {name: getattr(self, name) for name in tracked}
        saved_L, saved_kernel = self.L, self.kernel_fn
        try:
            for name in tracked:
                if name in theta:
                    setattr(self, name, theta[name])
            self.kernel_fn = self._get_kernel_function()
            K = self.kernel_fn(X_train, X_train)
            K += np.eye(len(X_train)) * (self.noise + self.jitter)
            self.L = cholesky(K, lower=True)
            yield
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
            self.L, self.kernel_fn = saved_L, saved_kernel

    def bind_training_data(self, data: Any) -> None:
        """Fit the GP to the data inference conditioned on.

        A GP predictive distribution is defined only relative to its training
        set, so sampling hyperparameters is not enough: the Cholesky factor of
        the training covariance has to exist before
        :meth:`predict` can be called with a posterior.

        Parameters
        ----------
        data : dict
            Prepared data holding ``X`` and ``y``. Anything else is ignored, so
            a model conditioned on a different data shape still runs.
        """
        if isinstance(data, dict) and "X" in data and "y" in data:
            self.fit(data["X"], data["y"])

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Make predictions at new locations.

        Parameters
        ----------
        X_new : array-like
            New locations to predict at
        posterior : PosteriorAnalysis, optional
            Posterior analysis object. If None, use current GP parameters.
        samples : int, default=100
            Number of posterior samples to use if posterior is provided
        return_std : bool, default=False
            Whether to return standard deviations

        Returns
        -------
        y_pred : ndarray
            Predicted mean
        y_std : ndarray, optional
            Predicted standard deviation
        """
        X_new = np.asarray(X_new)

        if posterior is not None:
            if self.X_train is None or self.y_train is None:
                raise ValueError(
                    "Model has not been fitted. Call fit() before posterior prediction."
                )

            means = []
            conditional_variances = []
            for theta in self._posterior_draws(posterior, samples):
                with self._temporary_parameters(theta):
                    if return_std:
                        mean, std = self._conditional_mean_std(X_new)
                        conditional_variances.append(std**2)
                    else:
                        mean = self._conditional_mean(X_new)
                means.append(mean)

            stacked = np.stack(means)
            mean_pred = np.asarray(np.mean(stacked, axis=0), dtype=float)

            if return_std:
                # Law of total variance. The spread of the per-draw means alone
                # captures only hyperparameter uncertainty and understates the
                # predictive interval badly -- for a well-identified posterior
                # the conditional GP variance is the dominant term.
                total_variance = np.mean(
                    np.stack(conditional_variances), axis=0
                ) + np.var(stacked, axis=0)
                return mean_pred, np.asarray(np.sqrt(total_variance), dtype=float)
            return mean_pred

        # No posterior: predict at the currently held hyperparameters.
        if return_std:
            return self._conditional_mean_std(X_new)
        return self._conditional_mean(X_new)

    def _fitted_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the training inputs, targets and Cholesky factor.

        Returns
        -------
        tuple of ndarray
            ``(X_train, y_train, L)``.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.X_train is None or self.y_train is None or self.L is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.X_train, self.y_train, self.L

    def _conditional_mean(self, X_new: np.ndarray) -> np.ndarray:
        """Posterior mean of the latent function at ``X_new``.

        Parameters
        ----------
        X_new : ndarray
            Locations to predict at.

        Returns
        -------
        ndarray
            Posterior mean, one entry per row of ``X_new``.
        """
        X_train, y_train, L = self._fitted_state()
        K_s = self.kernel_fn(X_train, X_new)
        alpha = solve_triangular(L, y_train - self.mean_function(X_train), lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)
        return np.asarray(self.mean_function(X_new) + K_s.T @ alpha, dtype=float)

    def _conditional_mean_std(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of the latent function.

        The standard deviation excludes observation noise, which belongs to a
        predictive draw rather than to the latent function; see
        :meth:`posterior_predictive`.

        Parameters
        ----------
        X_new : ndarray
            Locations to predict at.

        Returns
        -------
        tuple of ndarray
            ``(mean, std)``, each with one entry per row of ``X_new``.
        """
        X_train, _, L = self._fitted_state()
        mean = self._conditional_mean(X_new)
        K_s = self.kernel_fn(X_train, X_new)
        v = solve_triangular(L, K_s, lower=True)
        K_ss = self.kernel_fn(X_new, X_new)
        # Clipped at the jitter: the exact expression can go slightly negative
        # from rounding when a prediction point coincides with a training point.
        var = np.clip(np.diag(K_ss - v.T @ v), self.jitter, np.inf)
        return mean, np.asarray(np.sqrt(var), dtype=float)

    def posterior_predictive(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        samples: int = 100,
        random_seed: SeedLike = None,
    ) -> np.ndarray:
        """
        Generate posterior predictive samples.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object
        X : array-like, optional
            Locations to generate predictions for. If None, use observed locations.
        samples : int, default=100
            Number of posterior samples to use
        random_seed : int or numpy.random.Generator, optional
            Seed or generator for the observation-noise draws. ``None``
            (default) means a generator seeded from OS entropy, so results are
            not replayable; pass an int to replay, or a ``Generator`` to thread
            one stream through a pipeline. See
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.

        Returns
        -------
        ndarray of shape (samples, n_points)
            Posterior predictive samples
        """
        rng = resolve_rng(random_seed)

        if X is None:
            X = self.X_train

        X = np.asarray(X)
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Model has not been fitted. Call fit() before posterior prediction."
            )
        all_samples = []
        for theta in self._posterior_draws(posterior, samples):
            with self._temporary_parameters(theta):
                mean, std = self._conditional_mean_std(X)
            # std is the latent-function uncertainty; a predictive draw of an
            # observation also carries the observation-noise variance.
            all_samples.append(rng.normal(mean, np.sqrt(std**2 + theta["noise"])))

        if not all_samples:
            raise ValueError("posterior contains no usable parameter samples")
        return np.stack(all_samples)

    def predictive_interval(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        level: float = 0.95,
        samples: int = 200,
        random_seed: SeedLike = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a calibrated posterior predictive interval ``(mean, lower, upper)``.

        Unlike a credible band on the latent function, this interval is taken
        from the *predictive* draws (which carry observation noise), so its
        coverage is what a held-out observation is expected to satisfy. It is
        the object to feed :func:`geo_infer_bayes.core.evaluation` coverage
        diagnostics.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object.
        X : array-like, optional
            Locations to predict at. If None, use the training locations.
        level : float, default=0.95
            Nominal coverage of the interval, in ``(0, 1)``.
        samples : int, default=200
            Number of posterior predictive draws to base the interval on.
        random_seed : SeedLike, optional
            Seed or generator for the observation-noise draws. See
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.

        Returns
        -------
        tuple of ndarray
            ``(mean, lower, upper)``, each with one entry per prediction point.
        """
        interval_level = float(level)
        if not np.isfinite(interval_level) or not 0.0 < interval_level < 1.0:
            raise ValueError(
                "level must be a finite probability strictly between zero and one"
            )
        draws = self.posterior_predictive(
            posterior, X=X, samples=samples, random_seed=random_seed
        )
        tail = (1.0 - interval_level) / 2.0
        mean = np.asarray(np.mean(draws, axis=0), dtype=float)
        lower = np.asarray(np.percentile(draws, 100.0 * tail, axis=0), dtype=float)
        upper = np.asarray(
            np.percentile(draws, 100.0 * (1.0 - tail), axis=0), dtype=float
        )
        return mean, lower, upper

    def uncertainty_decomposition(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        samples: int = 50,
    ) -> dict[str, np.ndarray]:
        """
        Decompose predictive uncertainty into epistemic and aleatoric parts.

        For each hyperparameter draw the predictive mean and its (latent plus
        noise) variance are formed. The variance across the per-draw predictive
        means is *epistemic* -- it reflects uncertainty about the
        hyperparameters. The mean of the per-draw predictive variances is
        *aleatoric* -- the irreducible spread added by observation noise and
        the finite training sample. ``total = epistemic + aleatoric`` recovers
        the full predictive variance.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object.
        X : array-like, optional
            Locations to predict at. If None, use the training locations.
        samples : int, default=50
            Number of posterior draws to average over.

        Returns
        -------
        dict of str to ndarray
            ``epistemic``, ``aleatoric`` and ``total`` predictive standard
            deviations, plus the pooled ``mean`` prediction, one entry per
            prediction point.
        """
        if X is None:
            X = self.X_train
        if X is None:
            raise ValueError("X must be provided or the model must be fitted")
        X = np.asarray(X)
        self._fitted_state()  # raises if never fitted
        per_draw_means: list[np.ndarray] = []
        per_draw_vars: list[np.ndarray] = []
        for theta in self._posterior_draws(posterior, samples):
            with self._temporary_parameters(theta):
                mean, latent_std = self._conditional_mean_std(X)
            per_draw_means.append(np.asarray(mean, dtype=float))
            # Predictive variance at this draw: latent + observation noise.
            per_draw_vars.append(
                np.asarray(latent_std**2 + theta["noise"], dtype=float)
            )
        means_stack = np.stack(per_draw_means)
        var_stack = np.stack(per_draw_vars)
        epistemic = np.var(means_stack, axis=0)
        aleatoric = np.mean(var_stack, axis=0)
        total = epistemic + aleatoric
        return {
            "mean": np.asarray(np.mean(means_stack, axis=0), dtype=float),
            "epistemic": np.asarray(np.sqrt(epistemic), dtype=float),
            "aleatoric": np.asarray(np.sqrt(aleatoric), dtype=float),
            "total": np.asarray(np.sqrt(total), dtype=float),
        }

    def log_likelihood(
        self, theta: dict[str, Any], data: dict[str, np.ndarray]
    ) -> float:
        """
        Compute the marginal log-likelihood of the GP.

        Parameters
        ----------
        theta : dict
            Dictionary of parameter values
        data : dict
            Dictionary with 'X' and 'y' keys

        Returns
        -------
        float
            Log-likelihood value
        """
        X, y = data["X"], data["y"]

        # Set parameters from theta
        old_params = {}
        for param in ["lengthscale", "variance", "noise", "degree"]:
            if param in theta:
                old_params[param] = getattr(self, param)
                setattr(self, param, theta[param])

        # Update kernel function if needed
        if "kernel_type" in theta:
            old_params["kernel_type"] = self.kernel_type
            self.kernel_type = theta["kernel_type"]
            self.kernel_fn = self._get_kernel_function()

        # Compute kernel matrix
        K = self.kernel_fn(X, X)
        K += np.eye(len(X)) * (self.noise + self.jitter)

        # Compute log likelihood
        try:
            L = cholesky(K, lower=True)
            alpha = solve_triangular(L, y - self.mean_function(X), lower=True)
            alpha = solve_triangular(L.T, alpha, lower=False)

            # Marginalized log likelihood
            log_likelihood = -0.5 * np.dot(y - self.mean_function(X), alpha)
            log_likelihood -= np.sum(np.log(np.diag(L)))
            log_likelihood -= 0.5 * len(X) * np.log(2 * np.pi)
        except np.linalg.LinAlgError:
            log_likelihood = -np.inf

        # Restore parameters
        for param, value in old_params.items():
            setattr(self, param, value)
        if "kernel_type" in old_params:
            self.kernel_fn = self._get_kernel_function()

        return float(log_likelihood)

    def pointwise_log_likelihood(
        self, theta: dict[str, Any], data: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Decompose the GP marginal log-likelihood into per-observation terms.

        LOO and WAIC need a log-likelihood per observation, but a GP likelihood
        is joint, not factorized. The usable decomposition is the ordered
        conditional one: with ``K = L L^T`` and ``z = L^{-1} (y - m)``,

            log p(y_i | y_1..y_{i-1}) = -z_i^2 / 2 - log L_ii - log(2 pi) / 2

        and those terms sum exactly to the joint marginal log-likelihood. Naive
        alternatives are wrong in a way that is easy to miss: evaluating each
        point's own marginal density drops all correlation, and for a stationary
        kernel it drops the kernel entirely -- ``K(x, x)`` equals the signal
        variance whatever the lengthscale -- so every kernel scores identically.

        Parameters
        ----------
        theta : dict
            Hyperparameter values to evaluate at.
        data : dict
            Dictionary with ``X`` and ``y`` keys.

        Returns
        -------
        ndarray of shape (n_obs,)
            Per-observation log-likelihood contributions, summing to
            :meth:`log_likelihood`. All ``-inf`` if the covariance is not
            positive definite at ``theta``.

        Notes
        -----
        The decomposition depends on the observation order, though its sum does
        not. Diagnostics that treat the terms as exchangeable, Pareto-k among
        them, inherit that dependence.
        """
        X = np.asarray(data["X"], dtype=float)
        y = np.asarray(data["y"], dtype=float)

        with self._parameters_from(theta):
            K = self.kernel_fn(X, X)
            K += np.eye(len(X)) * (self.noise + self.jitter)
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return np.full(len(X), -np.inf)
            z = solve_triangular(L, y - self.mean_function(X), lower=True)

        return np.asarray(
            -0.5 * z**2 - np.log(np.diag(L)) - 0.5 * np.log(2 * np.pi), dtype=float
        )

    def log_prior(self, theta: dict[str, Any]) -> float:
        """
        Compute the log-prior for the GP parameters.

        Parameters
        ----------
        theta : dict
            Dictionary of parameter values

        Returns
        -------
        float
            Log-prior value
        """
        log_prior = 0.0

        # Log-normal prior for lengthscale
        if "lengthscale" in theta:
            mu = self.parameters["lengthscale"]["hyperparams"]["mu"]
            sigma = self.parameters["lengthscale"]["hyperparams"]["sigma"]
            log_prior += -0.5 * ((np.log(theta["lengthscale"]) - mu) / sigma) ** 2
            log_prior -= np.log(theta["lengthscale"] * sigma * np.sqrt(2 * np.pi))

        # Log-normal prior for variance
        if "variance" in theta:
            mu = self.parameters["variance"]["hyperparams"]["mu"]
            sigma = self.parameters["variance"]["hyperparams"]["sigma"]
            log_prior += -0.5 * ((np.log(theta["variance"]) - mu) / sigma) ** 2
            log_prior -= np.log(theta["variance"] * sigma * np.sqrt(2 * np.pi))

        # Log-normal prior for noise
        if "noise" in theta:
            mu = self.parameters["noise"]["hyperparams"]["mu"]
            sigma = self.parameters["noise"]["hyperparams"]["sigma"]
            log_prior += -0.5 * ((np.log(theta["noise"]) - mu) / sigma) ** 2
            log_prior -= np.log(theta["noise"] * sigma * np.sqrt(2 * np.pi))

        # Uniform prior for degree (if Matern)
        if "degree" in theta and self.kernel_type == "matern":
            low = self.parameters["degree"]["hyperparams"]["low"]
            high = self.parameters["degree"]["hyperparams"]["high"]
            if theta["degree"] < low or theta["degree"] > high:
                log_prior = -np.inf
            else:
                log_prior += -np.log(high - low)

        return log_prior


class SparseSpatialGP(SpatialGP):
    """Variational inducing-point Gaussian process for large spatial datasets.

    The model uses the collapsed variational bound for a Gaussian likelihood
    described by Titsias (2009).  Its fit cost is
    :math:`O(NM^2 + M^3)` for ``N`` observations and ``M`` inducing points,
    and it never constructs an ``N`` by ``N`` covariance matrix.  Sufficient
    statistics are accumulated in batches so the working memory is bounded by
    ``batch_size * M``.

    Parameters
    ----------
    inducing_points : array-like of shape (n_inducing, n_features), optional
        Fixed inducing-point locations.  When omitted, deterministic maximin
        locations are selected from the training inputs.
    n_inducing : int, default=100
        Number of locations to select when ``inducing_points`` is omitted.
    optimize_hyperparameters : bool, default=True
        Optimize lengthscale, signal variance, and observation-noise variance
        by maximizing the collapsed evidence lower bound during :meth:`fit`.
    max_iter : int, default=50
        Maximum L-BFGS-B iterations used for ELBO optimization.
    batch_size : int, default=2048
        Number of observations used for each sufficient-statistic update.
    **kwargs
        Kernel and mean-function arguments accepted by :class:`SpatialGP`.

    Notes
    -----
    ``noise`` is an observation-noise *variance*, matching :class:`SpatialGP`.
    The fitted variational distribution is exposed as
    ``variational_mean_`` and ``variational_covariance_`` in the original
    inducing-variable coordinates.
    """

    def __init__(
        self,
        inducing_points: np.ndarray | None = None,
        n_inducing: int = 100,
        optimize_hyperparameters: bool = True,
        max_iter: int = 50,
        batch_size: int = 2048,
        **kwargs: Any,
    ) -> None:
        # Accept the common spelling used by some GP libraries without leaking
        # it through to BayesianModel._setup_model().
        alias_count = kwargs.pop("num_inducing", None)
        if alias_count is None:
            alias_count = kwargs.pop("n_inducing_points", None)
        if alias_count is not None:
            if n_inducing != 100 and int(alias_count) != n_inducing:
                raise ValueError("Conflicting inducing-point counts were provided")
            n_inducing = int(alias_count)

        alias_locations = kwargs.pop("inducing_locations", None)
        if inducing_points is not None and alias_locations is not None:
            raise ValueError("Use only one of inducing_points and inducing_locations")
        if inducing_points is None:
            inducing_points = alias_locations

        if isinstance(n_inducing, (bool, np.bool_)) or not isinstance(
            n_inducing, (int, np.integer)
        ):
            raise TypeError("n_inducing must be an integer")
        if int(n_inducing) < 1:
            raise ValueError("n_inducing must be greater than zero")
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
            max_iter, (int, np.integer)
        ):
            raise TypeError("max_iter must be an integer")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be greater than zero")
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
            batch_size, (int, np.integer)
        ):
            raise TypeError("batch_size must be an integer")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be greater than zero")

        self.inducing_points = (
            None
            if inducing_points is None
            else np.asarray(inducing_points, dtype=float)
        )
        self.n_inducing = int(n_inducing)
        self.optimize_hyperparameters = bool(optimize_hyperparameters)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)

        self.inducing_points_: np.ndarray | None = None
        self.n_inducing_: int | None = None
        self.variational_mean_: np.ndarray | None = None
        self.variational_covariance_: np.ndarray | None = None
        self.elbo_: float | None = None
        self.initial_elbo_: float | None = None
        self.elbo_history_: list[float] = []
        self.optimization_result_: OptimizeResult | None = None
        self._inducing_cholesky: np.ndarray | None = None
        self._whitened_mean: np.ndarray | None = None
        self._whitened_covariance: np.ndarray | None = None

        super().__init__(**kwargs)
        self.name = "SparseSpatialGP"

    @staticmethod
    def _validate_training_data(
        X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return finite, aligned training arrays."""
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array[:, None]
        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be a non-empty one- or two-dimensional array")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must contain only finite values")

        y_array = np.asarray(y, dtype=float)
        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError("X and y must contain the same number of observations")
        if not np.all(np.isfinite(y_array)):
            raise ValueError("y must contain only finite values")
        return X_array, y_array

    def _mean_values(self, X: np.ndarray) -> np.ndarray:
        """Evaluate and validate the configured mean function."""
        values = np.asarray(self.mean_function(X), dtype=float)
        if values.ndim == 0:
            values = np.full(X.shape[0], float(values))
        values = values.reshape(-1)
        if values.shape != (X.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError("mean_function must return one finite value per input row")
        return values

    def _resolve_inducing_points(self, X: np.ndarray) -> np.ndarray:
        """Validate supplied locations or select deterministic maximin points."""
        if self.inducing_points is not None:
            locations = np.asarray(self.inducing_points, dtype=float)
            if locations.ndim == 1:
                if X.shape[1] == 1:
                    locations = locations[:, None]
                elif locations.size == X.shape[1]:
                    locations = locations[None, :]
            if locations.ndim != 2 or locations.shape[0] == 0:
                raise ValueError("inducing_points must contain at least one location")
            if locations.shape[1] != X.shape[1]:
                raise ValueError(
                    "inducing_points must have the same feature dimension as X"
                )
            if not np.all(np.isfinite(locations)):
                raise ValueError("inducing_points must contain only finite values")
            return np.array(locations, copy=True)

        count = min(self.n_inducing, X.shape[0])
        if count == X.shape[0]:
            return np.array(X, copy=True)

        # Start near the data centroid, then repeatedly choose the point farthest
        # from the selected set.  This provides deterministic spatial coverage
        # without introducing a clustering dependency or an N-by-N distance
        # matrix.
        centroid = np.mean(X, axis=0)
        first = int(np.argmin(np.sum((X - centroid) ** 2, axis=1)))
        selected = np.empty(count, dtype=int)
        selected[0] = first
        min_sq_distance = np.sum((X - X[first]) ** 2, axis=1)
        min_sq_distance[first] = -np.inf

        for index in range(1, count):
            next_point = int(np.argmax(min_sq_distance))
            selected[index] = next_point
            candidate_distance = np.sum((X - X[next_point]) ** 2, axis=1)
            min_sq_distance = np.minimum(min_sq_distance, candidate_distance)
            min_sq_distance[selected[: index + 1]] = -np.inf

        return np.array(X[selected], copy=True)

    def _validate_positive_hyperparameters(self) -> None:
        """Reject invalid kernel parameters before linear algebra begins."""
        for name in ("lengthscale", "variance", "noise", "jitter"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")

    def _collapsed_elbo(
        self,
        X: np.ndarray,
        y: np.ndarray,
        inducing_points: np.ndarray,
        *,
        return_posterior: bool = False,
    ) -> tuple[float, dict[str, np.ndarray] | None]:
        """Evaluate the Gaussian collapsed variational evidence lower bound."""
        self._validate_positive_hyperparameters()
        count = inducing_points.shape[0]
        K_mm = self.kernel_fn(inducing_points, inducing_points)
        K_mm = np.asarray(K_mm, dtype=float)
        K_mm += np.eye(count) * self.jitter
        L_mm = cholesky(K_mm, lower=True, check_finite=False)

        gram = np.zeros((count, count), dtype=float)
        cross = np.zeros(count, dtype=float)
        projected_trace = 0.0
        residual_sum_squares = 0.0

        for start in range(0, X.shape[0], self.batch_size):
            stop = min(start + self.batch_size, X.shape[0])
            X_batch = X[start:stop]
            residual = y[start:stop] - self._mean_values(X_batch)
            K_mn = self.kernel_fn(inducing_points, X_batch)
            projected = solve_triangular(L_mm, K_mn, lower=True, check_finite=False)
            gram += projected @ projected.T
            cross += projected @ residual
            projected_trace += float(np.sum(projected**2))
            residual_sum_squares += float(residual @ residual)

        noise = float(self.noise)
        B = np.eye(count) + gram / noise
        L_b = cholesky(B, lower=True, check_finite=False)
        scaled_cross = cross / noise
        solved_cross = solve_triangular(
            L_b, scaled_cross, lower=True, check_finite=False
        )

        # Every supported stationary kernel has k(x, x) == variance.  Avoiding
        # kernel_fn(X, X) here is the key large-N memory invariant.
        trace_residual = max(0.0, X.shape[0] * float(self.variance) - projected_trace)
        bound = -0.5 * X.shape[0] * np.log(2.0 * np.pi * noise)
        bound -= float(np.sum(np.log(np.diag(L_b))))
        bound -= 0.5 * residual_sum_squares / noise
        bound += 0.5 * float(solved_cross @ solved_cross)
        bound -= 0.5 * trace_residual / noise

        if not return_posterior:
            return float(bound), None

        whitened_mean = solve_triangular(
            L_b.T, solved_cross, lower=False, check_finite=False
        )
        inverse_factor = solve_triangular(
            L_b, np.eye(count), lower=True, check_finite=False
        )
        whitened_covariance = inverse_factor.T @ inverse_factor
        variational_mean = L_mm @ whitened_mean
        variational_covariance = L_mm @ whitened_covariance @ L_mm.T
        posterior = {
            "inducing_cholesky": L_mm,
            "whitened_mean": whitened_mean,
            "whitened_covariance": whitened_covariance,
            "variational_mean": variational_mean,
            "variational_covariance": variational_covariance,
        }
        return float(bound), posterior

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        optimize: bool | None = None,
    ) -> "SparseSpatialGP":
        """Fit the sparse GP and optimize its collapsed variational ELBO."""
        X_array, y_array = self._validate_training_data(X, y)
        locations = self._resolve_inducing_points(X_array)
        self._validate_positive_hyperparameters()

        self.X_train = X_array
        self.y_train = y_array
        self.inducing_points_ = locations
        self.n_inducing_ = locations.shape[0]
        # An exact N-by-N Cholesky factor is deliberately never populated.
        self.L = None

        initial_parameters = np.log(
            np.asarray([self.lengthscale, self.variance, self.noise], dtype=float)
        )
        initial_elbo, _ = self._collapsed_elbo(X_array, y_array, locations)
        self.initial_elbo_ = initial_elbo
        best_parameters = np.array(initial_parameters, copy=True)
        best_elbo = initial_elbo

        should_optimize = (
            self.optimize_hyperparameters if optimize is None else bool(optimize)
        )
        if should_optimize:

            def objective(log_parameters: np.ndarray) -> float:
                nonlocal best_elbo, best_parameters
                self.lengthscale, self.variance, self.noise = np.exp(log_parameters)
                try:
                    candidate, _ = self._collapsed_elbo(X_array, y_array, locations)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    return float(np.finfo(float).max / 100.0)
                if np.isfinite(candidate) and candidate > best_elbo:
                    best_elbo = candidate
                    best_parameters = np.array(log_parameters, copy=True)
                return (
                    float(-candidate)
                    if np.isfinite(candidate)
                    else float(np.finfo(float).max / 100.0)
                )

            # Log-space optimization preserves positivity.  The broad finite
            # bounds prevent overflow while allowing units from degrees to
            # projected metres and target scales from tiny to very large.
            self.optimization_result_ = minimize(
                objective,
                initial_parameters,
                method="L-BFGS-B",
                bounds=[(-18.0, 18.0)] * 3,
                options={"maxiter": self.max_iter},
            )
        else:
            self.optimization_result_ = None

        self.lengthscale, self.variance, self.noise = np.exp(best_parameters)
        final_elbo, posterior = self._collapsed_elbo(
            X_array, y_array, locations, return_posterior=True
        )
        if posterior is None:  # Defensive: return_posterior=True guarantees it.
            raise RuntimeError("Variational posterior construction failed")

        self.elbo_ = final_elbo
        self.elbo_history_ = [initial_elbo, final_elbo]
        self._inducing_cholesky = posterior["inducing_cholesky"]
        self._whitened_mean = posterior["whitened_mean"]
        self._whitened_covariance = posterior["whitened_covariance"]
        self.variational_mean_ = posterior["variational_mean"]
        self.variational_covariance_ = posterior["variational_covariance"]
        # Non-suffixed aliases make the variational parameters convenient in
        # exploratory workflows while retaining sklearn-style fitted names.
        self.variational_mean = self.variational_mean_
        self.variational_covariance = self.variational_covariance_
        return self

    def _sparse_fitted_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return inducing locations and the whitened variational posterior."""
        if (
            self.inducing_points_ is None
            or self._inducing_cholesky is None
            or self._whitened_mean is None
            or self._whitened_covariance is None
        ):
            raise ValueError("Model has not been fitted. Call fit() first.")
        return (
            self.inducing_points_,
            self._inducing_cholesky,
            self._whitened_mean,
            self._whitened_covariance,
        )

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict from the fitted inducing-point variational posterior."""
        del samples
        if posterior is not None:
            raise ValueError(
                "SparseSpatialGP uses its fitted variational posterior; "
                "posterior must be None"
            )
        X_array = np.asarray(X_new, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array[:, None]
        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X_new must contain at least one input row")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X_new must contain only finite values")

        locations, L_mm, whitened_mean, whitened_covariance = (
            self._sparse_fitted_state()
        )
        if X_array.shape[1] != locations.shape[1]:
            raise ValueError(
                "X_new has a different feature dimension from training data"
            )

        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        for start in range(0, X_array.shape[0], self.batch_size):
            stop = min(start + self.batch_size, X_array.shape[0])
            X_batch = X_array[start:stop]
            K_mx = self.kernel_fn(locations, X_batch)
            projected = solve_triangular(L_mm, K_mx, lower=True, check_finite=False)
            means.append(self._mean_values(X_batch) + projected.T @ whitened_mean)
            if return_std:
                prior_residual = float(self.variance) - np.sum(projected**2, axis=0)
                posterior_component = np.sum(
                    projected * (whitened_covariance @ projected), axis=0
                )
                variances.append(
                    np.clip(prior_residual + posterior_component, self.jitter, np.inf)
                )

        mean = np.asarray(np.concatenate(means), dtype=float)
        if not return_std:
            return mean
        variance = np.concatenate(variances)
        return mean, np.asarray(np.sqrt(variance), dtype=float)

    def evidence_lower_bound(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> float:
        """Return the collapsed ELBO at current hyperparameters."""
        if X is None and y is None:
            if self.elbo_ is None:
                raise ValueError("Model has not been fitted. Call fit() first.")
            return float(self.elbo_)
        if X is None or y is None:
            raise ValueError("X and y must be provided together")
        X_array, y_array = self._validate_training_data(X, y)
        locations = (
            self.inducing_points_
            if self.inducing_points_ is not None
            else self._resolve_inducing_points(X_array)
        )
        bound, _ = self._collapsed_elbo(X_array, y_array, locations)
        return bound

    def compute_elbo(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> float:
        """Alias for :meth:`evidence_lower_bound`."""
        return self.evidence_lower_bound(X, y)

    def log_likelihood(
        self, theta: dict[str, Any], data: dict[str, np.ndarray]
    ) -> float:
        """Use the sparse variational bound as the large-N likelihood proxy."""
        X_array, y_array = self._validate_training_data(data["X"], data["y"])
        locations = (
            self.inducing_points_
            if self.inducing_points_ is not None
            else self._resolve_inducing_points(X_array)
        )
        with self._parameters_from(theta):
            bound, _ = self._collapsed_elbo(X_array, y_array, locations)
        return bound

    def posterior_predictive(
        self,
        posterior: Any = None,
        X: np.ndarray | None = None,
        samples: int = 100,
        random_seed: SeedLike = None,
    ) -> np.ndarray:
        """Draw observations from the fitted sparse variational posterior."""
        if posterior is not None:
            raise ValueError(
                "SparseSpatialGP uses its fitted variational posterior; "
                "posterior must be None"
            )
        if not isinstance(samples, (int, np.integer)) or samples < 1:
            raise ValueError("samples must be a positive integer")
        if X is None:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() first.")
            X = self.X_train
        mean, std = self.predict(X, return_std=True)
        rng = resolve_rng(random_seed)
        observation_std = np.sqrt(std**2 + float(self.noise))
        return np.asarray(
            rng.normal(mean, observation_std, size=(int(samples), mean.size)),
            dtype=float,
        )
