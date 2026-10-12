"""
Bayesian models for geospatial applications: GaussianProcess and SpatialCovariance.
"""

import numpy as np
from typing import Any


class SpatialCovariance:
    """Factory for covariance specifications consumed by ``GaussianProcess``."""

    @staticmethod
    def rbf(length_scale: float = 1.0, variance: float = 1.0) -> dict[str, float | str]:
        """Return a squared-exponential covariance specification."""
        return {
            "kernel_type": "rbf",
            "length_scale": length_scale,
            "variance": variance,
        }

    @staticmethod
    def matern_32(
        length_scale: float = 1.0, variance: float = 1.0
    ) -> dict[str, float | str]:
        """Return a Matérn 3/2 covariance specification."""
        return {
            "kernel_type": "matern32",
            "length_scale": length_scale,
            "variance": variance,
        }

    @staticmethod
    def matern_52(
        length_scale: float = 1.0, variance: float = 1.0
    ) -> dict[str, float | str]:
        """Return a Matérn 5/2 covariance specification."""
        return {
            "kernel_type": "matern52",
            "length_scale": length_scale,
            "variance": variance,
        }


class GaussianProcess:
    """
    High-level Gaussian Process interface for geospatial applications.


    Parameters
    ----------
    kernel_type : str
        Covariance kernel type: 'rbf', 'matern32', or 'exponential'.
    length_scale : float
        Characteristic length scale of the kernel.
    signal_variance : float
        Signal variance (output scale) of the kernel.
    noise_variance : float
        Observation noise variance.
    jitter : float
        Small diagonal addition for numerical stability.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-2,
        jitter: float = 1e-6,
        covariance_function: dict[str, float | str] | None = None,
        mean_function: str = "constant",
        **kwargs: object,
    ) -> None:
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments for GaussianProcess: {sorted(kwargs)}"
            )
        if covariance_function is not None:
            kernel_type = str(covariance_function.get("kernel_type", kernel_type))
            length_scale = float(covariance_function.get("length_scale", length_scale))
            signal_variance = float(
                covariance_function.get("variance", signal_variance)
            )
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.jitter = jitter
        if mean_function not in ("zero", "constant"):
            raise ValueError(
                f"Unsupported mean function: {mean_function!r} "
                "(supported: 'zero', 'constant')"
            )
        self.mean_function = mean_function
        self._mean: float | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Kernel functions
    # ------------------------------------------------------------------

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the kernel (covariance) matrix between two sets of points.

        Parameters
        ----------
        X1 : ndarray of shape (n1, d)
        X2 : ndarray of shape (n2, d)

        Returns
        -------
        K : ndarray of shape (n1, n2)
        """
        sq_dists = self._squared_distances(X1, X2)

        if self.kernel_type == "rbf":
            return np.asarray(
                self.signal_variance * np.exp(-0.5 * sq_dists / (self.length_scale**2))
            )
        elif self.kernel_type == "matern32":
            r = np.sqrt(np.maximum(sq_dists, 0.0)) / self.length_scale
            sqrt3_r = np.sqrt(3.0) * r
            return np.asarray(self.signal_variance * (1.0 + sqrt3_r) * np.exp(-sqrt3_r))
        elif self.kernel_type == "matern52":
            r = np.sqrt(np.maximum(sq_dists, 0.0)) / self.length_scale
            sqrt5_r = np.sqrt(5.0) * r
            return np.asarray(
                self.signal_variance
                * (1.0 + sqrt5_r + (5.0 / 3.0) * r**2)
                * np.exp(-sqrt5_r)
            )
        elif self.kernel_type == "exponential":
            r = np.sqrt(np.maximum(sq_dists, 0.0)) / self.length_scale
            return np.asarray(self.signal_variance * np.exp(-r))
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    @staticmethod
    def _squared_distances(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute pairwise squared Euclidean distances.

        Parameters
        ----------
        X1 : ndarray of shape (n1, d)
        X2 : ndarray of shape (n2, d)

        Returns
        -------
        D : ndarray of shape (n1, n2)
        """
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        return np.asarray(X1_sq + X2_sq.T - 2.0 * X1 @ X2.T)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "GaussianProcess":
        """Fit the Gaussian process model to training data.

        Stores training data, computes the kernel matrix K, and solves
        for alpha = K_inv @ y via Cholesky decomposition for numerical
        stability.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input locations.
        y : ndarray of shape (n_samples,)
            Training target values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_train = X
        self.y_train = y

        n = X.shape[0]
        K = self._compute_kernel(X, X)
        K += (self.noise_variance + self.jitter) * np.eye(n)

        # Cholesky decomposition: K = L @ L^T
        self._L = np.linalg.cholesky(K)

        # Center targets on the mean function before solving: a constant mean
        # is estimated from the training targets, "zero" leaves y untouched.
        if self.mean_function == "constant":
            self._mean = float(np.mean(y))
        else:
            self._mean = 0.0
        z = np.linalg.solve(self._L, y - self._mean)
        self._alpha = np.linalg.solve(self._L.T, z)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X_new: np.ndarray,
        return_std: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty quantification.

        Computes the GP predictive mean and (optionally) standard
        deviation at new input locations.

        Parameters
        ----------
        X_new : ndarray of shape (n_new, n_features)
            New input locations.
        return_std : bool
            If True, return both mean and standard deviation.

        Returns
        -------
        mean : ndarray of shape (n_new,)
            Predictive mean.
        std : ndarray of shape (n_new,), optional
            Predictive standard deviation (only when ``return_std=True``).
        """
        if self.X_train is None or self._alpha is None or self._L is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict()."
            )

        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        # Cross-covariance between training and new points
        K_star = self._compute_kernel(self.X_train, X_new)

        # Predictive mean: mu_* = K_*^T @ alpha + m(x_*); for the constant
        # mean function m is the training mean estimated during fit.
        mean: np.ndarray = np.asarray(K_star.T @ self._alpha)
        mean = mean + self._mean

        if return_std:
            # v = L^{-1} @ K_*
            v = np.linalg.solve(self._L, K_star)

            # Predictive variance: var_* = k(x_*, x_*) - v^T @ v
            K_ss_diag = self.signal_variance * np.ones(X_new.shape[0])
            var = K_ss_diag - np.sum(v**2, axis=0)
            var = np.maximum(var, self.jitter)
            std: np.ndarray = np.asarray(np.sqrt(var))
            return mean, std

        return mean

    def log_marginal_likelihood(self) -> float:
        """Compute the log marginal likelihood of the fitted model.

        Returns
        -------
        lml : float
            Log marginal likelihood: -0.5 * y^T alpha - sum(log(diag(L))) - n/2 * log(2*pi)
        """
        if self.y_train is None or self._alpha is None or self._L is None:
            raise RuntimeError("Model has not been fitted.")

        n = len(self.y_train)
        data_fit = -0.5 * self.y_train @ self._alpha
        complexity = -np.sum(np.log(np.diag(self._L)))
        normalisation = -0.5 * n * np.log(2.0 * np.pi)
        return float(data_fit + complexity + normalisation)


__all__ = ["GaussianProcess", "SpatialCovariance"]
