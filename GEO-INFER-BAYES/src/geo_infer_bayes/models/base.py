"""
Base class for Bayesian models in the GEO-INFER-BAYES framework.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Any
from abc import ABC, abstractmethod

from ..utils.rng import SeedLike


class BayesianModel(ABC):
    """
    Abstract base class for all Bayesian models.

    This class defines the interface that all Bayesian models
    must implement to be compatible with the inference framework.

    Parameters
    ----------
    name : str
        Name of the model
    **kwargs : dict
        Additional model-specific parameters
    """

    def __init__(self, name: str, **kwargs: Any):
        self.name = name
        self.parameters: dict[str, Any] = {}
        self.priors: dict[str, Any] = {}
        self._setup_model(**kwargs)

    @abstractmethod
    def _setup_model(self, **kwargs: Any) -> None:
        """
        Set up the model structure and parameters.

        This method should be implemented by subclasses to define
        the specific model structure, parameters, and priors.

        Parameters
        ----------
        **kwargs : dict
            Model-specific parameters
        """
        raise RuntimeError("Bayesian model subclasses must define _setup_model()")

    @abstractmethod
    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        """
        Compute the log-likelihood for the model.

        Parameters
        ----------
        theta : dict
            Dictionary of parameter values
        data : any
            Data to compute the likelihood for

        Returns
        -------
        float
            Log-likelihood value
        """
        raise RuntimeError("Bayesian model subclasses must define log_likelihood()")

    @abstractmethod
    def log_prior(self, theta: dict[str, Any]) -> float:
        """
        Compute the log-prior for the model parameters.

        Parameters
        ----------
        theta : dict
            Dictionary of parameter values

        Returns
        -------
        float
            Log-prior value
        """
        raise RuntimeError("Bayesian model subclasses must define log_prior()")

    def log_posterior(self, theta: dict[str, Any], data: Any) -> float:
        """
        Compute the log-posterior for the model.

        Parameters
        ----------
        theta : dict
            Dictionary of parameter values
        data : any
            Data to compute the posterior for

        Returns
        -------
        float
            Log-posterior value
        """
        return self.log_likelihood(theta, data) + self.log_prior(theta)

    def prepare_data(self, data: np.ndarray | xr.Dataset | dict[str, Any]) -> Any:
        """
        Prepare data for inference.

        Parameters
        ----------
        data : array-like, Dataset, or dict
            Raw data to prepare

        Returns
        -------
        any
            Prepared data in the format expected by the model
        """
        # Default implementation - override for specific data preprocessing
        return data

    def bind_training_data(
        self, data: np.ndarray | xr.Dataset | dict[str, Any]
    ) -> None:
        """Retain the data the model was conditioned on.

        Inference produces a posterior over parameters *given* data, so a model
        must keep that data to make posterior predictions. Conditional models
        such as a Gaussian process also need it to form the predictive
        covariance. :meth:`geo_infer_bayes.core.BayesianInference.run` calls
        this after sampling, which is what lets a returned
        ``PosteriorAnalysis`` predict without a separate ``fit`` call.

        The default implementation is a no-op, which is correct for models
        whose posterior is self-contained. Subclasses holding training arrays
        should override it.

        Parameters
        ----------
        data : array-like, Dataset, or dict
            Prepared data, as returned by :meth:`prepare_data`.
        """
        return None

    @abstractmethod
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
            Posterior analysis object. If None, use prior predictive.
        samples : int, default=100
            Number of posterior samples to use
        return_std : bool, default=False
            Whether to return standard deviations

        Returns
        -------
        y_pred : ndarray
            Predictions
        y_std : ndarray, optional
            Standard deviations of predictions
        """
        raise RuntimeError("Bayesian model subclasses must define predict()")

    @abstractmethod
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
            Seed or generator for the predictive draws. See
            :func:`geo_infer_bayes.utils.rng.resolve_rng`.

        Returns
        -------
        ndarray
            Posterior predictive samples
        """
        raise RuntimeError(
            "Bayesian model subclasses must define posterior_predictive()"
        )

    def plot_prediction(
        self,
        posterior: Any,
        grid: np.ndarray | None = None,
        uncertainty: bool = True,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot model predictions from the posterior.

        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object
        grid : array-like, optional
            Grid points to predict on. If None, use a default grid
        uncertainty : bool, default=True
            Whether to plot uncertainty bounds
        **kwargs : dict
            Additional plotting options

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            Plot objects
        """
        # Default implementation for 2D spatial data
        # Subclasses should override for specific plotting needs
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create default grid if none provided
        if grid is None:
            x_range = np.linspace(0, 1, 50)
            y_range = np.linspace(0, 1, 50)
            grid_x, grid_y = np.meshgrid(x_range, y_range)
            grid = np.column_stack((grid_x.flatten(), grid_y.flatten()))

        # Make predictions
        if uncertainty:
            mean, std = self.predict(grid, posterior, return_std=True)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
        else:
            mean = self.predict(grid, posterior)

        # Reshape for 2D plotting if needed
        if grid.shape[1] == 2:
            n = int(np.sqrt(len(grid)))
            if n * n == len(grid):  # Perfect square
                grid_x = grid[:, 0].reshape(n, n)
                grid_y = grid[:, 1].reshape(n, n)
                mean_grid = mean.reshape(n, n)
                if uncertainty:
                    lower_grid = lower.reshape(n, n)
                    upper_grid = upper.reshape(n, n)

                # Plot the mean prediction as a contour plot
                contour = ax.contourf(grid_x, grid_y, mean_grid, cmap="viridis")
                fig.colorbar(contour, ax=ax, label="Mean prediction")

                # Plot uncertainty contours if requested
                if uncertainty:
                    ax.contour(
                        grid_x,
                        grid_y,
                        lower_grid,
                        colors="white",
                        alpha=0.5,
                        linestyles="dashed",
                        levels=5,
                    )
                    ax.contour(
                        grid_x,
                        grid_y,
                        upper_grid,
                        colors="white",
                        alpha=0.5,
                        linestyles="dashed",
                        levels=5,
                    )
            else:
                # Non-square grid, use scatter plot; show the symmetric
                # uncertainty band when available.
                scatter = ax.scatter(grid[:, 0], grid[:, 1], c=mean, cmap="viridis")
                fig.colorbar(scatter, ax=ax, label="Mean prediction")
                if uncertainty:
                    ax.errorbar(
                        grid[:, 0],
                        grid[:, 1],
                        yerr=1.96 * std,
                        fmt="none",
                        ecolor="white",
                        alpha=0.5,
                    )
        else:
            # 1D data or higher dimensions - just do a scatter plot
            scatter = ax.scatter(range(len(mean)), mean, c=mean, cmap="viridis")
            if uncertainty:
                ax.fill_between(range(len(mean)), lower, upper, alpha=0.3)
            fig.colorbar(scatter, ax=ax, label="Prediction")

        ax.set_title(f"{self.name} Spatial Prediction")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        return fig, ax
