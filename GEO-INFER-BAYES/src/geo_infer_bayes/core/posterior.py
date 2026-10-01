"""
Tools for analyzing posterior distributions from Bayesian inference.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
from typing import Any

from ..models.base import BayesianModel
from ..utils.rng import SeedLike
from .evaluation import empirical_coverage


class PosteriorAnalysis:
    """
    Analyze and visualize posterior distributions from Bayesian inference.

    This class provides tools for summarizing, analyzing, and visualizing
    posterior distributions obtained from Bayesian inference.

    Parameters
    ----------
    model : BayesianModel
        The model used for inference
    samples : dict or xarray.Dataset
        Posterior samples from inference
    data : dict or xarray.Dataset
        Data used for inference
    method : str
        Inference method used
    n_chains : int, default=1
        Number of chains the samplers ran. Samplers hand back draws already
        concatenated across chains, so this is what lets the split-chain
        diagnostics -- R-hat above all -- be computed at all. Leaving it at 1
        makes ``r_hat`` undefined, which ArviZ reports as ``NaN``.
    """

    def __init__(
        self,
        model: "BayesianModel",
        samples: dict[str, np.ndarray] | xr.Dataset | Any,
        data: dict[str, np.ndarray] | xr.Dataset | None,
        method: str,
        n_chains: int = 1,
    ):
        if not isinstance(n_chains, (int, np.integer)) or n_chains < 1:
            raise ValueError("n_chains must be a positive integer")
        self.model = model
        self.samples = samples
        self.data = data
        self.method = method
        self.n_chains = int(n_chains)

        # Convert samples to InferenceData if not already
        if not isinstance(samples, az.InferenceData):
            self.arviz_data = self._convert_to_arviz(samples)
        else:
            self.arviz_data = samples

    def chain_samples(self) -> dict[str, np.ndarray]:
        """Return the draws reshaped to ``(chain, draw, ...)``.

        Samplers concatenate chains in C order, so reshaping the leading axis
        recovers the original chain assignment exactly. Use this for any
        between-chain diagnostic; ``self.samples`` stays flat because
        prediction and model comparison want one pooled draw axis.

        Returns
        -------
        dict of str to ndarray
            One array per parameter, leading axis of length ``n_chains``.

        Raises
        ------
        TypeError
            If the samples are not a dict of arrays.
        ValueError
            If a parameter's draw count is not divisible by ``n_chains``.
        """
        if not isinstance(self.samples, dict):
            raise TypeError("chain_samples requires dict-valued samples")
        reshaped: dict[str, np.ndarray] = {}
        for name, values in self.samples.items():
            array = np.asarray(values)
            if array.shape[0] % self.n_chains:
                raise ValueError(
                    f"parameter '{name}' has {array.shape[0]} draws, which is "
                    f"not divisible by n_chains={self.n_chains}"
                )
            reshaped[name] = array.reshape(
                (self.n_chains, array.shape[0] // self.n_chains) + array.shape[1:]
            )
        return reshaped

    def _convert_to_arviz(self, samples: dict[str, np.ndarray] | xr.Dataset) -> Any:
        """Convert samples to ArviZ InferenceData format.

        Draws arrive pooled across chains; ArviZ needs an explicit chain axis
        to compute R-hat and split-ESS, so the pooled axis is unpacked here.
        """
        if isinstance(samples, dict):
            # ArviZ requires observed_data to be a dict; wrap raw arrays
            obs_data = self.data
            if obs_data is not None and not isinstance(obs_data, dict):
                obs_data = {"observed": np.asarray(obs_data)}
            posterior = samples
            if self.n_chains > 1:
                try:
                    posterior = self.chain_samples()
                except ValueError:
                    # Ragged draw counts: fall back to a single pooled chain
                    # rather than mislabelling which draw came from which chain.
                    posterior = samples
            return az.from_dict(posterior=posterior, observed_data=obs_data)
        # An xarray Dataset already carries chain/draw dims, so it is wrapped
        # as-is rather than reshaped.
        return az.InferenceData(posterior=samples)

    def summary(self, parameters: list[str] | None = None) -> pd.DataFrame:
        """
        Summarize the posterior distribution.

        Parameters
        ----------
        parameters : list of str, optional
            Parameters to include in the summary. If None, include all.

        Returns
        -------
        pandas.DataFrame
            Summary statistics for the posterior
        """
        return az.summary(self.arviz_data, var_names=parameters)

    def plot_trace(self, parameters: list[str] | None = None) -> None:
        """
        Plot MCMC traces for the posterior samples.

        Parameters
        ----------
        parameters : list of str, optional
            Parameters to plot. If None, plot all.
        """
        az.plot_trace(self.arviz_data, var_names=parameters)
        plt.tight_layout()

    def plot_posterior(self, parameters: list[str] | None = None) -> None:
        """
        Plot posterior distributions.

        Parameters
        ----------
        parameters : list of str, optional
            Parameters to plot. If None, plot all.
        """
        az.plot_posterior(self.arviz_data, var_names=parameters)
        plt.tight_layout()

    def plot_forest(self, parameters: list[str] | None = None) -> None:
        """
        Forest plot of posterior distributions.

        Parameters
        ----------
        parameters : list of str, optional
            Parameters to plot. If None, plot all.
        """
        az.plot_forest(self.arviz_data, var_names=parameters)
        plt.tight_layout()

    def plot_spatial_prediction(
        self, grid: np.ndarray | None = None, uncertainty: bool = True
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot spatial predictions from the posterior.

        Parameters
        ----------
        grid : array-like, optional
            Grid points to predict on. If None, use a default grid.
        uncertainty : bool, default=True
            Whether to plot uncertainty bounds

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            Plot objects
        """
        return self.model.plot_prediction(self, grid=grid, uncertainty=uncertainty)

    def predict(
        self, X_new: np.ndarray, samples: int = 100, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Make predictions at new locations using the posterior.

        Parameters
        ----------
        X_new : array-like
            New locations to predict at
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
        return self.model.predict(
            X_new, posterior=self, samples=samples, return_std=return_std
        )

    def credible_interval(
        self, parameter: str, alpha: float = 0.05
    ) -> tuple[float, float]:
        """
        Compute credible interval for a parameter.

        Parameters
        ----------
        parameter : str
            Name of the parameter
        alpha : float, default=0.05
            Significance level (e.g., 0.05 for 95% CI)

        Returns
        -------
        lower, upper : float, float
            Lower and upper bounds of the credible interval
        """
        if not isinstance(alpha, (int, float, np.floating, np.integer)):
            raise TypeError("alpha must be a real number")
        level = float(alpha)
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("alpha must be finite and strictly between zero and one")
        try:
            param_samples = (
                np.asarray(self.arviz_data.posterior[parameter].values)
                .astype(float)
                .reshape(-1)
            )
        except (AttributeError, KeyError) as exc:
            raise KeyError(
                f"posterior does not contain parameter {parameter!r}"
            ) from exc
        if param_samples.size == 0 or not np.all(np.isfinite(param_samples)):
            raise ValueError("posterior parameter samples must be non-empty and finite")
        lower = np.percentile(param_samples, 100.0 * level / 2.0)
        upper = np.percentile(param_samples, 100.0 * (1.0 - level / 2.0))
        return float(lower), float(upper)

    def posterior_predictive(
        self, X: np.ndarray | None = None, samples: int = 100
    ) -> np.ndarray:
        """
        Generate posterior predictive samples.

        Parameters
        ----------
        X : array-like, optional
            Locations to generate predictions for. If None, use observed locations.
        samples : int, default=100
            Number of posterior samples to use

        Returns
        -------
        ndarray
            Posterior predictive samples
        """
        return self.model.posterior_predictive(posterior=self, X=X, samples=samples)

    def epistemic_uncertainty(self, parameter: str) -> float:
        """
        Marginal posterior standard deviation of a parameter.

        This is the parameter-level share of the predictive uncertainty: it
        measures how much the data constrained the parameter, independent of
        the observation noise. Pair it with
        :meth:`predictive_interval` for the full predictive picture.

        Parameters
        ----------
        parameter : str
            Name of the parameter.

        Returns
        -------
        float
            Standard deviation of the pooled draws for ``parameter``.

        Raises
        ------
        ValueError
            If the posterior does not contain the parameter.
        """
        try:
            samples = np.asarray(self.arviz_data.posterior[parameter].values).astype(
                float
            )
        except (AttributeError, KeyError) as exc:
            raise KeyError(
                f"posterior does not contain parameter {parameter!r}"
            ) from exc
        values = samples.reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("posterior parameter samples must be finite and non-empty")
        return float(np.std(values))

    def predictive_interval(
        self,
        X: np.ndarray | None = None,
        level: float = 0.95,
        samples: int = 200,
        random_seed: SeedLike = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Posterior predictive interval ``(mean, lower, upper)`` at ``level``.

        The interval is formed from the empirical quantiles of the posterior
        predictive draws at each prediction point, so its coverage is what a
        held-out observation is expected to satisfy at the nominal rate.

        Parameters
        ----------
        X : array-like, optional
            Locations to predict at. If None, use the observed locations.
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

        Raises
        ------
        ValueError
            If the model's ``posterior_predictive`` does not accept
            ``random_seed``, or ``level`` is outside ``(0, 1)``.
        """
        interval_level = float(level)
        if not np.isfinite(interval_level) or not 0.0 < interval_level < 1.0:
            raise ValueError(
                "level must be a finite probability strictly between zero and one"
            )
        # ``self.model`` is declared as the abstract base, whose posterior
        # predictive signature has no ``random_seed``; concrete models extend
        # it, so it is routed through an ``Any`` handle.
        model: Any = self.model
        try:
            draws = model.posterior_predictive(
                posterior=self, X=X, samples=samples, random_seed=random_seed
            )
        except TypeError:  # pragma: no cover - legacy signature fallback
            draws = model.posterior_predictive(posterior=self, X=X, samples=samples)
        draws = np.asarray(draws, dtype=float)
        if draws.ndim == 1:
            draws = draws.reshape(1, -1)
        tail = (1.0 - interval_level) / 2.0
        mean = np.asarray(np.mean(draws, axis=0), dtype=float)
        lower = np.asarray(np.percentile(draws, 100.0 * tail, axis=0), dtype=float)
        upper = np.asarray(
            np.percentile(draws, 100.0 * (1.0 - tail), axis=0), dtype=float
        )
        return mean, lower, upper

    def coverage(
        self,
        X: np.ndarray,
        y: np.ndarray,
        level: float = 0.95,
        samples: int = 200,
        random_seed: SeedLike = None,
    ) -> float:
        """
        Empirical coverage of the predictive interval over held-out data.

        Computes the fraction of ``y`` falling inside the model's posterior
        predictive interval at ``level``. A calibrated model returns a value
        near ``level``; values consistently below it signal overconfidence.
        Use :func:`geo_infer_bayes.core.evaluation.empirical_coverage` directly
        to compare against an externally supplied interval.

        Parameters
        ----------
        X : array-like of shape (n_points,)
            Held-out locations.
        y : array-like of shape (n_points,)
            Held-out observed values.
        level : float, default=0.95
            Nominal coverage of the interval, in ``(0, 1)``.
        samples : int, default=200
            Number of posterior predictive draws to base the interval on.
        random_seed : SeedLike, optional
            Seed or generator for the observation-noise draws.

        Returns
        -------
        float
            Fraction of observations inside the ``level`` predictive interval.
        """
        _, lower, upper = self.predictive_interval(
            X=X, level=level, samples=samples, random_seed=random_seed
        )
        return empirical_coverage(y, lower, upper)
