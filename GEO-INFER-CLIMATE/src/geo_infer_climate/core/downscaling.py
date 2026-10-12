"""
Climate downscaling methods module.

Implements statistical bias correction and interpolation-based downscaling
to convert coarse-resolution climate model output to fine-resolution data.
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class DownscalingMethods:
    """
    Statistical downscaling and bias-correction methods.

    Bias correction supports linear (mean/variance rescaling) and empirical
    quantile mapping. Downscaling is interpolation-based (bilinear or
    nearest neighbour); this module does not implement regression or
    machine-learning downscaling.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize downscaling methods.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def bias_correction(
        self,
        model_data: xr.DataArray,
        observed_data: xr.DataArray,
        method: str = "linear",
    ) -> xr.DataArray:
        """
        Apply bias correction to climate model data.

        Args:
            model_data: Climate model output with a ``time`` dimension
            observed_data: Observed/reference data on the same grid as
                ``model_data`` (same non-time dimensions)
            method: Correction method ('linear' or 'quantile')

        Returns:
            Bias-corrected data
        """
        if method == "linear":
            return self._linear_bias_correction(model_data, observed_data)
        elif method == "quantile":
            return self._quantile_mapping(model_data, observed_data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _linear_bias_correction(
        self, model: xr.DataArray, observed: xr.DataArray
    ) -> xr.DataArray:
        """Linear bias correction (mean and variance rescaling)."""
        # Calculate bias statistics
        model_mean = model.mean(dim="time")
        observed_mean = observed.mean(dim="time")

        model_std = model.std(dim="time")
        observed_std = observed.std(dim="time")

        # Apply correction
        corrected = (model - model_mean) * (
            observed_std / (model_std + 1e-10)
        ) + observed_mean

        return corrected

    def _quantile_mapping(
        self, model: xr.DataArray, observed: xr.DataArray
    ) -> xr.DataArray:
        """Empirical quantile mapping bias correction.

        For every grid cell, quantiles of the model and observed time
        series define an empirical transfer function which is applied by
        piecewise-linear interpolation (:func:`numpy.interp`). Values
        outside the calibrated quantile range are clamped to the endpoint
        corrections, so the correction is bounded.
        """

        def _map_values(model_vals: np.ndarray, obs_vals: np.ndarray) -> np.ndarray:
            model_valid = model_vals[np.isfinite(model_vals)]
            obs_valid = obs_vals[np.isfinite(obs_vals)]
            if model_valid.size == 0 or obs_valid.size == 0:
                return np.full(model_vals.shape, np.nan)

            probs = np.linspace(0.02, 0.98, 25)
            model_q = np.quantile(model_valid, probs)
            obs_q = np.quantile(obs_valid, probs)

            mapped = np.interp(model_vals, model_q, obs_q)
            return np.where(np.isfinite(model_vals), mapped, np.nan)

        corrected = xr.apply_ufunc(
            _map_values,
            model,
            observed,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="forbidden",
            keep_attrs=True,
        )
        return corrected

    def statistical_downscaling(
        self, coarse_data: xr.DataArray, method: str = "linear"
    ) -> xr.DataArray:
        """
        Downscale coarse-resolution climate data onto a finer grid.

        This is an interpolation-only downscaling: the coarse field is
        refined by a factor of two in latitude and longitude using xarray
        interpolation. Regression- and machine-learning-based downscaling
        (e.g. topography-aware methods) are not implemented.

        Args:
            coarse_data: Coarse resolution climate data with ``lat`` and
                ``lon`` dimensions
            method: Interpolation method ('linear' or 'nearest')

        Returns:
            Downscaled fine-resolution data
        """
        if method not in ("linear", "nearest"):
            raise ValueError(f"Unsupported interpolation method: {method}")

        fine = coarse_data.interp(
            lat=np.linspace(
                float(coarse_data.lat.min()),
                float(coarse_data.lat.max()),
                len(coarse_data.lat) * 2,
            ),
            lon=np.linspace(
                float(coarse_data.lon.min()),
                float(coarse_data.lon.max()),
                len(coarse_data.lon) * 2,
            ),
            method=method,
        )
        return fine
