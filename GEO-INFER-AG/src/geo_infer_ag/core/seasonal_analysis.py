"""
Seasonal agricultural analysis functionality.
"""

from typing import Any
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr


class SeasonalAnalysis:
    """
    Perform seasonal analysis for agricultural data.

    This class provides methods for analyzing seasonal patterns in agricultural data,
    including growing season detection, phenological stage identification, and
    temporal trend analysis.

    Attributes:
        data: Time series data for analysis
        growing_season: Detected growing season periods
    """

    def __init__(
        self,
        time_series_data: pd.DataFrame | None = None,
        spatial_data: gpd.GeoDataFrame | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the seasonal analysis.

        Args:
            time_series_data: Optional time series data for analysis
            spatial_data: Optional spatial data for analysis
            config: Optional configuration parameters
        """
        self.time_series_data = time_series_data
        self.spatial_data = spatial_data
        self.config = config or {}
        self.growing_season: dict[str, Any] = {}

    def detect_growing_season(
        self,
        time_series: pd.Series | None = None,
        variable: str = "ndvi",
        method: str = "threshold",
        threshold: float = 0.3,
        smoothing_window: int = 7,
        min_length_days: int = 30,
    ) -> dict[str, Any]:
        """
        Detect growing season start, peak, and end dates.

        Args:
            time_series: Optional time series data (uses self.time_series_data if None)
            variable: Variable to use for detection (e.g., 'ndvi', 'evi')
            method: Detection method ('threshold', 'derivative', 'savitzky_golay')
            threshold: Threshold value for start/end detection
            smoothing_window: Window size for time series smoothing
            min_length_days: Minimum length of growing season in days

        Returns:
            Dictionary with growing season information

        Raises:
            ValueError: If method is invalid or data is missing
        """
        if time_series is None:
            if self.time_series_data is None:
                raise ValueError("No time series data provided")

            if variable not in self.time_series_data.columns:
                raise ValueError(f"Variable '{variable}' not found in time series data")

            time_series = self.time_series_data[variable]

        # Ensure time series is sorted by date
        time_series = time_series.sort_index()

        # Apply smoothing for noise reduction
        if smoothing_window > 1:
            time_series_smooth = time_series.rolling(
                window=smoothing_window, center=True
            ).mean()
            # Fill NaN values at edges with original values
            time_series_smooth = time_series_smooth.fillna(time_series)
        else:
            time_series_smooth = time_series

        def _season_summary(
            start: Any, end: Any, season_length: int
        ) -> dict[str, Any] | None:
            """Build a season summary with a peak strictly inside the detected interval."""
            season_data = time_series_smooth.loc[start:end]
            raw_season_data = time_series.loc[start:end]
            if len(raw_season_data) < 3:
                return None

            interior_raw = raw_season_data.iloc[1:-1]
            if interior_raw.empty:
                return None

            peak_date = interior_raw.idxmax()
            if not start < peak_date < end:
                return None

            peak_value = raw_season_data.loc[peak_date]
            return {
                "start_date": start,
                "peak_date": peak_date,
                "end_date": end,
                "length_days": season_length,
                "peak_value": peak_value,
                "mean_value": season_data.mean(),
            }

        # Detect growing season based on selected method
        if method == "threshold":
            # Threshold-based detection
            above_threshold = time_series_smooth > threshold

            # Find transitions (False -> True for start, True -> False for end)
            transitions = above_threshold.astype(int).diff().fillna(0)

            # Get start and end dates
            start_dates = time_series_smooth.index[transitions == 1].tolist()
            end_dates = time_series_smooth.index[transitions == -1].tolist()

            # Handle case where season starts before first observation
            if above_threshold.iloc[0] and len(start_dates) < len(end_dates):
                start_dates.insert(0, time_series_smooth.index[0])

            # Handle case where season ends after last observation
            if above_threshold.iloc[-1] and len(start_dates) > len(end_dates):
                end_dates.append(time_series_smooth.index[-1])

            # Match start and end dates to create seasons
            seasons = []
            for i in range(min(len(start_dates), len(end_dates))):
                start = start_dates[i]
                end = end_dates[i]

                # Filter out short seasons
                season_length = (end - start).days
                if season_length >= min_length_days:
                    season_summary = _season_summary(start, end, season_length)
                    if season_summary is not None:
                        seasons.append(season_summary)

            self.growing_season = {
                "variable": variable,
                "method": method,
                "threshold": threshold,
                "seasons": seasons,
            }

        elif method == "derivative":
            # First derivative method
            derivative = time_series_smooth.diff().fillna(0)

            # Smooth the derivative
            derivative_smooth = (
                derivative.rolling(window=smoothing_window, center=True)
                .mean()
                .fillna(derivative)
            )

            # Find positive and negative inflection points
            pos_inflection = (derivative_smooth > 0) & (derivative_smooth.shift(1) <= 0)
            neg_inflection = (derivative_smooth < 0) & (derivative_smooth.shift(1) >= 0)

            start_candidates = time_series_smooth.index[pos_inflection].tolist()
            end_candidates = time_series_smooth.index[neg_inflection].tolist()

            # Match start and end dates to create seasons
            seasons = []
            start_idx = 0

            for start_idx in range(len(start_candidates)):
                start = start_candidates[start_idx]

                # Find next negative inflection after start
                end_candidates_after_start = [e for e in end_candidates if e > start]
                if not end_candidates_after_start:
                    continue

                end = end_candidates_after_start[0]

                # Filter out short seasons
                season_length = (end - start).days
                if season_length >= min_length_days:
                    season_summary = _season_summary(start, end, season_length)
                    if season_summary is not None:
                        seasons.append(season_summary)

            self.growing_season = {
                "variable": variable,
                "method": method,
                "seasons": seasons,
            }

        else:
            raise ValueError(f"Unsupported detection method: {method}")

        return self.growing_season

    def identify_phenological_stages(
        self,
        crop_type: str,
        time_series: pd.Series | None = None,
        variable: str = "ndvi",
        reference_stages: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """
        Identify crop phenological stages using time series data.

        Args:
            crop_type: Type of crop for stage identification
            time_series: Optional time series data
            variable: Variable to use for stage identification
            reference_stages: Optional dictionary mapping stages to variable ranges

        Returns:
            Dictionary with phenological stage information

        Raises:
            ValueError: If no growing season detected or reference_stages not provided
        """
        if not self.growing_season or not self.growing_season.get("seasons"):
            raise ValueError(
                "No growing season detected. Run detect_growing_season first"
            )

        if time_series is None:
            if self.time_series_data is None:
                raise ValueError("No time series data provided")

            if variable not in self.time_series_data.columns:
                raise ValueError(f"Variable '{variable}' not found in time series data")

            time_series = self.time_series_data[variable]

        # Use default reference stages if not provided
        if reference_stages is None:
            if crop_type.lower() == "corn":
                reference_stages = {
                    "emergence": (0.2, 0.3),
                    "vegetative": (0.3, 0.7),
                    "flowering": (0.7, 0.9),
                    "grain_filling": (0.7, 0.8),
                    "maturity": (0.4, 0.7),
                    "senescence": (0.2, 0.4),
                }
            elif crop_type.lower() == "wheat":
                reference_stages = {
                    "emergence": (0.2, 0.3),
                    "tillering": (0.3, 0.5),
                    "stem_extension": (0.5, 0.7),
                    "heading": (0.7, 0.9),
                    "ripening": (0.3, 0.7),
                }
            elif crop_type.lower() == "soybean":
                reference_stages = {
                    "emergence": (0.2, 0.3),
                    "vegetative": (0.3, 0.7),
                    "flowering": (0.7, 0.9),
                    "pod_development": (0.7, 0.8),
                    "maturity": (0.3, 0.6),
                }
            else:
                # Generic stages
                reference_stages = {
                    "emergence": (0.2, 0.3),
                    "vegetative_growth": (0.3, 0.7),
                    "reproductive": (0.7, 0.9),
                    "maturity": (0.3, 0.6),
                }

        # Get the first detected season
        season = self.growing_season["seasons"][0]
        start_date = season["start_date"]
        end_date = season["end_date"]

        # Get time series data for the growing season
        season_data = time_series.loc[start_date:end_date]

        # Normalize data for comparison with reference ranges; a constant
        # series has zero range, so map it to the middle of the normalized
        # range instead of producing NaN for every observation.
        min_val = season_data.min()
        max_val = season_data.max()
        value_range = max_val - min_val
        if value_range == 0:
            normalized_data = pd.Series(0.5, index=season_data.index)
        else:
            normalized_data = (season_data - min_val) / value_range

        # Identify stages by comparing with reference ranges
        stages = {}
        for stage_name, (lower, upper) in reference_stages.items():
            # Find dates where normalized value is in the range
            stage_dates = normalized_data.index[
                (normalized_data >= lower) & (normalized_data <= upper)
            ].tolist()

            if stage_dates:
                stages[stage_name] = {
                    "start_date": stage_dates[0],
                    "end_date": stage_dates[-1],
                    "length_days": (stage_dates[-1] - stage_dates[0]).days,
                    "mean_value": season_data.loc[stage_dates].mean(),
                }

        phenology_result = {
            "crop_type": crop_type,
            "variable": variable,
            "season": {
                "start_date": start_date,
                "end_date": end_date,
                "length_days": (end_date - start_date).days,
            },
            "stages": stages,
        }

        # Store results
        self.phenological_stages = phenology_result
        return phenology_result

    def analyze_temporal_trends(
        self,
        time_series: pd.Series | None = None,
        variable: str = "ndvi",
        period: str = "daily",
        detrend: bool = False,
        window_size: int = 7,
    ) -> dict[str, Any]:
        """
        Analyze temporal trends in agricultural data.

        Args:
            time_series: Optional time series data
            variable: Variable to analyze
            period: Aggregation period ('daily', 'weekly', 'monthly', 'annual')
            detrend: Whether to remove trend before analysis
            window_size: Window size for moving averages

        Returns:
            Dictionary with trend analysis results

        Raises:
            ValueError: If time series data is missing
        """
        if time_series is None:
            if self.time_series_data is None:
                raise ValueError("No time series data provided")

            if variable not in self.time_series_data.columns:
                raise ValueError(f"Variable '{variable}' not found in time series data")

            time_series = self.time_series_data[variable]

        # Ensure time series is sorted
        time_series = time_series.sort_index()

        # Resample time series to specified period
        if period == "weekly":
            resampled = time_series.resample("W").mean()
        elif period == "monthly":
            resampled = time_series.resample("ME").mean()
        elif period == "annual":
            resampled = time_series.resample("YE").mean()
        else:
            resampled = time_series

        # Calculate moving average
        moving_avg = resampled.rolling(window=window_size, center=True).mean()

        # Detrend if requested
        detrended = None
        if detrend:
            # Create time values for regression
            time_values = np.arange(len(resampled))

            # Fit linear trend
            mask = ~np.isnan(resampled)
            if np.sum(mask) > 1:  # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_values[mask], resampled.values[mask]
                )

                # Calculate trend line
                trend_line = intercept + slope * time_values

                # Remove trend
                detrended = resampled - trend_line
            else:
                detrended = resampled.copy()

        # Calculate statistics
        trend_data_out: dict[str, Any] = {
            "original": resampled,
            "moving_avg": moving_avg,
        }
        trend_results: dict[str, Any] = {
            "variable": variable,
            "period": period,
            "statistics": {
                "mean": resampled.mean(),
                "std": resampled.std(),
                "min": resampled.min(),
                "max": resampled.max(),
                "median": resampled.median(),
            },
            "data": trend_data_out,
        }

        # Add trend analysis if there are enough data points
        if len(resampled.dropna()) > 2:
            time_values = np.arange(len(resampled))
            mask = ~np.isnan(resampled)

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                time_values[mask], resampled.values[mask]
            )

            trend_results["trend_analysis"] = {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_err": std_err,
                "trend_line": intercept + slope * time_values,
            }

            if detrend:
                trend_results["data"]["detrended"] = detrended

        # Store results
        self.trend_results = trend_results
        return trend_results

    def analyze_spatial_temporal_patterns(
        self,
        dataset: xr.Dataset | None = None,
        variable: str = "ndvi",
        time_dim: str = "time",
        lat_dim: str = "lat",
        lon_dim: str = "lon",
    ) -> dict[str, Any]:
        """
        Analyze spatial-temporal patterns in agricultural data.

        Args:
            dataset: Optional xarray Dataset with spatial-temporal data
            variable: Variable to analyze
            time_dim: Name of time dimension in dataset
            lat_dim: Name of latitude dimension
            lon_dim: Name of longitude dimension

        Returns:
            Dictionary with spatial-temporal analysis results

        Raises:
            ValueError: If dataset is not provided
        """
        if dataset is None:
            raise ValueError("No dataset provided for spatial-temporal analysis")

        if variable not in dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")

        # Extract the data array for the variable
        da = dataset[variable]

        # Calculate temporal mean
        temporal_mean = da.mean(dim=time_dim)

        # Calculate temporal variability (standard deviation)
        temporal_std = da.std(dim=time_dim)

        # Calculate coefficient of variation (CV)
        coefficient_of_variation = temporal_std / temporal_mean

        # Calculate spatial mean (average across all locations for each time)
        spatial_mean = da.mean(dim=[lat_dim, lon_dim])

        # Calculate regional statistics
        def _scalar(value: Any) -> float:
            """Extract one numeric value from numpy/xarray scalar output."""
            return float(np.asarray(value).reshape(-1)[0])

        results = {
            "variable": variable,
            "temporal_stats": {
                "mean": temporal_mean,
                "std": temporal_std,
                "cv": coefficient_of_variation,
            },
            "spatial_series": spatial_mean,
            "dataset_stats": {
                "global_mean": _scalar(da.mean().values),
                "global_std": _scalar(da.std().values),
                "global_min": _scalar(da.min().values),
                "global_max": _scalar(da.max().values),
            },
        }

        # Store results
        self.spatial_temporal_results = results
        return results

    def plot_growing_season(self, ax: Any = None, **kwargs: Any) -> Any:
        """
        Plot the detected growing season.

        Args:
            ax: Optional matplotlib axis for plotting
            **kwargs: Additional keyword arguments for plotting

        Returns:
            The plot axis

        Raises:
            ValueError: If no growing season detected
        """
        if not self.growing_season or not self.growing_season.get("seasons"):
            raise ValueError("No growing season detected")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Get time series data
        variable = self.growing_season["variable"]

        if (
            self.time_series_data is None
            or variable not in self.time_series_data.columns
        ):
            raise ValueError(
                f"Cannot plot: time series data for '{variable}' not available"
            )

        time_series = self.time_series_data[variable]

        # Plot original time series
        time_series.plot(ax=ax, label=variable.upper(), **kwargs)

        # Plot growing seasons
        for i, season in enumerate(self.growing_season["seasons"]):
            start = season["start_date"]
            peak = season["peak_date"]
            end = season["end_date"]

            # Plot vertical lines for start, peak, and end
            ax.axvline(
                x=start,
                color="g",
                linestyle="--",
                alpha=0.7,
                label="Season Start" if i == 0 else "",
            )
            ax.axvline(
                x=peak,
                color="r",
                linestyle="--",
                alpha=0.7,
                label="Season Peak" if i == 0 else "",
            )
            ax.axvline(
                x=end,
                color="b",
                linestyle="--",
                alpha=0.7,
                label="Season End" if i == 0 else "",
            )

            # Highlight the growing season period
            ax.axvspan(start, end, alpha=0.2, color="gray")

            # Add text labels
            ax.text(
                start,
                time_series.max() * 0.8,
                f"S{i + 1}",
                bbox=dict(facecolor="white", alpha=0.7),
            )

        # Add labels and legend
        ax.set_xlabel("Date")
        ax.set_ylabel(variable.upper())
        ax.set_title(f"Growing Season Detection ({variable.upper()})")
        ax.legend()

        return ax
