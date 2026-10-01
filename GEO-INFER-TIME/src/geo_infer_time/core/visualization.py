"""
Temporal Visualization Module for GEO-INFER-TIME.

Provides visualization methods for time series analysis including plots
for decomposition, forecasts, anomalies, and diagnostics.
"""

import logging
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for matplotlib
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Visualization disabled.")


class TemporalVisualization:
    """
    Visualization engine for temporal analysis.

    Provides methods for creating various time series plots including
    line plots, decomposition panels, forecast visualizations, and more.
    """

    def __init__(
        self, style: str = "seaborn-v0_8-whitegrid", figsize: tuple[int, int] = (12, 6)
    ):
        """
        Initialize visualization engine.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        if len(figsize) != 2 or any(
            not isinstance(value, (int, float)) or value <= 0 for value in figsize
        ):
            raise ValueError("figsize must contain two positive numbers")
        self.figsize = figsize
        self.style = style

    def _new_figure(self, *args: Any, **kwargs: Any) -> Any:
        """Create a figure under this instance's style without global mutation."""
        if not HAS_MATPLOTLIB:
            return None, None
        try:
            with plt.style.context(self.style):
                return plt.subplots(*args, **kwargs)
        except OSError:
            with plt.style.context("default"):
                return plt.subplots(*args, **kwargs)

    @staticmethod
    def _series(values: list[float], name: str) -> np.ndarray:
        """Validate a non-empty finite numeric series."""
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _x_values(
        timestamps: list | None, length: int, name: str = "timestamps"
    ) -> Any:
        """Validate optional timestamps and return a plotting x-axis."""
        if timestamps is None:
            return np.arange(length)
        if len(timestamps) != length:
            raise ValueError(f"{name} must have one value per observation")
        return timestamps

    @staticmethod
    def _save(fig: Any, save_path: Path | None) -> None:
        """Persist a figure and create its parent directory when needed."""
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {path}")

    def plot_timeseries(
        self,
        values: list[float],
        timestamps: list | None = None,
        title: str = "Time Series",
        ylabel: str = "Value",
        xlabel: str = "Time",
        color: str = "#2E86AB",
        show_grid: bool = True,
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create a basic time series plot.

        Args:
            values: Time series values
            timestamps: Optional timestamps (index otherwise)
            title: Plot title
            ylabel: Y-axis label
            xlabel: X-axis label
            color: Line color
            show_grid: Whether to show grid
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure or None if not available
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available"}

        values_array = self._series(values, "values")
        x = self._x_values(timestamps, len(values_array))
        fig, ax = self._new_figure(figsize=self.figsize)
        ax.plot(x, values_array, color=color, linewidth=1.5)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        if show_grid:
            ax.grid(True, alpha=0.3)

        if timestamps is not None and len(timestamps) > 0:
            if hasattr(timestamps[0], "strftime"):
                fig.autofmt_xdate()

        fig.tight_layout()
        self._save(fig, save_path)

        return fig

    def plot_decomposition(
        self,
        trend: list[float],
        seasonal: list[float],
        residual: list[float],
        original: list[float] | None = None,
        timestamps: list | None = None,
        title: str = "Time Series Decomposition",
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create a decomposition plot with trend, seasonal, and residual panels.

        Args:
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component
            original: Optional original series
            timestamps: Optional timestamps
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available"}

        component_arrays = [
            self._series(trend, "trend"),
            self._series(seasonal, "seasonal"),
            self._series(residual, "residual"),
        ]
        if len({len(array) for array in component_arrays}) != 1:
            raise ValueError("trend, seasonal, and residual must have equal lengths")
        orig_arr = None
        if original is not None:
            orig_arr = self._series(original, "original")
            if len(orig_arr) != len(component_arrays[0]):
                raise ValueError("original must match decomposition component length")
        x = self._x_values(timestamps, len(component_arrays[0]))
        n_panels = 4 if orig_arr is not None else 3
        fig, axes = self._new_figure(
            n_panels, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5)
        )
        axes = np.atleast_1d(axes)

        components: list[np.ndarray] = []
        labels = []
        colors = []

        if orig_arr is not None:
            components.append(orig_arr)
            labels.append("Original")
            colors.append("#2E86AB")

        components.extend(component_arrays)
        labels.extend(["Trend", "Seasonal", "Residual"])
        colors.extend(["#E94F37", "#4DAA57", "#7B68EE"])

        for ax, data, label, color in zip(axes, components, labels, colors):
            ax.plot(x, data, color=color, linewidth=1.2)
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)

        axes[0].set_title(title, fontsize=14, fontweight="bold")
        axes[-1].set_xlabel("Time", fontsize=11)

        fig.tight_layout()
        self._save(fig, save_path)

        return fig

    def plot_forecast(
        self,
        historical: list[float],
        forecast: list[float],
        confidence_lower: list[float] | None = None,
        confidence_upper: list[float] | None = None,
        timestamps_historical: list | None = None,
        timestamps_forecast: list | None = None,
        title: str = "Time Series Forecast",
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create a forecast plot with confidence intervals.

        Args:
            historical: Historical values
            forecast: Forecasted values
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound
            timestamps_historical: Timestamps for historical data
            timestamps_forecast: Timestamps for forecast
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available"}

        historical_array = self._series(historical, "historical")
        forecast_array = self._series(forecast, "forecast")
        if (confidence_lower is None) != (confidence_upper is None):
            raise ValueError(
                "confidence_lower and confidence_upper must be provided together"
            )
        lower = upper = None
        if confidence_lower is not None and confidence_upper is not None:
            lower = self._series(confidence_lower, "confidence_lower")
            upper = self._series(confidence_upper, "confidence_upper")
            if len(lower) != len(forecast_array) or len(upper) != len(forecast_array):
                raise ValueError("confidence bounds must match forecast length")
            if np.any(lower > upper):
                raise ValueError("confidence_lower must not exceed confidence_upper")
        n_hist = len(historical_array)
        x_hist = self._x_values(timestamps_historical, n_hist, "timestamps_historical")
        x_forecast = (
            self._x_values(
                timestamps_forecast, len(forecast_array), "timestamps_forecast"
            )
            if timestamps_forecast is not None
            else np.arange(n_hist, n_hist + len(forecast_array))
        )
        fig, ax = self._new_figure(figsize=self.figsize)
        ax.plot(
            x_hist, historical_array, color="#2E86AB", linewidth=1.5, label="Historical"
        )
        ax.plot(
            x_forecast,
            forecast_array,
            color="#E94F37",
            linewidth=2,
            linestyle="--",
            label="Forecast",
        )

        # Confidence intervals
        if lower is not None and upper is not None:
            ax.fill_between(
                x_forecast, lower, upper, color="#E94F37", alpha=0.2, label="95% CI"
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Vertical line at forecast start
        if len(x_hist) > 0 and len(x_forecast) > 0:
            ax.axvline(
                x=x_hist[-1] if hasattr(x_hist[-1], "__add__") else n_hist - 1,
                color="gray",
                linestyle=":",
                alpha=0.7,
            )

        fig.tight_layout()
        self._save(fig, save_path)

        return fig

    def plot_acf_pacf(
        self,
        acf_values: list[float],
        pacf_values: list[float] | None = None,
        confidence_bound: float = 0.0,
        title: str = "Autocorrelation Analysis",
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create ACF and PACF plots.

        Args:
            acf_values: Autocorrelation function values
            pacf_values: Partial autocorrelation function values
            confidence_bound: Significance bound (e.g., 1.96/sqrt(n))
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available"}

        acf_array = self._series(acf_values, "acf_values")
        pacf_array = (
            None if pacf_values is None else self._series(pacf_values, "pacf_values")
        )
        if not np.isfinite(confidence_bound) or confidence_bound < 0:
            raise ValueError("confidence_bound must be finite and non-negative")
        n_panels = 2 if pacf_array is not None else 1
        fig, axes = self._new_figure(
            n_panels, 1, figsize=(self.figsize[0], self.figsize[1])
        )

        if n_panels == 1:
            axes = [axes]

        # ACF
        lags = range(len(acf_array))
        axes[0].bar(lags, acf_array, color="#2E86AB", width=0.3)
        axes[0].axhline(y=0, color="black", linewidth=0.5)
        if confidence_bound > 0:
            axes[0].axhline(y=confidence_bound, color="red", linestyle="--", alpha=0.7)
            axes[0].axhline(y=-confidence_bound, color="red", linestyle="--", alpha=0.7)
        axes[0].set_ylabel("ACF", fontsize=10)
        axes[0].set_title(title, fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        # PACF
        if pacf_array is not None:
            axes[1].bar(range(len(pacf_array)), pacf_array, color="#4DAA57", width=0.3)
            axes[1].axhline(y=0, color="black", linewidth=0.5)
            if confidence_bound > 0:
                axes[1].axhline(
                    y=confidence_bound, color="red", linestyle="--", alpha=0.7
                )
                axes[1].axhline(
                    y=-confidence_bound, color="red", linestyle="--", alpha=0.7
                )
            axes[1].set_ylabel("PACF", fontsize=10)
            axes[1].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Lag", fontsize=11)

        fig.tight_layout()
        self._save(fig, save_path)

        return fig

    def plot_anomalies(
        self,
        values: list[float],
        anomaly_indices: list[int],
        timestamps: list | None = None,
        title: str = "Anomaly Detection",
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create a plot highlighting anomalies.

        Args:
            values: Time series values
            anomaly_indices: Indices of detected anomalies
            timestamps: Optional timestamps
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available"}

        values_arr = self._series(values, "values")
        x = self._x_values(timestamps, len(values_arr))
        anomaly_indices = list(anomaly_indices)
        if any(
            not isinstance(index, int) or not 0 <= index < len(values_arr)
            for index in anomaly_indices
        ):
            raise ValueError("anomaly_indices must contain valid integer positions")
        fig, ax = self._new_figure(figsize=self.figsize)

        # Plot main series
        ax.plot(x, values_arr, color="#2E86AB", linewidth=1.2, label="Values")

        # Highlight anomalies
        if anomaly_indices:
            anomaly_x = [x[i] for i in anomaly_indices]
            anomaly_y = [values_arr[i] for i in anomaly_indices]
            ax.scatter(
                anomaly_x,
                anomaly_y,
                color="#E94F37",
                s=100,
                zorder=5,
                label=f"Anomalies ({len(anomaly_indices)})",
                edgecolors="white",
                linewidths=2,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, save_path)

        return fig

    def plot_rolling_statistics(
        self,
        values: list[float],
        rolling_mean: list[float],
        rolling_std: list[float] | None = None,
        timestamps: list | None = None,
        window: int = 0,
        title: str = "Rolling Statistics",
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create a plot with rolling mean and standard deviation bands.

        Args:
            values: Original time series values
            rolling_mean: Rolling mean values
            rolling_std: Rolling standard deviation (for bands)
            timestamps: Optional timestamps
            window: Window size (for title)
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available"}

        values_arr = self._series(values, "values")
        rolling_mean_arr = self._series(rolling_mean, "rolling_mean")
        if len(rolling_mean_arr) > len(values_arr):
            raise ValueError("rolling_mean cannot be longer than values")
        if rolling_std is not None:
            rolling_std_arr = self._series(rolling_std, "rolling_std")
            if len(rolling_std_arr) != len(rolling_mean_arr) or np.any(
                rolling_std_arr < 0
            ):
                raise ValueError(
                    "rolling_std must match rolling_mean and be non-negative"
                )
        else:
            rolling_std_arr = None
        x = self._x_values(timestamps, len(values_arr))
        fig, ax = self._new_figure(figsize=self.figsize)

        # Original series
        ax.plot(
            x, values_arr, color="#2E86AB", alpha=0.5, linewidth=1, label="Original"
        )

        # Rolling mean
        # Adjust x length for rolling values
        x_rolling = x[-len(rolling_mean_arr) :]
        ax.plot(
            x_rolling,
            rolling_mean_arr,
            color="#E94F37",
            linewidth=2,
            label="Rolling Mean",
        )

        # Standard deviation bands
        if rolling_std_arr is not None:
            ax.fill_between(
                x_rolling,
                rolling_mean_arr - 2 * rolling_std_arr,
                rolling_mean_arr + 2 * rolling_std_arr,
                color="#E94F37",
                alpha=0.2,
                label="±2 Std Dev",
            )

        title_text = f"{title}" if window == 0 else f"{title} (window={window})"
        ax.set_title(title_text, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, save_path)

        return fig

    def plot_seasonality(
        self,
        values: list[float],
        period: int,
        timestamps: list | None = None,
        title: str = "Seasonal Subseries",
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create seasonal subseries plot.

        Args:
            values: Time series values
            period: Seasonal period (e.g., 12 for monthly, 7 for weekly)
            timestamps: Optional timestamps
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available"}

        if not isinstance(period, int) or period <= 0:
            raise ValueError("period must be a positive integer")
        values_arr = self._series(values, "values")
        self._x_values(timestamps, len(values_arr))
        fig, ax = self._new_figure(figsize=self.figsize)

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in np.linspace(0, 1, period)]

        for season in range(period):
            season_values = values_arr[season::period]
            season_indices = np.arange(len(season_values))
            ax.plot(
                season_indices,
                season_values,
                "o-",
                color=colors[season],
                alpha=0.7,
                markersize=4,
                label=f"Season {season + 1}",
            )

        ax.set_title(f"{title} (period={period})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(loc="best", ncol=min(4, period), fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, save_path)

        return fig

    def create_dashboard(
        self,
        values: list[float],
        timestamps: list | None = None,
        decomposition: dict[str, list[float]] | None = None,
        forecast: dict[str, Any] | None = None,
        anomalies: list[int] | None = None,
        title: str = "Time Series Dashboard",
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create a comprehensive multi-panel dashboard.

        Args:
            values: Time series values
            timestamps: Optional timestamps
            decomposition: Optional decomposition dict with 'trend', 'seasonal', 'residual'
            forecast: Optional forecast dict with 'values', 'lower', 'upper'
            anomalies: Optional list of anomaly indices
            title: Dashboard title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available"}

        values_arr = self._series(values, "values")
        x = self._x_values(timestamps, len(values_arr))
        anomalies = [] if anomalies is None else list(anomalies)
        if any(
            not isinstance(index, int) or not 0 <= index < len(values_arr)
            for index in anomalies
        ):
            raise ValueError("anomalies must contain valid integer positions")
        # Determine layout
        n_rows = 2
        n_cols = 2
        fig, axes = self._new_figure(
            n_rows, n_cols, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.5)
        )
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        # Panel 1: Original series with anomalies
        ax1 = axes[0, 0]
        ax1.plot(x, values_arr, color="#2E86AB", linewidth=1.2)
        if anomalies:
            anomaly_x = [x[i] for i in anomalies]
            anomaly_y = [values_arr[i] for i in anomalies]
            ax1.scatter(anomaly_x, anomaly_y, color="#E94F37", s=50, zorder=5)
        ax1.set_title(
            "Time Series" + (f" ({len(anomalies)} anomalies)" if anomalies else ""),
            fontsize=11,
        )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Decomposition or Distribution
        ax2 = axes[0, 1]
        if decomposition and "trend" in decomposition:
            ax2.plot(decomposition["trend"], color="#E94F37", label="Trend")
            if "seasonal" in decomposition:
                ax2.plot(
                    decomposition["seasonal"],
                    color="#4DAA57",
                    alpha=0.5,
                    label="Seasonal",
                )
            ax2.set_title("Decomposition", fontsize=11)
            ax2.legend(fontsize=8)
        else:
            ax2.hist(values_arr, bins=30, color="#2E86AB", edgecolor="white", alpha=0.7)
            ax2.axvline(
                np.mean(values_arr), color="#E94F37", linestyle="--", label="Mean"
            )
            ax2.axvline(
                np.median(values_arr), color="#4DAA57", linestyle="--", label="Median"
            )
            ax2.set_title("Distribution", fontsize=11)
            ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Rolling statistics
        ax3 = axes[1, 0]
        window = min(20, len(values_arr) // 5)
        if window > 1:
            rolling_mean = pd.Series(values_arr).rolling(window=window).mean()
            rolling_std = pd.Series(values_arr).rolling(window=window).std()
            ax3.plot(x, values_arr, color="#2E86AB", alpha=0.3, linewidth=1)
            ax3.plot(
                x, rolling_mean, color="#E94F37", linewidth=2, label="Rolling Mean"
            )
            ax3.fill_between(
                x,
                rolling_mean - 2 * rolling_std,
                rolling_mean + 2 * rolling_std,
                color="#E94F37",
                alpha=0.2,
            )
            ax3.set_title(f"Rolling Statistics (window={window})", fontsize=11)
        else:
            ax3.plot(x, values_arr, color="#2E86AB")
            ax3.set_title("Time Series", fontsize=11)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Value")
        ax3.grid(True, alpha=0.3)

        # Panel 4: Forecast or ACF
        ax4 = axes[1, 1]
        if forecast and "values" in forecast:
            n_hist = len(values_arr)
            ax4.plot(range(n_hist), values_arr, color="#2E86AB", label="Historical")
            forecast_x = range(n_hist, n_hist + len(forecast["values"]))
            ax4.plot(
                forecast_x,
                forecast["values"],
                color="#E94F37",
                linestyle="--",
                linewidth=2,
                label="Forecast",
            )
            if "lower" in forecast and "upper" in forecast:
                ax4.fill_between(
                    forecast_x,
                    forecast["lower"],
                    forecast["upper"],
                    color="#E94F37",
                    alpha=0.2,
                )
            ax4.set_title("Forecast", fontsize=11)
            ax4.legend(fontsize=8)
        else:
            # ACF plot
            from statsmodels.tsa.stattools import acf

            acf_values = acf(values_arr, nlags=min(40, len(values_arr) // 2), fft=True)
            ax4.bar(range(len(acf_values)), acf_values, color="#2E86AB", width=0.3)
            ax4.axhline(y=0, color="black", linewidth=0.5)
            conf = 1.96 / np.sqrt(len(values_arr))
            ax4.axhline(y=conf, color="red", linestyle="--", alpha=0.7)
            ax4.axhline(y=-conf, color="red", linestyle="--", alpha=0.7)
            ax4.set_title("Autocorrelation", fontsize=11)
        ax4.set_xlabel("Lag" if forecast is None else "Time")
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, save_path)

        return fig
