"""
Results visualization utilities for economic analysis.
"""

from typing import Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging


def _require_frame(data: pd.DataFrame, required: list[str], name: str) -> pd.DataFrame:
    """Validate a nonempty tabular visualization input."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if data.empty:
        raise ValueError(f"{name} must not be empty")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    return data


def _finite_values(values: Any, name: str) -> np.ndarray:
    """Return finite numeric values for plotting."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _save_figure(
    fig: plt.Figure, save_path: Path | None, logger: logging.Logger
) -> None:
    """Save a figure to a nested path without changing global plot state."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Visualization saved to %s", save_path)


def _metric_text(value: Any, suffix: str = "") -> str:
    """Format an optional finite metric without assuming it is numeric."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{value:.3f}{suffix}" if np.isfinite(value) else "N/A"


class ResultsVisualizer:
    """
    Utility class for visualizing economic analysis results.

    Provides methods for creating charts, maps, and interactive visualizations.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the ResultsVisualizer.

        Args:
            config: Optional configuration for visualization
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.style = self.config.get("style", "seaborn-v0_8")
        self.palette = self.config.get("palette", "husl")

    def plot_economic_indicators(
        self,
        data: pd.DataFrame,
        indicators: list[str],
        title: str = "Economic Indicators",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot economic indicators over time or across regions.

        Args:
            data: DataFrame with economic indicators
            indicators: List of indicators to plot
            title: Chart title
            save_path: Optional path to save the chart

        Returns:
            Matplotlib figure object
        """
        data = _require_frame(data, indicators, "data")
        if not indicators:
            raise ValueError("indicators must not be empty")
        for indicator in indicators:
            _finite_values(data[indicator], f"data[{indicator!r}]")
        fig, axes = plt.subplots(len(indicators), 1, figsize=(12, 6 * len(indicators)))
        if len(indicators) == 1:
            axes = [axes]

        for i, indicator in enumerate(indicators):
            data[indicator].plot(ax=axes[i], title=f"{indicator}")
            axes[i].set_ylabel(indicator)
            axes[i].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        _save_figure(fig, save_path, self.logger)

        return fig

    def create_choropleth_map(
        self,
        gdf: gpd.GeoDataFrame,
        value_column: str,
        title: str = "Choropleth Map",
        cmap: str = "viridis",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Create a choropleth map for spatial economic data.

        Args:
            gdf: GeoDataFrame with spatial data
            value_column: Column to visualize
            title: Map title
            cmap: Colormap name
            save_path: Optional path to save the map

        Returns:
            Matplotlib figure object
        """
        _require_frame(gdf, [value_column], "gdf")
        _finite_values(gdf[value_column], f"gdf[{value_column!r}]")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        gdf.plot(
            column=value_column,
            ax=ax,
            cmap=cmap,
            legend=True,
            legend_kwds={"shrink": 0.6},
        )

        ax.set_title(title, fontsize=16)
        ax.axis("off")

        fig.tight_layout()
        _save_figure(fig, save_path, self.logger)

        return fig

    def plot_policy_comparison(
        self,
        comparison_data: dict[str, dict[str, float]],
        metrics: list[str],
        title: str = "Policy Comparison",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Create a comparison chart for policy scenarios.

        Args:
            comparison_data: Dictionary with scenario data
            metrics: List of metrics to compare
            title: Chart title
            save_path: Optional path to save the chart

        Returns:
            Matplotlib figure object
        """
        if not comparison_data or not metrics:
            raise ValueError("comparison_data and metrics must not be empty")
        # Convert to DataFrame for easier plotting
        df_data = []
        for scenario, data in comparison_data.items():
            for metric in metrics:
                if metric in data:
                    df_data.append(
                        {"Scenario": scenario, "Metric": metric, "Value": data[metric]}
                    )

        df = pd.DataFrame(df_data)
        _require_frame(df, ["Scenario", "Metric", "Value"], "comparison data")
        _finite_values(df["Value"], "comparison values")

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        sns.barplot(data=df, x="Metric", y="Value", hue="Scenario", ax=ax)
        ax.set_title(title, fontsize=16)
        ax.set_ylabel("Impact Value")
        ax.tick_params(axis="x", rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.tight_layout()
        _save_figure(fig, save_path, self.logger)

        return fig

    def plot_distributional_effects(
        self,
        effects: dict[str, float],
        title: str = "Distributional Effects",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot distributional effects across income quintiles or regions.

        Args:
            effects: Dictionary with distributional effects
            title: Chart title
            save_path: Optional path to save the chart

        Returns:
            Matplotlib figure object
        """
        if not effects:
            raise ValueError("effects must not be empty")
        categories = list(effects.keys())
        values = _finite_values(list(effects.values()), "effects")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        bars = ax.bar(categories, values)

        # Color bars based on positive/negative values
        for bar, value in zip(bars, values):
            if value >= 0:
                bar.set_color("green")
            else:
                bar.set_color("red")

        ax.set_title(title, fontsize=16)
        ax.set_ylabel("Effect Size")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)

        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        _save_figure(fig, save_path, self.logger)

        return fig

    def create_spatial_heatmap(
        self,
        data: pd.DataFrame,
        value_column: str,
        title: str = "Spatial Heatmap",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Create a spatial heatmap visualization.

        Args:
            data: DataFrame with spatial data
            value_column: Column to visualize
            title: Chart title
            save_path: Optional path to save the chart

        Returns:
            Matplotlib figure object
        """
        _require_frame(data, [value_column], "data")
        _finite_values(data[value_column], f"data[{value_column!r}]")
        if {"latitude", "longitude"} <= set(data.columns):
            _finite_values(data["latitude"], "latitude")
            _finite_values(data["longitude"], "longitude")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Create heatmap
        if "latitude" in data.columns and "longitude" in data.columns:
            # Scatter plot with color intensity
            scatter = ax.scatter(
                data["longitude"],
                data["latitude"],
                c=data[value_column],
                cmap="viridis",
                s=50,
                alpha=0.7,
            )
            plt.colorbar(scatter, ax=ax, label=value_column)
        else:
            # Regular heatmap if no spatial coordinates
            heatmap_data = data.pivot_table(index=data.index, values=value_column)
            sns.heatmap(heatmap_data, ax=ax, cmap="viridis", annot=True, fmt=".2f")
            ax.set_title(title)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Longitude" if "longitude" in data.columns else "X")
        ax.set_ylabel("Latitude" if "latitude" in data.columns else "Y")

        fig.tight_layout()
        _save_figure(fig, save_path, self.logger)

        return fig

    def plot_time_series_decomposition(
        self,
        time_series: pd.Series,
        decomposition_type: str = "additive",
        title: str = "Time Series Decomposition",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot time series decomposition (trend, seasonal, residual).

        Args:
            time_series: Time series data
            decomposition_type: Type of decomposition ('additive', 'multiplicative')
            title: Chart title
            save_path: Optional path to save the chart

        Returns:
            Matplotlib figure object
        """
        if not isinstance(time_series, pd.Series) or time_series.empty:
            raise ValueError("time_series must be a nonempty pandas Series")
        _finite_values(time_series.to_numpy(), "time_series")
        if decomposition_type not in {"additive", "multiplicative"}:
            raise ValueError("decomposition_type must be additive or multiplicative")
        from statsmodels.tsa.seasonal import seasonal_decompose

        try:
            # Perform decomposition
            decomposition = seasonal_decompose(time_series, model=decomposition_type)

            # Create subplots
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))

            # Original series
            axes[0].plot(time_series.index, time_series.values)
            axes[0].set_title("Original Series")
            axes[0].grid(True, alpha=0.3)

            # Trend
            axes[1].plot(time_series.index, decomposition.trend)
            axes[1].set_title("Trend")
            axes[1].grid(True, alpha=0.3)

            # Seasonal
            axes[2].plot(time_series.index, decomposition.seasonal)
            axes[2].set_title("Seasonal")
            axes[2].grid(True, alpha=0.3)

            # Residual
            axes[3].plot(time_series.index, decomposition.resid)
            axes[3].set_title("Residual")
            axes[3].grid(True, alpha=0.3)

            fig.suptitle(title, fontsize=16)
            fig.tight_layout()
            _save_figure(fig, save_path, self.logger)

            return fig

        except Exception as e:
            self.logger.error(f"Time series decomposition failed: {str(e)}")
            # Return simple time series plot as fallback
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            time_series.plot(ax=ax, title=title)
            _save_figure(fig, save_path, self.logger)
            return fig

    def create_model_diagnostics_plot(
        self,
        model_results: dict[str, Any],
        title: str = "Model Diagnostics",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Create comprehensive model diagnostics visualization.

        Args:
            model_results: Dictionary with model results and diagnostics
            title: Chart title
            save_path: Optional path to save the chart

        Returns:
            Matplotlib figure object
        """
        if not isinstance(model_results, dict):
            raise TypeError("model_results must be a mapping")
        if "residuals" in model_results:
            residuals = _finite_values(model_results["residuals"], "residuals")
        else:
            residuals = None
        if "fitted_values" in model_results:
            fitted = _finite_values(model_results["fitted_values"], "fitted_values")
            if residuals is not None and len(fitted) != len(residuals):
                raise ValueError("fitted_values and residuals must have equal length")
        else:
            fitted = None
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals plot
        if residuals is not None:
            axes[0, 0].plot(residuals, "o", alpha=0.5)
            axes[0, 0].axhline(y=0, color="red", linestyle="--")
            axes[0, 0].set_title("Residuals")
            axes[0, 0].grid(True, alpha=0.3)

        # Q-Q plot for normality
        if residuals is not None:
            from scipy import stats

            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
            axes[0, 1].scatter(osm, osr, alpha=0.5)
            axes[0, 1].plot(osm, slope * osm + intercept, "r--")
            axes[0, 1].set_title("Q-Q Plot")
            axes[0, 1].grid(True, alpha=0.3)

        # Residuals vs Fitted values
        if fitted is not None and residuals is not None:
            axes[1, 0].scatter(fitted, residuals, alpha=0.5)
            axes[1, 0].axhline(y=0, color="red", linestyle="--")
            axes[1, 0].set_xlabel("Fitted Values")
            axes[1, 0].set_ylabel("Residuals")
            axes[1, 0].set_title("Residuals vs Fitted")
            axes[1, 0].grid(True, alpha=0.3)

        # Model performance metrics
        axes[1, 1].text(0.1, 0.8, f"R²: {_metric_text(model_results.get('r_squared'))}")
        axes[1, 1].text(0.1, 0.6, f"RMSE: {_metric_text(model_results.get('rmse'))}")
        axes[1, 1].text(0.1, 0.4, f"MAE: {_metric_text(model_results.get('mae'))}")
        axes[1, 1].text(
            0.1, 0.2, f"MAPE: {_metric_text(model_results.get('mape'), '%')}"
        )
        axes[1, 1].set_title("Model Metrics")
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")

        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        _save_figure(fig, save_path, self.logger)

        return fig

    def plot_spatial_autocorrelation(
        self,
        data: gpd.GeoDataFrame,
        variable: str,
        spatial_weights: np.ndarray,
        title: str = "Spatial Autocorrelation",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot spatial autocorrelation analysis including Moran's I scatterplot.

        Args:
            data: GeoDataFrame with spatial data
            variable: Variable to analyze
            spatial_weights: Spatial weights matrix
            title: Chart title
            save_path: Optional path to save the chart

        Returns:
            Matplotlib figure object
        """
        _require_frame(data, [variable], "data")
        values = _finite_values(data[variable], f"data[{variable!r}]")
        spatial_weights = np.asarray(spatial_weights, dtype=float)
        if spatial_weights.shape != (len(values), len(values)):
            raise ValueError("spatial_weights must be an n by n matrix")
        if not np.all(np.isfinite(spatial_weights)):
            raise ValueError("spatial_weights must contain finite values")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Extract values and spatial lag
        wy_values = spatial_weights @ values

        # Moran's I scatterplot
        axes[0].scatter(values, wy_values, alpha=0.7)
        axes[0].axhline(y=np.mean(wy_values), color="red", linestyle="--", alpha=0.7)
        axes[0].axvline(x=np.mean(values), color="red", linestyle="--", alpha=0.7)
        axes[0].set_xlabel(f"{variable}")
        axes[0].set_ylabel(f"Spatial Lag of {variable}")
        axes[0].set_title("Moran's I Scatterplot")
        axes[0].grid(True, alpha=0.3)

        # Local indicators of spatial association (LISA) map
        # Calculate local Moran's I
        n = len(values)
        denominator = values.T @ values / n
        local_morans = values * wy_values / denominator if denominator else np.zeros(n)

        # Simple classification for visualization
        lisa_classes = np.zeros(n)
        lisa_classes[
            (values > np.mean(values)) & (local_morans > np.mean(local_morans))
        ] = 1  # High-High
        lisa_classes[
            (values < np.mean(values)) & (local_morans < np.mean(local_morans))
        ] = 2  # Low-Low
        lisa_classes[
            (values > np.mean(values)) & (local_morans < np.mean(local_morans))
        ] = 3  # High-Low
        lisa_classes[
            (values < np.mean(values)) & (local_morans > np.mean(local_morans))
        ] = 4  # Low-High

        # Plot LISA clusters on map
        plot_data = data.copy()
        plot_data["_lisa_class"] = lisa_classes
        plot_data.plot(
            column="_lisa_class", ax=axes[1], cmap="Set1", legend=True, categorical=True
        )
        axes[1].set_title("LISA Clusters")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")

        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        _save_figure(fig, save_path, self.logger)

        return fig

    def create_interactive_dashboard(
        self,
        data_dict: dict[str, Any],
        dashboard_type: str = "economic_overview",
        output_path: Path | None = None,
    ) -> str:
        """
        Create interactive dashboard (simplified - would use Plotly/Dash in practice).

        Args:
            data_dict: Dictionary with dashboard data
            dashboard_type: Type of dashboard
            output_path: Optional path to save HTML

        Returns:
            HTML string for interactive dashboard
        """
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a mapping")
        # This is a simplified implementation
        # In practice, would use Plotly, Dash, or similar for interactive dashboards

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GEO-INFER-ECON Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .chart {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>GEO-INFER-ECON Interactive Dashboard</h1>
            <div class="dashboard">
                <div class="chart">
                    <h3>Economic Indicators</h3>
                    <div id="economic-chart" style="height: 400px;"></div>
                </div>
                <div class="chart">
                    <h3>Spatial Analysis</h3>
                    <div id="spatial-chart" style="height: 400px;"></div>
                </div>
            </div>
            <script>
                // Baseline for interactive charts
                // In practice, would create actual Plotly charts
                console.log('Dashboard data:', {data});
            </script>
        </body>
        </html>
        """

        # In a real implementation, would populate with actual data
        html = html_template.format(data=json.dumps(data_dict, default=str))
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding="utf-8")
        return html
