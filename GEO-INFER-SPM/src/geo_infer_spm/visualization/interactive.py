"""
Interactive visualization tools for GEO-INFER-SPM

This module provides interactive visualization capabilities using
web-based plotting libraries for exploratory data analysis and
result presentation.
"""

import numpy as np
from typing import Any
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..models.data_models import SPMResult


def create_interactive_map(
    spm_result: SPMResult,
    contrast_idx: int = 0,
    map_type: str = "scattergeo",
    **kwargs: Any,
) -> Any | None:
    """
    Create interactive geographical map of SPM results.

    Args:
        spm_result: SPM analysis results
        contrast_idx: Index of contrast to visualize
        map_type: Type of map ('scattergeo', 'choropleth')
        **kwargs: Additional plotting parameters

    Returns:
        Plotly figure object or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn(
            "plotly not available for interactive visualization", stacklevel=2
        )
        return None

    if not isinstance(contrast_idx, int) or contrast_idx < 0:
        raise ValueError("contrast_idx must be a non-negative integer")
    coordinates = np.asarray(spm_result.spm_data.coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or coordinates.shape[0] == 0:
        raise ValueError("spm_result coordinates must be a non-empty (n, >=2) array")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("spm_result coordinates must contain only finite values")

    if contrast_idx < len(spm_result.contrasts):
        contrast = spm_result.contrasts[contrast_idx]
        stat_values = (
            contrast.t_statistic.flatten()
            if contrast.t_statistic.ndim > 1
            else contrast.t_statistic
        )
        stat_values = np.asarray(stat_values, dtype=float).reshape(-1)
        if len(stat_values) != len(coordinates):
            raise ValueError("contrast statistics must align with coordinates")
        if contrast.significance_mask is not None and len(
            np.asarray(contrast.significance_mask).reshape(-1)
        ) != len(coordinates):
            raise ValueError("contrast significance_mask must align with coordinates")
    else:
        # Fallback to beta coefficients when no contrasts available
        contrast = None
        beta = np.asarray(spm_result.beta_coefficients, dtype=float)
        if beta.size == 0:
            raise ValueError("beta_coefficients must not be empty")
        if beta.ndim == 1 and len(beta) == len(coordinates):
            stat_values = beta
        elif beta.ndim == 2 and beta.shape[0] == len(coordinates):
            stat_values = beta[:, 0]
        elif beta.ndim == 2 and beta.shape[1] == len(coordinates):
            stat_values = beta[0]
        else:
            stat_values = np.full(len(coordinates), float(np.mean(beta)))
        if not np.all(np.isfinite(stat_values)):
            raise ValueError("statistical values must be finite")

    # Prepare data

    # Create hover information
    hover_text = []
    for i in range(len(coordinates)):
        sig_status = (
            "Significant"
            if (
                contrast is not None
                and hasattr(contrast, "significance_mask")
                and contrast.significance_mask is not None
                and contrast.significance_mask[i]
            )
            else "Not significant"
        )
        p_val_str = f"{contrast.p_values[i]:.3f}" if contrast is not None else "N/A"
        hover_text.append(
            f"Point {i}<br>"
            f"Longitude: {coordinates[i, 0]:.4f}<br>"
            f"Latitude: {coordinates[i, 1]:.4f}<br>"
            f"T-statistic: {stat_values[i]:.3f}<br>"
            f"P-value: {p_val_str}<br>"
            f"Status: {sig_status}"
        )

    if map_type == "scattergeo":
        # Create scatter geo plot
        fig = go.Figure(
            data=go.Scattergeo(
                lon=coordinates[:, 0],
                lat=coordinates[:, 1],
                text=hover_text,
                mode="markers",
                marker=dict(
                    size=8,
                    color=stat_values,
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="T-statistic"),
                    line=dict(width=1, color="black"),
                    # Highlight significant points
                    symbol=(
                        "star"
                        if (
                            contrast is not None
                            and contrast.significance_mask is not None
                            and np.any(contrast.significance_mask)
                        )
                        else "circle"
                    ),
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )

        # Update layout for geographical projection
        fig.update_layout(
            title=f"SPM Statistical Map - {getattr(contrast, 'name', f'Contrast {contrast_idx}')}",
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="RebeccaPurple",
                projection_type="natural earth",
                showland=True,
                landcolor="LightGreen",
                showocean=True,
                oceancolor="LightBlue",
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
        )

    elif map_type == "choropleth":
        # Deferred: see docs/deferred_statistical_methods.md
        # ("Choropleth rendering (polygon data)").
        warnings.warn(
            "Choropleth map requires polygon data. Using scatter plot instead.",
            stacklevel=2,
        )
        return create_interactive_map(spm_result, contrast_idx, "scattergeo", **kwargs)

    else:
        raise ValueError(f"Unknown map type: {map_type}")

    return fig


def create_dashboard(
    spm_result: SPMResult, include_diagnostics: bool = True
) -> Any | None:
    """
    Create comprehensive interactive dashboard of SPM results.

    Args:
        spm_result: SPM analysis results
        include_diagnostics: Whether to include diagnostic plots

    Returns:
        Plotly figure with dashboard or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("plotly not available for dashboard creation", stacklevel=2)
        return None

    # Create subplot figure
    n_rows = 2 if include_diagnostics else 1
    n_cols = 3

    subplot_titles = []
    if spm_result.contrasts:
        subplot_titles.append("Statistical Map")
        subplot_titles.append("T-Statistic Distribution")
        subplot_titles.append("P-Value Distribution")

    if include_diagnostics:
        subplot_titles.extend(["Residuals vs Fitted", "Q-Q Plot", "Cook's Distance"])

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[
            [{"type": "scattergeo"}, {"type": "histogram"}, {"type": "histogram"}],
            (
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
                if include_diagnostics
                else []
            ),
        ],
    )

    # Add statistical map
    if spm_result.contrasts:
        contrast = spm_result.contrasts[0]  # Use first contrast
        coordinates = spm_result.spm_data.coordinates
        stat_values = np.asarray(contrast.t_statistic, dtype=float).reshape(-1)
        if len(stat_values) != len(spm_result.spm_data.coordinates):
            raise ValueError("contrast statistics must align with coordinates")

        fig.add_trace(
            go.Scattergeo(
                lon=coordinates[:, 0],
                lat=coordinates[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=stat_values,
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="T-statistic", x=0.25),
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # T-statistic histogram
        fig.add_trace(
            go.Histogram(x=stat_values, nbinsx=30, showlegend=False), row=1, col=2
        )

        # P-value histogram
        fig.add_trace(
            go.Histogram(x=contrast.p_values, nbinsx=30, showlegend=False), row=1, col=3
        )

    # Add diagnostic plots
    if include_diagnostics:
        residuals = spm_result.residuals
        fitted = spm_result.spm_data.data - residuals

        # Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=fitted,
                y=residuals,
                mode="markers",
                marker=dict(size=4, opacity=0.6),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Q-Q plot
        from scipy import stats

        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode="markers",
                marker=dict(size=4, opacity=0.6),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=slope * osm + intercept,
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Deferred: see docs/deferred_statistical_methods.md
        # ("Cook's distance (diagnostic plots)").
        n, p = spm_result.design_matrix.matrix.shape
        mse = np.sum(residuals**2) / max(n - p, 1)
        hat_matrix = (
            spm_result.design_matrix.matrix
            @ np.linalg.pinv(
                spm_result.design_matrix.matrix.T @ spm_result.design_matrix.matrix
            )
            @ spm_result.design_matrix.matrix.T
        )
        leverage = np.clip(np.diag(hat_matrix), 0.0, 1.0 - np.finfo(float).eps)
        cooks_d = (residuals**2 / max(p * mse, np.finfo(float).eps)) * (
            leverage / (1 - leverage) ** 2
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(cooks_d))),
                y=cooks_d,
                mode="markers",
                marker=dict(size=4, opacity=0.6),
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    # Update layout
    fig.update_layout(height=800, title_text="SPM Analysis Dashboard", showlegend=False)

    # Update geo subplot
    fig.update_geos(
        showframe=False, showcoastlines=True, projection_type="natural earth"
    )

    return fig


def create_time_series_explorer(spm_result: SPMResult) -> Any | None:
    """
    Create interactive time series explorer for temporal SPM data.

    Args:
        spm_result: SPM analysis results with temporal dimension

    Returns:
        Plotly figure for time series exploration or None
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("plotly not available for time series explorer", stacklevel=2)
        return None

    if not spm_result.spm_data.has_temporal:
        warnings.warn(
            "No temporal data available for time series explorer", stacklevel=2
        )
        return None

    import numpy as np

    data = (
        spm_result.spm_data.data
    )  # shape: (n_locations, n_timepoints) or (n_timepoints,)
    time_labels = getattr(spm_result.spm_data, "time_labels", None)

    if data.ndim == 1:
        # Single time series
        n_time = len(data)
        mean_ts = data
        std_ts = np.zeros_like(data)
    else:
        # Multiple locations — show spatial mean ± 1 std
        n_time = data.shape[1] if data.ndim == 2 else data.shape[0]
        mean_ts = np.mean(data, axis=0)
        std_ts = np.std(data, axis=0)

    if time_labels is None:
        time_labels = list(range(n_time))
    elif len(time_labels) != n_time:
        raise ValueError("time_labels must have one value per time point")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Spatial Mean ± 1 SD", "Residual Time Series"],
        shared_xaxes=True,
        vertical_spacing=0.15,
    )

    # Row 1: Mean ± confidence band
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=(mean_ts + std_ts).tolist(),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=(mean_ts - std_ts).tolist(),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(68,68,255,0.2)",
            name="±1 SD",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=mean_ts.tolist(),
            mode="lines+markers",
            name="Spatial Mean",
            line=dict(color="rgb(68,68,255)", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Row 2: Residual time series (if available)
    residuals = spm_result.residuals
    if residuals is not None and residuals.ndim >= 1:
        if residuals.ndim == 1:
            res_ts = residuals
        else:
            res_ts = (
                np.mean(residuals, axis=0)
                if residuals.shape[1] == n_time
                else residuals[:n_time]
            )

        fig.add_trace(
            go.Scatter(
                x=time_labels[: len(res_ts)],
                y=res_ts.tolist(),
                mode="lines+markers",
                name="Mean Residual",
                line=dict(color="rgb(255,68,68)", width=1),
                marker=dict(size=3),
            ),
            row=2,
            col=1,
        )
        # Zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        title="SPM Time Series Explorer",
        height=700,
        xaxis2_title="Time",
        yaxis_title="Value",
        yaxis2_title="Residual",
        showlegend=True,
    )

    return fig
