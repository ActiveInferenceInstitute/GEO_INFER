"""
Visualization utilities for GEO-INFER-BIO.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
from typing import Any
from pathlib import Path


class BioVisualizer:
    """A class for visualizing biological data with spatial context."""

    def __init__(self) -> None:
        """Initialize the BioVisualizer."""
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            try:
                plt.style.use("seaborn")
            except OSError:
                plt.style.use("ggplot")
        sns.set_palette("husl")

    @staticmethod
    def _as_frame(data: Any, required_columns: tuple[str, ...]) -> pd.DataFrame:
        """Normalize supported inputs and validate columns used by a plot."""
        if isinstance(data, dict):
            if not data:
                raise ValueError("data mapping must contain at least one table")
            data = next(iter(data.values()))
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame or a non-empty mapping")
        missing = [column for column in required_columns if column not in data]
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")
        if data.empty:
            raise ValueError("data must contain at least one row")
        return data.copy()

    @staticmethod
    def _spatial_frame(data: Any, value_column: str | None = None) -> pd.DataFrame:
        """Validate a table that will be converted to point geometries."""
        required: tuple[str, ...] = ("longitude", "latitude")
        if value_column is not None:
            required += (value_column,)
        frame = BioVisualizer._as_frame(data, required)
        for column in (
            ("longitude", "latitude")
            if value_column is None
            else ("longitude", "latitude", value_column)
        ):
            values = pd.to_numeric(frame[column], errors="coerce")
            if not values.notna().all():
                raise ValueError(f"{column} must contain finite numeric values")
            frame[column] = values
        if ((frame["longitude"] < -180) | (frame["longitude"] > 180)).any():
            raise ValueError("longitude values must be between -180 and 180")
        if ((frame["latitude"] < -90) | (frame["latitude"] > 90)).any():
            raise ValueError("latitude values must be between -90 and 90")
        return frame

    @staticmethod
    def _finish(fig: plt.Figure, output_path: str | None) -> plt.Figure:
        """Lay out, optionally persist, and release a rendered figure."""
        fig.tight_layout()
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_spatial_distribution(
        self,
        data: Any,
        output_path: str | None = None,
        title: str = "Spatial Distribution",
    ) -> plt.Figure:
        """
        Plot spatial distribution of biological features.

        Args:
            data: DataFrame or dict of DataFrames containing spatial data
            output_path: Optional path to save the plot
            title: Plot title
        """
        data = self._spatial_frame(data)
        gdf = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data.longitude, data.latitude),
            crs="EPSG:4326",
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.plot(ax=ax, markersize=100)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        return self._finish(fig, output_path)

    def plot_gc_distribution(
        self,
        data: pd.DataFrame,
        output_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot GC content distribution.

        Args:
            data: DataFrame containing GC content data
            output_path: Optional path to save the plot
        """
        data = self._spatial_frame(data, "gc_content")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        sns.histplot(data=data, x="gc_content", ax=ax1)
        ax1.set_title("GC Content Distribution")
        ax1.set_xlabel("GC Content (%)")
        ax1.set_ylabel("Count")

        # Spatial plot
        gdf = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data.longitude, data.latitude),
            crs="EPSG:4326",
        )
        gdf.plot(column="gc_content", ax=ax2, legend=True)
        ax2.set_title("Spatial GC Content Distribution")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")

        return self._finish(fig, output_path)

    def plot_motif_density(
        self,
        data: pd.DataFrame,
        output_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot motif density distribution.

        Args:
            data: DataFrame containing motif density data
            output_path: Optional path to save the plot
        """
        data = self._spatial_frame(data, "motif_count")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Bar plot
        sns.barplot(data=data, x="sequence_id", y="motif_count", ax=ax1)
        ax1.set_title("Motif Density by Sequence")
        ax1.set_xlabel("Sequence ID")
        ax1.set_ylabel("Motif Count")
        ax1.tick_params(axis="x", rotation=45)

        # Spatial plot
        gdf = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data.longitude, data.latitude),
            crs="EPSG:4326",
        )
        gdf.plot(column="motif_count", ax=ax2, legend=True)
        ax2.set_title("Spatial Motif Density")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")

        return self._finish(fig, output_path)

    def plot_coding_potential(
        self,
        data: pd.DataFrame,
        output_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot coding potential distribution.

        Args:
            data: DataFrame containing coding potential data
            output_path: Optional path to save the plot
        """
        data = self._spatial_frame(data, "coding_regions")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Bar plot
        sns.barplot(data=data, x="sequence_id", y="coding_regions", ax=ax1)
        ax1.set_title("Coding Regions by Sequence")
        ax1.set_xlabel("Sequence ID")
        ax1.set_ylabel("Number of Coding Regions")
        ax1.tick_params(axis="x", rotation=45)

        # Spatial plot
        gdf = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data.longitude, data.latitude),
            crs="EPSG:4326",
        )
        gdf.plot(column="coding_regions", ax=ax2, legend=True)
        ax2.set_title("Spatial Coding Potential")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")

        return self._finish(fig, output_path)

    def plot_sequence_alignment(
        self,
        alignment: Any,
        output_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot sequence alignment.

        Args:
            alignment: MultipleSeqAlignment object
            output_path: Optional path to save the plot
        """
        if alignment is None:
            raise ValueError("alignment must not be None")
        alignment_frame = pd.DataFrame(alignment)
        if alignment_frame.empty:
            raise ValueError("alignment must contain at least one sequence")
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(
            alignment_frame,
            ax=ax,
            cmap="YlOrRd",
            cbar_kws={"label": "Nucleotide"},
        )
        ax.set_title("Sequence Alignment")
        ax.set_xlabel("Position")
        ax.set_ylabel("Sequence ID")

        return self._finish(fig, output_path)
