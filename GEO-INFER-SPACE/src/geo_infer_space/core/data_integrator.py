#!/usr/bin/env python3
"""
General Data Integrator Module

This module provides capabilities for integrating data from multiple sources
into a unified geospatial dataset.
"""

import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform

logger = logging.getLogger(__name__)


def _reproject(frame: gpd.GeoDataFrame, target_crs: CRS) -> gpd.GeoDataFrame:
    """Reproject with scalar Shapely transforms for pyproj 3.7 compatibility."""
    if frame.crs == target_crs:
        return frame
    transformer = Transformer.from_crs(frame.crs, target_crs, always_xy=True)
    result = frame.copy()
    result["geometry"] = result.geometry.map(
        lambda geometry: shapely_transform(transformer.transform, geometry)
    )
    return result.set_crs(target_crs, allow_override=True)


class DataIntegrator:
    """
    Integrates data from multiple sources into a unified geospatial dataset.
    """

    def __init__(self, sources: list[dict[str, str]]) -> None:
        """
        Initialize the integrator with data sources.

        Args:
            sources: List of data sources with 'name' and 'path'
        """
        self.sources = sources
        self.integrated_data: gpd.GeoDataFrame = gpd.GeoDataFrame()

    def integrate_data(self) -> gpd.GeoDataFrame:
        """
        Integrate all data sources.

        Returns:
            Integrated GeoDataFrame
        """
        dataframes = []
        target_crs = None

        for source in self.sources:
            try:
                if source["path"].endswith(".geojson"):
                    df = gpd.read_file(source["path"])
                elif source["path"].endswith(".csv"):
                    df = pd.read_csv(source["path"])
                    if "geometry" in df.columns:
                        df = gpd.GeoDataFrame(
                            df, geometry=gpd.GeoSeries.from_wkt(df["geometry"])
                        )
                elif source["path"].endswith(".shp"):
                    df = gpd.read_file(source["path"])
                else:
                    continue

                # Set target CRS from first valid source
                if target_crs is None and df.crs is not None:
                    target_crs = df.crs

                if df.crs is None and target_crs is not None and "geometry" in df:
                    df = df.set_crs(target_crs)

                # Harmonize CRS if needed
                if (
                    df.crs is not None
                    and target_crs is not None
                    and df.crs != target_crs
                ):
                    df = _reproject(df, target_crs)
                    logger.info(
                        f"Transformed {source['name']} from {df.crs} to {target_crs}"
                    )

                df["source"] = source["name"]
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to load {source['name']}: {e}")

        if dataframes:
            self.integrated_data = gpd.GeoDataFrame(
                pd.concat(dataframes, ignore_index=True), crs=target_crs
            )
        return self.integrated_data

    def export_integrated_data(
        self, output_path: Path, format: str = "geojson"
    ) -> None:
        """
        Export integrated data.

        Args:
            output_path: Path to save file
            format: Output format ('geojson', 'shp', etc.)
        """
        if not self.integrated_data.empty:
            self.integrated_data.to_file(output_path, driver=format.upper())
            logger.info(f"Exported integrated data to {output_path}")
        else:
            logger.warning("No data to export")
