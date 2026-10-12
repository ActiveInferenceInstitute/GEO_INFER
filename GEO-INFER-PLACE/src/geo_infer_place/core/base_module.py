#!/usr/bin/env python3
"""
Base Module for Cascadian Agricultural Land Analysis

This module defines the abstract base class for all specialized analysis modules
in the Cascadian framework. It enforces a standardized workflow for data
acquisition, caching, H3 processing, and analysis.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import h3

if TYPE_CHECKING:
    from .unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)


class BaseAnalysisModule(ABC):
    """
    Abstract Base Class for a GEO-INFER analysis module.

    Each subclass is responsible for a specific data domain (e.g., Zoning, Water Rights).
    The base class provides a standardized workflow:
    1.  Check for cached, H3-processed data.
    2.  If not found, acquire raw data from source.
    3.  Index the raw vector data onto H3 cells (``process_to_h3``).
    4.  Cache the H3 data.
    5.  Load and perform final analysis on the H3 data.
    """

    def __init__(self, backend: "CascadianAgriculturalH3Backend", module_name: str):
        """
        Initialize the module.

        Args:
            backend: A reference to the main CascadianAgriculturalH3Backend instance.
            module_name: The name of the module (e.g., 'zoning').
        """
        self.backend = backend
        self.module_name = module_name
        self.resolution = backend.resolution
        self.target_hexagons = backend.target_hexagons

        # Define standardized data paths
        self.data_dir = Path(self.backend.base_data_dir) / self.module_name
        self.data_dir.mkdir(exist_ok=True)
        self.h3_cache_path = (
            self.data_dir / f"{self.module_name}_h3_res{self.resolution}.json"
        )

    @abstractmethod
    def acquire_raw_data(self) -> Path:
        """
        Acquires raw data from its source (API, file download, etc.).

        This method must implement caching for the raw data file itself, i.e.,
        it should check if the raw file exists before re-downloading it.

        Returns:
            The file path to the acquired raw data.
        """
        raise RuntimeError(
            "BaseAnalysisModule.acquire_raw_data requires a concrete module implementation"
        )

    def process_to_h3(self, raw_data_path: Path) -> dict[str, list[dict[str, Any]]]:
        """
        Index a vector data file (GeoJSON, Shapefile, ...) onto H3 cells.

        Polygonal features cover the cells whose centroids fall inside them
        (``h3.geo_to_cells``); point features map to their containing cell.
        Other geometry types are skipped. Each cell collects the JSON-safe
        property dicts of every feature that covers it.

        Args:
            raw_data_path: Path to the raw geospatial data file.

        Returns:
            Mapping of H3 cell index to the list of covering feature properties.
        """
        gdf = gpd.read_file(raw_data_path)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        geometry_column = gdf.geometry.name
        h3_data: dict[str, list[dict[str, Any]]] = {}
        skipped = 0
        for _, row in gdf.iterrows():
            geom = row[geometry_column]
            if geom is None or geom.is_empty:
                skipped += 1
                continue
            if geom.geom_type == "Point":
                cells = [h3.latlng_to_cell(geom.y, geom.x, self.resolution)]
            elif geom.geom_type in ("Polygon", "MultiPolygon"):
                cells = h3.geo_to_cells(geom.__geo_interface__, self.resolution)
            else:
                skipped += 1
                continue
            properties = json.loads(
                row.drop(labels=[geometry_column]).to_json(default_handler=str)
            )
            for cell in cells:
                h3_data.setdefault(cell, []).append(properties)

        logger.info(
            "[%s] Indexed %d features onto %d H3 cells at resolution %d (%d skipped)",
            self.module_name,
            len(gdf) - skipped,
            len(h3_data),
            self.resolution,
            skipped,
        )
        return h3_data

    @abstractmethod
    def run_final_analysis(self, h3_data: dict) -> dict:
        """
        Performs the final, module-specific analysis on H3-indexed data.

        Args:
            h3_data: The H3-indexed data, loaded from cache or freshly processed.

        Returns:
            A dictionary of H3 hexagons with the final analysis results.
        """
        raise RuntimeError(
            "BaseAnalysisModule.run_final_analysis requires a concrete module implementation"
        )

    def run_analysis(self) -> dict:
        """
        Executes the full, standardized workflow for the module.

        This method orchestrates the caching, acquisition, and processing steps.

        Returns:
            The final H3-indexed analysis results for this module.
        """
        # 1. Check for cached H3 data
        if self.h3_cache_path.exists():
            logger.info(
                f"[{self.module_name}] Found cached H3 data. Loading from {self.h3_cache_path}"
            )
            with open(self.h3_cache_path) as f:
                h3_data = json.load(f)
        else:
            logger.info(
                f"[{self.module_name}] No cached H3 data found. Starting data acquisition..."
            )
            # 2. Acquire raw data
            raw_data_path = self.acquire_raw_data()

            if not raw_data_path or not raw_data_path.exists():
                logger.error(
                    f"[{self.module_name}] Raw data acquisition failed. Aborting module processing."
                )
                return {}

            # 3. Process raw data to H3
            h3_data = self.process_to_h3(raw_data_path)

            # 4. Cache the H3 data
            if h3_data:
                logger.info(
                    f"[{self.module_name}] Caching new H3 data to {self.h3_cache_path}"
                )
                with open(self.h3_cache_path, "w") as f:
                    json.dump(h3_data, f)

        # 5. Run the final analysis on the (now available) H3 data
        logger.info(
            f"[{self.module_name}] Running final analysis on {len(h3_data)} hexagons."
        )
        final_results = self.run_final_analysis(h3_data)

        return final_results
