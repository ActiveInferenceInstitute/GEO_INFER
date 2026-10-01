#!/usr/bin/env python3
"""
Base Analysis Module for GEO-INFER-SPACE

This module provides the foundational class for all analysis modules
in the GEO-INFER-SPACE framework, implementing standardized workflows
for data acquisition, processing, and analysis with H3 spatial indexing.
"""

import logging
import json
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any
import geopandas as gpd
from shapely.geometry import shape

from geo_infer_space.core import SpatialIndexingInterface

logger = logging.getLogger(__name__)


class BaseAnalysisModule(ABC):
    """
    Abstract Base Class for a GEO-INFER analysis module.

    Each subclass is responsible for a specific data domain (e.g., Zoning, Water Rights).
    The base class provides a standardized workflow:
    1.  Check for cached H3-processed data.
    2.  If not found, acquire raw data from source.
    3.  Process raw data through the package's H3 v4 spatial interface.
    4.  Cache the H3 data.
    5.  Load and perform final analysis on the H3 data.
    """

    def __init__(
        self,
        module_name: str,
        config_path: Path | None = None,
        h3_resolution: int = 8,
    ) -> None:
        """
        Initialize the base analysis module.

        Args:
            module_name: Name of the module for logging and identification
            config_path: Path to configuration file (optional)
            h3_resolution: H3 resolution for spatial indexing (default: 8)
        """
        self.module_name = module_name
        self.config_path = config_path
        self.h3_resolution = h3_resolution
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize H3 cache path
        self.h3_cache_path = (
            self.output_dir / f"{self.module_name}_h3_res{self.h3_resolution}.json"
        )

        # Initialize target hexagons (will be set by backend)
        self.target_hexagons: set[str] = set()

        self.logger = logging.getLogger(f"{__name__}.{module_name}")

        # Load configuration if provided
        self.config: dict[str, Any] = {}
        if config_path and config_path.exists():
            self._load_config()  # type: ignore[attr-defined]

        # Initialize spatial interface
        self.spatial = SpatialIndexingInterface()

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

    def process_to_h3(self, raw_data_path: Path) -> dict:
        """
        Processes a raw data file (e.g., GeoJSON, Shapefile) into an H3-indexed dictionary.

        This method uses the package's H3 v4 spatial interface.

        Args:
            raw_data_path: Path to the raw geospatial data file.

        Returns:
            A dictionary of H3-indexed data.
        """
        try:
            logger.info(
                f"[{self.module_name}] 🔄 Using direct H3 processing for {raw_data_path}"
            )

            start_time = time.time()
            h3_data = self._direct_h3_processing(raw_data_path)
            processing_time = time.time() - start_time

            if h3_data:
                logger.info(
                    f"[{self.module_name}] ✅ Direct H3 processing completed successfully in {processing_time:.1f}s"
                )
                logger.info(
                    f"[{self.module_name}] 🎯 Generated {len(h3_data)} H3 cells"
                )
            else:
                logger.error(f"[{self.module_name}] ❌ Direct H3 processing failed")

            return h3_data

        except Exception as e:
            logger.error(f"[{self.module_name}] ❌ H3 processing failed: {e}")
            return {}

    def _direct_h3_processing(self, raw_data_path: Path) -> dict:
        """
        Process raw data directly to H3 format using H3 v4 API.

        Args:
            raw_data_path: Path to the raw data file (GeoJSON)

        Returns:
            Dictionary with H3-indexed data
        """
        try:
            logger.info(
                f"[{self.module_name}] 🔄 Using direct H3 processing for {raw_data_path}"
            )
            logger.info(
                f"[{self.module_name}] 🔄 Starting direct H3 processing for {raw_data_path}"
            )

            # Load GeoJSON data
            logger.info(f"[{self.module_name}] 📁 Loading GeoJSON data...")
            gdf = gpd.read_file(raw_data_path)
            logger.info(
                f"[{self.module_name}] 📊 Loaded {len(gdf)} features for direct processing"
            )

            # Initialize H3 data storage
            h3_data: dict[str, Any] = {}
            processed_features = 0
            failed_features = 0

            logger.info(f"[{self.module_name}] 🔄 Processing features...")

            for idx, row in gdf.iterrows():
                try:
                    # Get the geometry - this should already be a shapely geometry from geopandas
                    geom = row.geometry

                    # Debug the geometry type
                    logger.debug(
                        f"[{self.module_name}] 🔍 Feature {idx} geometry type: {type(geom)}"
                    )

                    # Ensure we have a shapely geometry
                    if hasattr(geom, "geom_type"):
                        # It's already a shapely geometry
                        shapely_geom = geom
                        logger.debug(
                            f"[{self.module_name}] ✅ Feature {idx} has shapely geometry: {shapely_geom.geom_type}"
                        )
                    elif isinstance(geom, dict):
                        # Convert from GeoJSON dict to shapely
                        shapely_geom = shape(geom)
                        logger.debug(
                            f"[{self.module_name}] ✅ Feature {idx} converted from dict to shapely: {shapely_geom.geom_type}"
                        )
                    elif isinstance(geom, list):
                        # This is a list of coordinates - convert to shapely polygon
                        logger.debug(
                            f"[{self.module_name}] 🔄 Feature {idx} has coordinate list, converting to polygon"
                        )
                        # Create a polygon from the coordinate list
                        if len(geom) >= 3:
                            # Convert to shapely polygon
                            from shapely.geometry import Polygon

                            shapely_geom = Polygon(geom)
                            logger.debug(
                                f"[{self.module_name}] ✅ Feature {idx} converted from list to shapely: {shapely_geom.geom_type}"
                            )
                        else:
                            logger.warning(
                                f"[{self.module_name}] ⚠️ Invalid coordinate list (less than 3 points) for feature {idx}"
                            )
                            failed_features += 1
                            continue
                    else:
                        logger.warning(
                            f"[{self.module_name}] ⚠️ Unrecognized geometry type: {type(geom)} for feature {idx}"
                        )
                        failed_features += 1
                        continue

                    # Handle different geometry types
                    if shapely_geom.geom_type == "Point":
                        # Buffer points to create small polygons
                        shapely_geom = shapely_geom.buffer(0.001)
                        logger.info(
                            f"[{self.module_name}] 🔄 Buffered Point to Polygon for feature {idx}"
                        )

                    if shapely_geom.geom_type == "Polygon":
                        # Convert polygon to H3 cells using H3 v4 API
                        # H3 v4 expects a GeoJSON-like polygon format
                        # Convert shapely polygon to GeoJSON format
                        geojson_polygon = {
                            "type": "Polygon",
                            "coordinates": [list(shapely_geom.exterior.coords)],
                        }

                        try:
                            # Use unified spatial interface to convert polygon to cells
                            h3_cells = self.spatial.polygon_to_cells(
                                geojson_polygon, self.h3_resolution
                            )
                            logger.debug(
                                f"[{self.module_name}] ✅ Feature {idx} converted to {len(h3_cells)} H3 cells"
                            )

                            # Store the H3 cells for this feature
                            for h3_cell in h3_cells:
                                if h3_cell not in h3_data:
                                    h3_data[h3_cell] = {}

                                # Store the feature data in the H3 cell
                                if self.module_name not in h3_data[h3_cell]:
                                    h3_data[h3_cell][self.module_name] = []

                                h3_data[h3_cell][self.module_name].append(
                                    {
                                        "feature_id": idx,
                                        "properties": row.to_dict(),
                                        "geometry": shapely_geom.__geo_interface__,
                                    }
                                )

                            # Increment processed features counter
                            processed_features += 1

                        except Exception as e:
                            logger.warning(
                                f"[{self.module_name}] ⚠️ H3 conversion failed for feature {idx}: {e}"
                            )
                            failed_features += 1
                            continue

                    elif shapely_geom.geom_type == "MultiPolygon":
                        # Handle MultiPolygon by processing each polygon separately
                        logger.debug(
                            f"[{self.module_name}] 🔄 Processing MultiPolygon for feature {idx}"
                        )

                        for polygon_idx, polygon in enumerate(shapely_geom.geoms):
                            try:
                                geojson_polygon = {
                                    "type": "Polygon",
                                    "coordinates": [list(polygon.exterior.coords)],
                                }

                                h3_cells = self.spatial.polygon_to_cells(
                                    geojson_polygon, self.h3_resolution
                                )
                                logger.debug(
                                    f"[{self.module_name}] ✅ Feature {idx} polygon {polygon_idx} converted to {len(h3_cells)} H3 cells"
                                )

                                # Store the H3 cells for this feature
                                for h3_cell in h3_cells:
                                    if h3_cell not in h3_data:
                                        h3_data[h3_cell] = {}

                                    if self.module_name not in h3_data[h3_cell]:
                                        h3_data[h3_cell][self.module_name] = []

                                    h3_data[h3_cell][self.module_name].append(
                                        {
                                            "feature_id": f"{idx}_{polygon_idx}",
                                            "properties": row.to_dict(),
                                            "geometry": polygon.__geo_interface__,
                                        }
                                    )

                            except Exception as e:
                                logger.warning(
                                    f"[{self.module_name}] ⚠️ H3 conversion failed for feature {idx} polygon {polygon_idx}: {e}"
                                )
                                continue

                        processed_features += 1

                    else:
                        logger.warning(
                            f"[{self.module_name}] ⚠️ Unsupported geometry type: {shapely_geom.geom_type} for feature {idx}"
                        )
                        failed_features += 1

                except Exception as e:
                    logger.warning(
                        f"[{self.module_name}] ⚠️ Failed to process feature {idx}: {e}"
                    )
                    failed_features += 1

            logger.info(f"[{self.module_name}] ✅ Direct processing completed:")
            logger.info(
                f"[{self.module_name}]   📊 Processed features: {processed_features}"
            )
            logger.info(f"[{self.module_name}]   ❌ Failed features: {failed_features}")
            logger.info(f"[{self.module_name}]   🎯 H3 cells generated: {len(h3_data)}")

            if not h3_data:
                logger.error(f"[{self.module_name}] ❌ Direct H3 processing failed")
                return {}

            # Convert the h3_data to a serializable format before returning
            # This ensures that Shapely geometries are converted to GeoJSON
            serializable_h3_data: dict[str, Any] = {}
            for h3_cell, cell_data in h3_data.items():
                serializable_h3_data[h3_cell] = {}
                for module_name, module_data in cell_data.items():
                    serializable_h3_data[h3_cell][module_name] = []
                    for feature_data in module_data:
                        # Ensure geometry is in GeoJSON format
                        if "geometry" in feature_data:
                            if hasattr(feature_data["geometry"], "__geo_interface__"):
                                feature_data["geometry"] = feature_data[
                                    "geometry"
                                ].__geo_interface__
                        serializable_h3_data[h3_cell][module_name].append(feature_data)

            return serializable_h3_data

        except Exception as e:
            logger.error(f"[{self.module_name}] ❌ Error in direct H3 processing: {e}")
            return {}

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

    def _ensure_json_serializable(self, data: Any) -> Any:
        """
        Recursively convert any Shapely geometry objects to GeoJSON format for JSON serialization.

        Args:
            data: Any data structure that might contain Shapely objects

        Returns:
            The same data structure with all Shapely objects converted to GeoJSON
        """
        if isinstance(data, dict):
            return {
                key: self._ensure_json_serializable(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._ensure_json_serializable(item) for item in data]
        elif hasattr(data, "__geo_interface__"):
            # Convert Shapely geometry to GeoJSON
            return data.__geo_interface__
        else:
            return data

    def _validate_cache_file(self, cache_path: Path) -> bool:
        """
        Validate that a cache file is not corrupted.

        Args:
            cache_path: Path to the cache file

        Returns:
            True if the cache file is valid, False otherwise
        """
        try:
            if not cache_path.exists():
                return False

            # Check file size - if it's too small, it's probably corrupted
            file_size = cache_path.stat().st_size
            if file_size < 100:  # Less than 100 bytes is suspicious
                logger.warning(
                    f"[{self.module_name}] ⚠️ Cache file too small ({file_size} bytes), likely corrupted"
                )
                return False

            # Try to read the first few characters to check if it's valid JSON
            with open(cache_path) as f:
                first_chars = f.read(50)
                if not first_chars.strip().startswith("{"):
                    logger.warning(
                        f"[{self.module_name}] ⚠️ Cache file doesn't start with valid JSON"
                    )
                    return False

            # Try to parse the entire file
            with open(cache_path) as f:
                json.load(f)

            return True

        except Exception as e:
            logger.warning(f"[{self.module_name}] ⚠️ Cache validation failed: {e}")
            return False

    def run_analysis(self) -> dict:
        """
        Executes the full, standardized workflow for the module with real data tracking.

        This method orchestrates the caching, acquisition, and processing steps with detailed logging.

        Returns:
            The final H3-indexed analysis results for this module.
        """
        logger.info(f"[{self.module_name}] 🚀 Starting real data analysis workflow...")

        # Track data processing statistics
        analysis_stats: dict[str, Any] = {
            "cached_data_used": False,
            "raw_data_acquired": False,
            "h3_data_processed": 0,
            "final_analysis_completed": False,
            "errors": [],
        }

        try:
            # 1. Check for cached H3 data
            h3_data = {}
            raw_data_acquired = False

            if self.h3_cache_path.exists():
                logger.info(
                    f"[{self.module_name}] 📁 Found cached H3 data. Loading from {self.h3_cache_path}"
                )
                try:
                    if self._validate_cache_file(self.h3_cache_path):
                        with open(self.h3_cache_path) as f:
                            h3_data = json.load(f)

                        if h3_data and len(h3_data) > 0:
                            analysis_stats["cached_data_used"] = True
                            analysis_stats["h3_data_processed"] = len(h3_data)
                            logger.info(
                                f"[{self.module_name}] ✅ Loaded {len(h3_data)} hexagons from cache"
                            )
                        else:
                            logger.warning(
                                f"[{self.module_name}] ⚠️ Cached data is empty"
                            )
                            h3_data = {}
                    else:
                        logger.warning(
                            f"[{self.module_name}] ⚠️ Cached data file is corrupted. Deleting and regenerating..."
                        )
                        self.h3_cache_path.unlink()
                        h3_data = {}

                except Exception as e:
                    error_msg = f"Failed to load cached data: {e}"
                    analysis_stats["errors"].append(error_msg)
                    logger.error(f"[{self.module_name}] ❌ {error_msg}")
                    logger.info(
                        f"[{self.module_name}] 🗑️ Deleting corrupted cache file and regenerating..."
                    )

                    # Delete the corrupted cache file
                    try:
                        self.h3_cache_path.unlink()
                        logger.info(
                            f"[{self.module_name}] ✅ Deleted corrupted cache file"
                        )
                    except Exception as del_e:
                        logger.warning(
                            f"[{self.module_name}] ⚠️ Could not delete corrupted cache file: {del_e}"
                        )

                    h3_data = {}
            else:
                logger.info(
                    f"[{self.module_name}] 📁 No cached H3 data found. Starting real data acquisition..."
                )

            # 2. If no valid cached data, acquire and process raw data
            if not h3_data or len(h3_data) == 0:
                logger.info(f"[{self.module_name}] 🔍 Acquiring raw data...")

                # Acquire raw data with detailed tracking
                try:
                    raw_data_path = self.acquire_raw_data()

                    if raw_data_path and raw_data_path.exists():
                        # Check if file has real content
                        file_size = raw_data_path.stat().st_size
                        if file_size > 100:  # More than just headers
                            analysis_stats["raw_data_acquired"] = True
                            raw_data_acquired = True
                            logger.info(
                                f"[{self.module_name}] ✅ Raw data acquired: {raw_data_path} ({file_size} bytes)"
                            )
                        else:
                            logger.warning(
                                f"[{self.module_name}] ⚠️ Raw data file too small: {raw_data_path} ({file_size} bytes)"
                            )
                    else:
                        logger.warning(
                            f"[{self.module_name}] ⚠️ No raw data path returned or file doesn't exist"
                        )

                except Exception as e:
                    error_msg = f"Raw data acquisition failed: {e}"
                    analysis_stats["errors"].append(error_msg)
                    logger.error(f"[{self.module_name}] ❌ {error_msg}")
                    raw_data_path = None

                # 3. Process raw data to H3 if acquisition was successful
                if raw_data_acquired and raw_data_path and raw_data_path.exists():
                    try:
                        logger.info(
                            f"[{self.module_name}] 🔄 Processing raw data to H3..."
                        )
                        h3_data = self.process_to_h3(raw_data_path)

                        if h3_data and len(h3_data) > 0:
                            analysis_stats["h3_data_processed"] = len(h3_data)
                            logger.info(
                                f"[{self.module_name}] ✅ Processed {len(h3_data)} hexagons to H3"
                            )
                        else:
                            logger.warning(
                                f"[{self.module_name}] ⚠️ H3 processing returned empty data"
                            )

                    except Exception as e:
                        error_msg = f"H3 processing failed: {e}"
                        analysis_stats["errors"].append(error_msg)
                        logger.error(f"[{self.module_name}] ❌ {error_msg}")
                        h3_data = {}

                # 4. Cache the H3 data if processing was successful
                if h3_data and len(h3_data) > 0:
                    try:
                        logger.info(
                            f"[{self.module_name}] 💾 Caching new H3 data to {self.h3_cache_path}"
                        )

                        # Ensure all data is JSON serializable by converting any remaining Shapely objects
                        serializable_h3_data = self._ensure_json_serializable(h3_data)

                        with open(self.h3_cache_path, "w") as f:
                            json.dump(serializable_h3_data, f)
                        logger.info(
                            f"[{self.module_name}] ✅ H3 data cached successfully"
                        )
                    except Exception as e:
                        error_msg = f"Failed to cache H3 data: {e}"
                        analysis_stats["errors"].append(error_msg)
                        logger.error(f"[{self.module_name}] ❌ {error_msg}")

            # 5. Run the final analysis on the (now available) H3 data
            if h3_data and len(h3_data) > 0:
                try:
                    logger.info(
                        f"[{self.module_name}] 🔬 Running final analysis on {len(h3_data)} hexagons..."
                    )
                    final_results = self.run_final_analysis(h3_data)

                    if final_results and len(final_results) > 0:
                        analysis_stats["final_analysis_completed"] = True
                        logger.info(
                            f"[{self.module_name}] ✅ Final analysis completed: {len(final_results)} results"
                        )

                        # Log sample of final results
                        sample_keys = list(final_results.keys())[:3]
                        logger.info(
                            f"[{self.module_name}] 📊 Sample final results keys: {sample_keys}"
                        )
                        for key in sample_keys:
                            sample_data = final_results[key]
                            logger.info(
                                f"[{self.module_name}] 📊 Sample result for {key}: {sample_data}"
                            )
                    else:
                        logger.warning(
                            f"[{self.module_name}] ⚠️ Final analysis returned empty results"
                        )
                        final_results = {}

                except Exception as e:
                    error_msg = f"Final analysis failed: {e}"
                    analysis_stats["errors"].append(error_msg)
                    logger.error(f"[{self.module_name}] ❌ {error_msg}")
                    final_results = {}
            else:
                logger.warning(
                    f"[{self.module_name}] ⚠️ No H3 data available for final analysis"
                )
                final_results = {}

            # Log comprehensive analysis summary
            logger.info(f"[{self.module_name}] 📊 ANALYSIS SUMMARY:")
            logger.info(
                f"[{self.module_name}]   - Cached data used: {analysis_stats['cached_data_used']}"
            )
            logger.info(
                f"[{self.module_name}]   - Raw data acquired: {analysis_stats['raw_data_acquired']}"
            )
            logger.info(
                f"[{self.module_name}]   - H3 cells processed: {analysis_stats['h3_data_processed']}"
            )
            logger.info(
                f"[{self.module_name}]   - Final analysis completed: {analysis_stats['final_analysis_completed']}"
            )
            logger.info(
                f"[{self.module_name}]   - Final results: {len(final_results)} hexagons"
            )
            if analysis_stats["errors"]:
                logger.info(
                    f"[{self.module_name}]   - Errors: {len(analysis_stats['errors'])}"
                )
                for error in analysis_stats["errors"]:
                    logger.info(f"[{self.module_name}]     ❌ {error}")

            if len(final_results) == 0:
                logger.error(
                    f"[{self.module_name}] ❌ CRITICAL: No final results generated!"
                )
                logger.error(
                    f"[{self.module_name}] 🔍 This indicates a fundamental processing failure."
                )
                logger.error(
                    f"[{self.module_name}] 🔍 Check data sources, file paths, and analysis logic."
                )

            return final_results

        except Exception as e:
            error_msg = f"Module analysis workflow failed: {e}"
            analysis_stats["errors"].append(error_msg)
            logger.error(f"[{self.module_name}] ❌ {error_msg}")
            return {}
