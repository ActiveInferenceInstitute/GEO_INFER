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
import re
import os
import tempfile
import h3
from copy import deepcopy
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any
import geopandas as gpd

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
        h3_resolution: int | None = None,
        *,
        output_dir: Path | None = None,
    ) -> None:
        """
        Initialize the base analysis module.

        Args:
            module_name: Name of the module for logging and identification
            config_path: Path to configuration file (optional)
            h3_resolution: H3 resolution for spatial indexing (default: 8)
        """
        if not isinstance(module_name, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", module_name
        ):
            raise ValueError("module_name must be a simple module identifier")
        self.module_name = module_name
        self.config_path = Path(config_path) if config_path is not None else None
        self.config: dict[str, Any] = {}
        if self.config_path is not None:
            self._load_config()
        spatial = self.config.get("spatial", {})
        if not isinstance(spatial, dict):
            raise ValueError("spatial configuration must be a mapping")
        configured_resolution = spatial.get("h3_resolution", 8)
        self.h3_resolution = (
            configured_resolution if h3_resolution is None else h3_resolution
        )
        if (
            isinstance(self.h3_resolution, bool)
            or not isinstance(self.h3_resolution, int)
            or not 0 <= self.h3_resolution <= 15
        ):
            raise ValueError("h3_resolution must be an integer between 0 and 15")
        self.output_dir = Path(output_dir) if output_dir is not None else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h3_cache_path = (
            self.output_dir / f"{module_name}_h3_res{self.h3_resolution}.json"
        )
        self.target_hexagons: set[str] = set()
        self.logger = logging.getLogger(f"{__name__}.{module_name}")
        self.spatial = SpatialIndexingInterface()

    def _load_config(self) -> None:
        """Read an explicit JSON/YAML mapping; malformed or missing files raise."""
        if self.config_path is None:
            return
        with self.config_path.open(encoding="utf-8") as stream:
            loaded = (
                json.load(stream)
                if self.config_path.suffix.lower() == ".json"
                else yaml.safe_load(stream)
            )
        if not isinstance(loaded, dict):
            raise ValueError("Analysis configuration must be a mapping")
        self.config = deepcopy(loaded)

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
        """Convert a real vector file to H3; read/CRS/topology failures propagate."""
        started = time.monotonic()
        result = self._direct_h3_processing(raw_data_path)
        logger.info(
            "[%s] Processed %d H3 cells in %.3fs",
            self.module_name,
            len(result),
            time.monotonic() - started,
        )
        return result

    def _direct_h3_processing(self, raw_data_path: Path) -> dict:
        """Index WGS84 points and polygon coverage, preserving holes and identity.

        Projected files are reprojected explicitly. Point observations occupy
        their containing H3 cell, without an invented angular buffer. Polygon
        coverage uses H3's center-containment rule; small polygons may contain
        no cell centroids. Unsupported, empty or invalid geometries raise.
        """
        from geo_infer_space.analytics.vector import _reproject

        frame = gpd.read_file(raw_data_path)
        if frame.crs is None:
            raise ValueError(
                "Vector source must declare its coordinate reference system"
            )
        if any(
            geometry is None or geometry.is_empty or not geometry.is_valid
            for geometry in frame.geometry
        ):
            raise ValueError("Vector source has an empty or invalid geometry")
        frame = _reproject(frame, "EPSG:4326")
        result: dict[str, Any] = {}
        for feature_id, row in frame.iterrows():
            geometry = row.geometry
            if geometry is None or geometry.is_empty or not geometry.is_valid:
                raise ValueError(
                    f"Feature {feature_id!r} has an empty or invalid geometry"
                )
            if geometry.geom_type == "Point":
                cells = [
                    self.spatial.latlng_to_cell(
                        geometry.y, geometry.x, self.h3_resolution
                    )
                ]
            elif geometry.geom_type in {"Polygon", "MultiPolygon"}:
                cells = self.spatial.polygon_to_cells(
                    geometry.__geo_interface__, self.h3_resolution
                )
            else:
                raise ValueError(f"Unsupported vector geometry: {geometry.geom_type}")
            properties = row.drop(labels=frame.geometry.name).to_dict()
            for cell in cells:
                result.setdefault(cell, {}).setdefault(self.module_name, []).append(
                    {
                        "feature_id": feature_id,
                        "properties": deepcopy(properties),
                        "geometry": deepcopy(geometry.__geo_interface__),
                    }
                )
        return result

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
        """Validate parseable H3 cache identities rather than arbitrary byte size.

        Cache files remain local derived artifacts. Their names bind the module
        and resolution; upstream freshness is the caller's explicit cache policy.
        """
        try:
            with cache_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            return (
                isinstance(payload, dict)
                and bool(payload)
                and all(
                    isinstance(cell, str)
                    and h3.is_valid_cell(cell)
                    and h3.int_to_str(h3.str_to_int(cell)) == cell
                    and h3.get_resolution(cell) == self.h3_resolution
                    and isinstance(data, dict)
                    and isinstance(data.get(self.module_name), list)
                    for cell, data in payload.items()
                )
            )
        except (OSError, ValueError, TypeError):
            return False

    def run_analysis(self, *, use_cache: bool = True) -> dict:
        """Run acquisition, native processing, atomic caching and domain analysis.

        A valid local cache can be reused. Invalid cache bytes are retained until
        successful regeneration replaces them. Acquisition, conversion, cache
        persistence and final-analysis failures propagate without empty sentinels.
        """
        if not isinstance(use_cache, bool):
            raise TypeError("use_cache must be a boolean")
        if use_cache and self._validate_cache_file(self.h3_cache_path):
            with self.h3_cache_path.open(encoding="utf-8") as stream:
                h3_data = json.load(stream)
            logger.info(
                "[%s] Reusing %d cached H3 cells", self.module_name, len(h3_data)
            )
        else:
            raw_path = Path(self.acquire_raw_data())
            if not raw_path.is_file():
                raise FileNotFoundError(f"Raw vector source does not exist: {raw_path}")
            h3_data = self.process_to_h3(raw_path)
            if not isinstance(h3_data, dict) or not h3_data:
                raise ValueError(
                    "Vector source generated no H3 cells at the selected resolution"
                )
            serializable = self._ensure_json_serializable(h3_data)
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.h3_cache_path.parent,
                    prefix=f".{self.h3_cache_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    json.dump(serializable, stream, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.h3_cache_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        result = self.run_final_analysis(h3_data)
        if not isinstance(result, dict):
            raise TypeError("Final analysis must return a dictionary")
        logger.info(
            "[%s] Completed domain analysis with %d results",
            self.module_name,
            len(result),
        )
        return result
