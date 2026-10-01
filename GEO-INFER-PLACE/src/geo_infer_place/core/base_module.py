#!/usr/bin/env python3
"""
Base Module for Cascadian Agricultural Land Analysis

This module defines the abstract base class for all specialized analysis modules
in the Cascadian framework. It enforces a standardized workflow for data
acquisition, caching, H3 processing, and analysis.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import json
import os

# A forward declaration for type hinting the backend without circular imports
from typing import TYPE_CHECKING

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
    3.  Process raw data into H3 using the backend's OSC H3 loader.
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

    def process_to_h3(self, raw_data_path: Path) -> dict:
        """
        Processes a raw data file (e.g., GeoJSON, Shapefile) into an H3-indexed dictionary.

        This method uses the backend's shared H3DataLoader (from GEO-INFER-SPACE).

        Args:
            raw_data_path: Path to the raw geospatial data file.

        Returns:
            A dictionary of H3-indexed data.
        """
        if not self.backend.h3_loader:
            logger.error(
                f"[{self.module_name}] H3 loader not available. Cannot process data."
            )
            return {}

        try:  # type: ignore[unreachable]
            # Define a temporary output path for the H3 conversion
            h3_output_path = self.data_dir / f"temp_{self.module_name}_h3.geojson"

            logger.info(
                f"[{self.module_name}] Using H3 loader to process {raw_data_path} -> {h3_output_path}"
            )

            h3_data = self.backend.h3_loader.load_data(
                input_file=str(raw_data_path), output_file=str(h3_output_path)
            )

            # After processing, we would typically load the h3_output_path file
            # For now, this part of the logic is incomplete in the original file
            # We will assume success returns the data directly or we load the file
            if h3_output_path.exists():
                with open(h3_output_path) as f:
                    h3_data = json.load(f)
                # Clean up the temporary file
                os.remove(h3_output_path)
            else:
                # This case needs to be handled based on what h3_loader.load_data returns
                h3_data = {}

        except Exception as e:
            logger.error(
                f"[{self.module_name}] Failed to process data to H3: {e}", exc_info=True
            )
            h3_data = {}

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
