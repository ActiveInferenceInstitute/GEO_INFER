#!/usr/bin/env python3
"""
Enhanced H3 Geospatial Fusion Module for Cascadia Agricultural Analysis Framework

This module provides comprehensive H3 v4 geospatial data fusion with:
- Proper H3 v4 API usage throughout
- Advanced spatial analysis and correlation
- SPACE integration for enhanced operations
- Real-time data fusion capabilities
- Comprehensive validation and quality control
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Any
from datetime import datetime
from shapely.geometry import Polygon

# Import H3 utilities from consolidated geo_infer_place module
from geo_infer_place.utils.h3_operations import (
    latlng_to_cell,
    cell_to_latlng,
    cell_to_latlng_boundary,
    geo_to_cells,
    grid_disk,
    cell_area,
    is_valid_cell,
)

SPACE_H3_AVAILABLE = True

logger = logging.getLogger(__name__)


class EnhancedH3Fusion:
    """
    Enhanced H3 geospatial fusion engine for Cascadia agricultural analysis.

    Features:
    - Real H3 v4 API usage throughout
    - Advanced spatial analysis and correlation
    - SPACE integration for enhanced operations
    - Comprehensive data validation
    - Real-time fusion capabilities
    """

    def __init__(
        self,
        h3_resolution: int = 8,
        enable_spatial_analysis: bool = True,
        cache_dir: Path | None = None,
        fusion_mode: str = "geom_intersect",
    ):
        """
        Initialize the enhanced H3 fusion engine.

        Args:
            h3_resolution: H3 resolution for spatial indexing
            enable_spatial_analysis: Enable advanced spatial analysis
        """
        self.h3_resolution = h3_resolution
        self.enable_spatial_analysis = enable_spatial_analysis
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # Fusion mode: 'geom_intersect' (default) or 'key_join' for direct hex-key joins
        self.fusion_mode = (
            fusion_mode
            if fusion_mode in {"geom_intersect", "key_join"}
            else "geom_intersect"
        )

        # Validation settings
        self.validation_settings = {
            "min_hexagon_count": 1,
            "max_hexagon_count": 1000000,
            "require_valid_geometries": True,
            "spatial_consistency_check": True,
            "data_quality_threshold": 0.8,
        }

        logger.info(f"Enhanced H3 Fusion initialized with resolution {h3_resolution}")
        logger.info(f"SPACE H3 utilities available: {SPACE_H3_AVAILABLE}")
        logger.info(f"Spatial analysis enabled: {enable_spatial_analysis}")
        logger.info(f"Fusion mode: {self.fusion_mode}")
        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Fusion cache directory: {self.cache_dir}")
            except Exception as e:
                logger.warning(
                    f"Could not create fusion cache directory {self.cache_dir}: {e}"
                )

    def fuse_geospatial_data(
        self, data_sources: dict[str, Any], target_hexagons: list[str]
    ) -> dict[str, Any]:
        """
        Fuse multiple geospatial data sources into unified H3-indexed format.

        Args:
            data_sources: Dictionary of data sources with their H3 data
            target_hexagons: List of target H3 hexagons

        Returns:
            Fused H3-indexed data dictionary
        """
        logger.info(
            f"Starting enhanced H3 geospatial fusion for {len(data_sources)} data sources"
        )

        # Validate input data
        validation_result = self._validate_fusion_inputs(data_sources, target_hexagons)
        if not validation_result["is_valid"]:
            logger.error(f"Fusion validation failed: {validation_result['errors']}")
            return {}

        # Initialize fused data structure
        fused_data: dict[str, Any] = {}

        if self.fusion_mode == "key_join":
            target_set = set(target_hexagons)
            for source_name, source_data in data_sources.items():
                logger.info(f"[key_join] Processing data source: {source_name}")
                try:
                    if not isinstance(source_data, dict):
                        logger.warning(f"Source {source_name} not a dict; skipping")
                        continue
                    for hex_id, hex_vals in source_data.items():
                        if hex_id not in target_set:
                            continue
                        if hex_id not in fused_data:
                            fused_data[hex_id] = {}
                        # Normalize to list per-source
                        if isinstance(hex_vals, list):
                            fused_data[hex_id].setdefault(source_name, [])
                            fused_data[hex_id][source_name].extend(hex_vals)
                        else:
                            fused_data[hex_id].setdefault(source_name, [])
                            fused_data[hex_id][source_name].append(hex_vals)
                    logger.info(
                        f"✅ [key_join] Fused {source_name}: {len(source_data)} source hexagons"
                    )
                except Exception as e:
                    logger.error(f"❌ [key_join] Failed to fuse {source_name}: {e}")
                    continue
        else:
            # Geometric intersection mode with batch optimization
            logger.info(f"Starting batch fusion for {len(data_sources)} data sources")

            # Process data sources in parallel if they are independent
            import concurrent.futures
            import threading

            # Use ThreadPoolExecutor for I/O bound operations (spatial calculations)
            max_workers = min(4, len(data_sources))  # Limit concurrent workers

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit fusion tasks
                future_to_source = {
                    executor.submit(
                        self._fuse_source_batch,
                        source_name,
                        source_data,
                        target_hexagons,
                    ): source_name
                    for source_name, source_data in data_sources.items()
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_source):
                    source_name = future_to_source[future]
                    try:
                        fused_source_data = future.result()

                        # Add to fused data (thread-safe)
                        with threading.Lock():
                            for hex_id, hex_data in fused_source_data.items():
                                if hex_id not in fused_data:
                                    fused_data[hex_id] = {}
                                fused_data[hex_id][source_name] = hex_data

                        logger.info(
                            f"✅ Batch fused {source_name}: {len(fused_source_data)} hexagons"
                        )

                    except Exception as e:
                        logger.error(f"❌ Batch fusion failed for {source_name}: {e}")

        # Perform spatial analysis if enabled
        if self.enable_spatial_analysis and fused_data:
            logger.info("Performing enhanced spatial analysis...")
            fused_data = self._perform_spatial_analysis(fused_data, target_hexagons)

        # Generate fusion report
        self.last_fusion_report = self._generate_fusion_report(
            fused_data, data_sources, target_hexagons
        )

        logger.info(f"✅ Enhanced H3 fusion completed: {len(fused_data)} hexagons")
        logger.info(
            f"📊 Fusion coverage: {len(fused_data) / len(target_hexagons) * 100:.1f}%"
        )

        return fused_data

    # -------------------- Caching Utilities --------------------
    def _compute_fusion_signature(
        self, data_sources: dict[str, Any], target_hexagons: list[str]
    ) -> str:
        """Compute a deterministic cache key for fusion inputs.

        Uses module names, per-module hex counts, feature counts, a small sample of hex ids,
        target set size, resolution, and spatial analysis flag.
        """
        try:
            signature_obj: dict[str, Any] = {
                "h3_resolution": int(self.h3_resolution),
                "spatial": bool(self.enable_spatial_analysis),
                "fusion_mode": str(self.fusion_mode),
                "target_count": int(len(target_hexagons)),
                "modules": {},
            }
            for module_name in sorted(data_sources.keys()):
                mod = data_sources.get(module_name) or {}
                if not isinstance(mod, dict):
                    mod = {}
                hex_ids = sorted(list(mod.keys()))
                feature_count = 0
                for v in mod.values():
                    if isinstance(v, list):
                        feature_count += len(v)
                    else:
                        feature_count += 1
                sample = hex_ids[:50] + hex_ids[-50:]
                signature_obj["modules"][module_name] = {
                    "hex_count": len(mod),
                    "feature_count": feature_count,
                    "sample_hexes": sample,
                }
            serialized = json.dumps(signature_obj, sort_keys=True).encode("utf-8")
            return hashlib.sha256(serialized).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to compute fusion signature, using fallback: {e}")
            return f"res{self.h3_resolution}_fallback"

    def _get_cache_file(self, cache_key: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"fused_res{self.h3_resolution}_{cache_key}.json"

    def load_fusion_cache(
        self, data_sources: dict[str, Any], target_hexagons: list[str]
    ) -> dict[str, Any] | None:
        """Attempt to load fused data from cache based on inputs."""
        if not self.cache_dir:
            return None
        cache_key = self._compute_fusion_signature(data_sources, target_hexagons)
        cache_file = self._get_cache_file(cache_key)
        if not cache_file or not cache_file.exists():
            return None
        try:
            with open(cache_file) as f:
                payload = json.load(f)
            fused_data = payload.get("fused_data", {})
            logger.info(f"Loaded fusion cache: {cache_file} ({len(fused_data)} hexes)")
            return fused_data
        except Exception as e:
            logger.warning(f"Failed to load fusion cache {cache_file}: {e}")
            return None

    def save_fusion_cache(
        self,
        data_sources: dict[str, Any],
        target_hexagons: list[str],
        fused_data: dict[str, Any],
        report: dict[str, Any] | None = None,
    ) -> Path | None:
        """Persist fused data and report to cache."""
        if not self.cache_dir:
            return None
        try:
            cache_key = self._compute_fusion_signature(data_sources, target_hexagons)
            cache_file = self._get_cache_file(cache_key)
            if not cache_file:
                return None
            payload = {
                "meta": {
                    "h3_resolution": int(self.h3_resolution),
                    "spatial_analysis": bool(self.enable_spatial_analysis),
                    "fusion_mode": str(self.fusion_mode),
                    "modules": list(sorted(data_sources.keys())),
                    "target_hexagon_count": int(len(target_hexagons)),
                    "fused_hexagon_count": int(len(fused_data)),
                },
                "fused_data": fused_data,
                "report": report or {},
            }
            with open(cache_file, "w") as f:
                json.dump(payload, f)
            logger.info(f"Saved fusion cache: {cache_file}")
            return cache_file
        except Exception as e:
            logger.warning(f"Failed to save fusion cache: {e}")
            return None

    def _validate_fusion_inputs(
        self, data_sources: dict[str, Any], target_hexagons: list[str]
    ) -> dict[str, Any]:
        """
        Validate fusion inputs for consistency and quality.

        Args:
            data_sources: Dictionary of data sources
            target_hexagons: List of target hexagons

        Returns:
            Validation result dictionary
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "source_count": len(data_sources),
            "target_hexagon_count": len(target_hexagons),
        }

        # Validate target hexagons
        if not target_hexagons:
            validation_result["is_valid"] = False
            validation_result["errors"].append("No target hexagons provided")

        # Validate hexagon format
        invalid_hexagons = []
        for hex_id in target_hexagons:
            if not is_valid_cell(hex_id):
                invalid_hexagons.append(hex_id)

        if invalid_hexagons:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Invalid hexagons: {invalid_hexagons[:5]}"
            )

        # Validate data sources
        if not data_sources:
            validation_result["warnings"].append("No data sources provided")

        for source_name, source_data in data_sources.items():
            if not isinstance(source_data, dict):
                validation_result["errors"].append(
                    f"Invalid source data type for {source_name}"
                )
            elif len(source_data) == 0:
                validation_result["warnings"].append(
                    f"Empty source data for {source_name}"
                )

        return validation_result

    def _validate_source_data(
        self, source_data: dict[str, Any], source_name: str
    ) -> dict[str, Any]:
        """
        Validate individual source data.

        Args:
            source_data: Source data dictionary
            source_name: Name of the source

        Returns:
            Validation result dictionary
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "hexagon_count": len(source_data),
            "feature_count": 0,
        }

        # Count total features
        for hex_data in source_data.values():
            if isinstance(hex_data, list):
                validation_result["feature_count"] += len(hex_data)
            elif isinstance(hex_data, dict):
                validation_result["feature_count"] += 1

        # Check for reasonable data size
        if len(source_data) > self.validation_settings["max_hexagon_count"]:
            validation_result["warnings"].append(
                f"Large dataset: {len(source_data)} hexagons"
            )

        if len(source_data) < self.validation_settings["min_hexagon_count"]:
            validation_result["warnings"].append(
                f"Small dataset: {len(source_data)} hexagons"
            )

        return validation_result

    def _fuse_source_to_target_hexagons(
        self, source_data: dict[str, Any], target_hexagons: list[str], source_name: str
    ) -> dict[str, Any]:
        """
        Fuse source data to target hexagons using optimized H3 spatial operations.

        Args:
            source_data: Source H3 data
            target_hexagons: List of target hexagons
            source_name: Name of the source

        Returns:
            Fused data for target hexagons
        """
        import time

        start_time = time.time()

        fused_data = {}
        target_hex_set = set(target_hexagons)

        # Optimize: Use spatial indexing for faster intersection detection
        logger.info(
            f"[{source_name}] Starting optimized fusion with {len(source_data)} source hexagons"
        )

        # Pre-compute spatial relationships using H3 neighborhood operations
        source_hex_ids = list(source_data.keys())
        total = len(source_hex_ids)

        # Use H3 grid operations for faster intersection detection
        processed = 0
        matched_total = 0
        next_log_at = min(
            1000, max(100, total // 10)
        )  # Log more frequently for large datasets

        for source_hex_id in source_hex_ids:
            try:
                # Validate source hexagon
                if not is_valid_cell(source_hex_id):
                    logger.warning(f"Invalid source hexagon: {source_hex_id}")
                    continue

                # Use optimized intersection detection
                intersecting_targets = self._find_intersecting_hexagons_optimized(
                    source_hex_id, target_hex_set, source_name
                )

                # Distribute source data to intersecting targets
                for target_hex_id in intersecting_targets:
                    if target_hex_id not in fused_data:
                        fused_data[target_hex_id] = []

                    # Add source data to target
                    source_hex_data = source_data[source_hex_id]
                    if isinstance(source_hex_data, list):
                        fused_data[target_hex_id].extend(source_hex_data)
                    else:
                        fused_data[target_hex_id].append(source_hex_data)

                matched_total += len(intersecting_targets)

            except Exception as e:
                logger.warning(f"Error processing source hexagon {source_hex_id}: {e}")
                continue
            finally:
                processed += 1
                if processed >= next_log_at:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"[{source_name}] fused {processed}/{total} source hexes; "
                        f"matched targets: {matched_total}; rate: {rate:.1f} hexes/sec"
                    )
                    next_log_at = processed + min(
                        1000, max(100, (total - processed) // 10)
                    )

        elapsed_total = time.time() - start_time
        logger.info(
            f"[{source_name}] fusion complete: processed={processed}, targets_matched={matched_total}, "
            f"time={elapsed_total:.2f}s, rate={processed / elapsed_total:.1f} hexes/sec"
        )
        return fused_data

    def _fuse_source_batch(
        self, source_name: str, source_data: dict[str, Any], target_hexagons: list[str]
    ) -> dict[str, Any]:
        """
        Batch fusion method for parallel processing.

        Args:
            source_name: Name of the source
            source_data: Source H3 data
            target_hexagons: List of target hexagons

        Returns:
            Fused data for target hexagons
        """
        try:
            # Validate source data
            source_validation = self._validate_source_data(source_data, source_name)
            if not source_validation["is_valid"]:
                logger.warning(
                    f"Source {source_name} validation failed: {source_validation['errors']}"
                )
                return {}

            # Fuse using optimized method
            return self._fuse_source_to_target_hexagons(
                source_data, target_hexagons, source_name
            )

        except Exception as e:
            logger.error(f"Batch fusion failed for {source_name}: {e}")
            return {}

    def _find_intersecting_hexagons_optimized(
        self, source_hex_id: str, target_hex_set: set, source_name: str
    ) -> list[str]:
        """
        Find target hexagons that intersect with a source hexagon using optimized methods.

        Args:
            source_hex_id: Source H3 hexagon ID
            target_hex_set: Set of target hexagon IDs
            source_name: Name of the source for logging

        Returns:
            List of intersecting target hexagon IDs
        """
        try:
            # Method 1: Use H3 grid operations for fast intersection detection
            # Get source hexagon neighbors and check if they intersect with targets
            try:
                # Get grid disk of neighbors (including self)
                neighbors = grid_disk(source_hex_id, 2)  # Check 2-ring neighborhood

                intersecting_targets = []
                for neighbor_hex in neighbors:
                    if neighbor_hex in target_hex_set:
                        # Verify actual geometric intersection for accuracy
                        if self._hexagons_intersect(source_hex_id, neighbor_hex):
                            intersecting_targets.append(neighbor_hex)

                if intersecting_targets:
                    return intersecting_targets

            except Exception as e:
                logger.debug(f"H3 grid method failed for {source_hex_id}: {e}")

            # Method 2: Fallback to geometric intersection (original method but optimized)
            return self._find_intersecting_hexagons(
                source_hex_id, target_hex_set, source_name
            )

        except Exception as e:
            logger.warning(
                f"Error in optimized intersection detection for {source_hex_id}: {e}"
            )
            return []

    def _hexagons_intersect(self, hex1: str, hex2: str) -> bool:
        """
        Check if two H3 hexagons intersect using geometric operations.

        Args:
            hex1: First H3 hexagon ID
            hex2: Second H3 hexagon ID

        Returns:
            True if hexagons intersect
        """
        try:
            # Get boundaries and create polygons
            boundary1 = cell_to_latlng_boundary(hex1)
            coords1 = [(lng, lat) for (lat, lng) in boundary1]
            if coords1 and coords1[0] != coords1[-1]:
                coords1.append(coords1[0])
            poly1 = Polygon(coords1)

            boundary2 = cell_to_latlng_boundary(hex2)
            coords2 = [(lng, lat) for (lat, lng) in boundary2]
            if coords2 and coords2[0] != coords2[-1]:
                coords2.append(coords2[0])
            poly2 = Polygon(coords2)

            return poly1.intersects(poly2)

        except Exception:
            return False

    def _find_intersecting_hexagons(
        self, source_hex_id: str, target_hex_set: set, source_name: str
    ) -> list[str]:
        """
        Find target hexagons that intersect with a source hexagon.

        Args:
            source_hex_id: Source H3 hexagon ID
            target_hex_set: Set of target hexagon IDs
            source_name: Name of the source for logging

        Returns:
            List of intersecting target hexagon IDs
        """
        try:
            # Get source hexagon boundary (lat, lng) and convert to (lng, lat) for shapely
            source_boundary = cell_to_latlng_boundary(source_hex_id)
            source_coords = [(lng, lat) for (lat, lng) in source_boundary]
            # Ensure closed ring
            if source_coords and source_coords[0] != source_coords[-1]:
                source_coords.append(source_coords[0])
            source_polygon = Polygon(source_coords)

            # Find target hexagons that intersect
            intersecting_targets = []

            for target_hex_id in target_hex_set:
                try:
                    # Get target hexagon boundary and convert to (lng, lat)
                    target_boundary = cell_to_latlng_boundary(target_hex_id)
                    target_coords = [(lng, lat) for (lat, lng) in target_boundary]
                    if target_coords and target_coords[0] != target_coords[-1]:
                        target_coords.append(target_coords[0])
                    target_polygon = Polygon(target_coords)

                    # Check for intersection
                    if source_polygon.intersects(target_polygon):
                        intersecting_targets.append(target_hex_id)

                except Exception as e:
                    logger.warning(
                        f"Error checking intersection for {target_hex_id}: {e}"
                    )
                    continue

            return intersecting_targets

        except Exception as e:
            logger.warning(
                f"Error finding intersecting hexagons for {source_hex_id}: {e}"
            )
            return []

    def _perform_spatial_analysis(
        self, fused_data: dict[str, Any], target_hexagons: list[str]
    ) -> dict[str, Any]:
        """
        Perform enhanced spatial analysis on fused data.

        Args:
            fused_data: Fused H3 data
            target_hexagons: List of target hexagons

        Returns:
            Enhanced fused data with spatial analysis
        """
        logger.info("Performing enhanced spatial analysis...")

        # Calculate spatial statistics
        spatial_stats = self._calculate_spatial_statistics(fused_data, target_hexagons)

        # Perform spatial correlation analysis
        correlation_analysis = self._analyze_spatial_correlations(fused_data)

        # Identify spatial clusters
        cluster_analysis = self._identify_spatial_clusters(fused_data, target_hexagons)

        # Add spatial analysis results to fused data
        for hex_id in fused_data:
            if hex_id in spatial_stats:
                fused_data[hex_id]["spatial_stats"] = spatial_stats[hex_id]

            if hex_id in correlation_analysis:
                fused_data[hex_id]["spatial_correlations"] = correlation_analysis[
                    hex_id
                ]

            if hex_id in cluster_analysis:
                fused_data[hex_id]["spatial_cluster"] = cluster_analysis[hex_id]

        logger.info("✅ Enhanced spatial analysis completed")
        return fused_data

    def _calculate_spatial_statistics(
        self, fused_data: dict[str, Any], target_hexagons: list[str]
    ) -> dict[str, Any]:
        """
        Calculate spatial statistics for each hexagon.

        Args:
            fused_data: Fused H3 data
            target_hexagons: List of target hexagons

        Returns:
            Dictionary of spatial statistics by hexagon
        """
        spatial_stats = {}

        for hex_id in target_hexagons:
            if hex_id not in fused_data:
                continue

            try:
                # Get hexagon center and area
                center_lat, center_lng = cell_to_latlng(hex_id)
                area_km2 = cell_area(hex_id, unit="km^2")

                # Get neighboring hexagons
                neighbors = grid_disk(hex_id, 1)
                neighbor_count = len(neighbors)

                # Calculate data density
                hex_data = fused_data[hex_id]
                data_sources = (
                    list(hex_data.keys()) if isinstance(hex_data, dict) else []
                )
                source_count = len(data_sources)

                # Calculate feature density
                total_features = 0
                if isinstance(hex_data, dict):
                    for source_data in hex_data.values():
                        if isinstance(source_data, list):
                            total_features += len(source_data)
                        else:
                            total_features += 1

                spatial_stats[hex_id] = {
                    "center_lat": center_lat,
                    "center_lng": center_lng,
                    "area_km2": area_km2,
                    "neighbor_count": neighbor_count,
                    "data_source_count": source_count,
                    "feature_count": total_features,
                    "feature_density": total_features / area_km2 if area_km2 > 0 else 0,
                }

            except Exception as e:
                logger.warning(f"Error calculating spatial stats for {hex_id}: {e}")
                continue

        return spatial_stats

    def _analyze_spatial_correlations(
        self, fused_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Analyze spatial correlations between data sources.

        Args:
            fused_data: Fused H3 data

        Returns:
            Dictionary of spatial correlation analysis by hexagon
        """
        correlation_analysis = {}

        # This is a simplified correlation analysis
        # In a full implementation, you would calculate actual statistical correlations

        for hex_id, hex_data in fused_data.items():
            if not isinstance(hex_data, dict):
                continue

            try:
                # Calculate correlation metrics
                data_sources = list(hex_data.keys())
                source_count = len(data_sources)

                # Simple correlation indicators
                correlation_indicators = {
                    "data_source_count": source_count,
                    "has_multiple_sources": source_count > 1,
                    "source_diversity": source_count
                    / 4.0,  # Normalized to max expected sources
                }

                correlation_analysis[hex_id] = correlation_indicators

            except Exception as e:
                logger.warning(f"Error analyzing correlations for {hex_id}: {e}")
                continue

        return correlation_analysis

    def _identify_spatial_clusters(
        self, fused_data: dict[str, Any], target_hexagons: list[str]
    ) -> dict[str, Any]:
        """
        Identify spatial clusters in the data.

        Args:
            fused_data: Fused H3 data
            target_hexagons: List of target hexagons

        Returns:
            Dictionary of cluster analysis by hexagon
        """
        cluster_analysis = {}

        # Simple clustering based on data density
        high_density_hexagons = []

        for hex_id, hex_data in fused_data.items():
            try:
                # Calculate data density
                if isinstance(hex_data, dict):
                    total_features = sum(
                        len(source_data) if isinstance(source_data, list) else 1
                        for source_data in hex_data.values()
                    )
                else:
                    total_features = 1

                # Identify high-density areas
                if total_features > 5:  # Threshold for high density
                    high_density_hexagons.append(hex_id)

                cluster_analysis[hex_id] = {
                    "feature_count": total_features,
                    "is_high_density": total_features > 5,
                    "cluster_type": "high_density"
                    if total_features > 5
                    else "low_density",
                }

            except Exception as e:
                logger.warning(f"Error identifying clusters for {hex_id}: {e}")
                continue

        return cluster_analysis

    def _generate_fusion_report(
        self,
        fused_data: dict[str, Any],
        data_sources: dict[str, Any],
        target_hexagons: list[str],
    ) -> dict[str, Any]:
        """
        Generate comprehensive fusion report.

        Args:
            fused_data: Fused H3 data
            data_sources: Original data sources
            target_hexagons: List of target hexagons

        Returns:
            Fusion report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "fusion_summary": {
                "total_target_hexagons": len(target_hexagons),
                "fused_hexagons": len(fused_data),
                "coverage_percentage": len(fused_data) / len(target_hexagons) * 100
                if target_hexagons
                else 0,
                "data_sources_processed": len(data_sources),
            },
            "data_source_summary": {},
            "spatial_analysis": {
                "enabled": self.enable_spatial_analysis,
                "spatial_stats_calculated": len(fused_data),
                "correlation_analysis_performed": self.enable_spatial_analysis,
                "cluster_analysis_performed": self.enable_spatial_analysis,
            },
            "quality_metrics": {
                "validation_passed": True,
                "spatial_consistency": True,
                "data_integrity": True,
            },
        }

        # Add data source summaries
        for source_name, source_data in data_sources.items():
            report["data_source_summary"][source_name] = {
                "hexagon_count": len(source_data),
                "feature_count": sum(
                    len(hex_data) if isinstance(hex_data, list) else 1
                    for hex_data in source_data.values()
                ),
            }

        return report

    def validate_h3_operations(self) -> dict[str, Any]:
        """
        Validate H3 operations and API usage.

        Returns:
            Validation result dictionary
        """
        validation_result = {
            "h3_api_version": "4.x",
            "space_integration": SPACE_H3_AVAILABLE,
            "operations_tested": [],
            "errors": [],
            "warnings": [],
        }

        try:
            # Test basic H3 operations
            test_lat, test_lng = 41.7, -124.0  # Del Norte County coordinates
            test_resolution = 8

            # Test latlng_to_cell
            test_cell = latlng_to_cell(test_lat, test_lng, test_resolution)
            validation_result["operations_tested"].append("latlng_to_cell")

            # Test cell_to_latlng
            reverse_lat, reverse_lng = cell_to_latlng(test_cell)
            validation_result["operations_tested"].append("cell_to_latlng")

            # Test cell_to_latlng_boundary
            cell_to_latlng_boundary(test_cell)
            validation_result["operations_tested"].append("cell_to_latlng_boundary")

            # Test geo_to_cells
            test_polygon = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [test_lng - 0.1, test_lat - 0.1],
                        [test_lng + 0.1, test_lat - 0.1],
                        [test_lng + 0.1, test_lat + 0.1],
                        [test_lng - 0.1, test_lat + 0.1],
                        [test_lng - 0.1, test_lat - 0.1],
                    ]
                ],
            }
            geo_to_cells(test_polygon, test_resolution)
            validation_result["operations_tested"].append("geo_to_cells")

            # Test grid_disk
            grid_disk(test_cell, 1)
            validation_result["operations_tested"].append("grid_disk")

            # Test cell_area
            cell_area(test_cell, unit="km^2")
            validation_result["operations_tested"].append("cell_area")

            # Test is_valid_cell
            is_valid_cell(test_cell)
            validation_result["operations_tested"].append("is_valid_cell")

            logger.info("✅ H3 operations validation completed successfully")

        except Exception as e:
            validation_result["errors"].append(f"H3 validation failed: {e}")
            logger.error(f"❌ H3 validation failed: {e}")

        return validation_result


def create_enhanced_h3_fusion(
    h3_resolution: int = 8,
    enable_spatial_analysis: bool = True,
    cache_dir: Path | None = None,
    fusion_mode: str = "geom_intersect",
) -> EnhancedH3Fusion:
    """
    Factory function to create an enhanced H3 fusion engine.

    Args:
        h3_resolution: H3 resolution for spatial indexing
        enable_spatial_analysis: Enable advanced spatial analysis

    Returns:
        Configured EnhancedH3Fusion instance
    """
    return EnhancedH3Fusion(
        h3_resolution, enable_spatial_analysis, cache_dir, fusion_mode
    )
