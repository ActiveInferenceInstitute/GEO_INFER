#!/usr/bin/env python3
"""
Enhanced Data Manager for Cascadia Agricultural Analysis Framework

This module provides comprehensive data management with:
- Real H3 v4 geospatial data fusion
- Reproducible data module structure
- Intelligent caching and data validation
- SPACE integration for advanced geospatial operations
"""

import logging
import json
from pathlib import Path
from typing import Any
from datetime import datetime
import geopandas as gpd
import h3
import time
import psutil
import sys

# Import enhanced logging
from .enhanced_logging import (
    DataSourceLogger,
    ProcessingLogger,
    log_geodataframe_summary,
)

# Import H3 utilities from consolidated geo_infer_place module
try:
    from geo_infer_place.utils.h3_operations import (
        latlng_to_cell,
        cell_to_latlng,
        cell_to_latlng_boundary,
        geo_to_cells,
        polygon_to_cells,
        grid_disk,
        grid_distance,
        cell_area,
        get_resolution,
        is_valid_cell,
        are_neighbor_cells,
    )

    SPACE_H3_AVAILABLE = True
except ImportError:
    # Fallback to direct h3 imports
    latlng_to_cell = h3.latlng_to_cell
    cell_to_latlng = h3.cell_to_latlng
    cell_to_latlng_boundary = h3.cell_to_boundary

    def polygon_to_cells(geojson, res):
        return list(h3.geo_to_cells(geojson, res))

    grid_disk = h3.grid_disk
    grid_distance = h3.grid_distance
    cell_area = h3.cell_area
    get_resolution = h3.get_resolution
    is_valid_cell = h3.is_valid_cell
    are_neighbor_cells = h3.are_neighbor_cells

    def geo_to_cells(geojson, res):
        return list(h3.geo_to_cells(geojson, res))

    SPACE_H3_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedDataManager:
    """
    Comprehensive data manager for Cascadia agricultural analysis.

    Features:
    - Real H3 v4 geospatial data fusion
    - Reproducible data module structure
    - Intelligent caching with validation
    - SPACE integration for advanced operations
    - Data quality assessment and reporting
    """

    def __init__(self, base_data_dir: Path, h3_resolution: int = 8):
        """
        Initialize the enhanced data manager.

        Args:
            base_data_dir: Base directory for data storage
            h3_resolution: H3 resolution for spatial indexing
        """
        self.base_data_dir = Path(base_data_dir)
        self.h3_resolution = h3_resolution
        self.cache_dir = self.base_data_dir / "cache"
        self.empirical_dir = self.base_data_dir / "empirical"
        self.processed_dir = self.base_data_dir / "processed"

        # Create directory structure
        for dir_path in [self.cache_dir, self.empirical_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Data validation settings
        self.validation_settings = {
            "min_file_size_bytes": 100,
            "max_file_size_mb": 100,
            "required_geometry_types": ["Polygon", "Point", "MultiPolygon"],
            "required_crs": "EPSG:4326",
            "h3_validation": True,
            "data_quality_threshold": 0.8,
        }

        # Initialize enhanced loggers
        self.data_logger = DataSourceLogger("data_manager")
        self.processing_logger = ProcessingLogger("data_manager")

        logger.info(f"Enhanced Data Manager initialized with H3 resolution {h3_resolution}")
        logger.info(f"SPACE H3 utilities available: {SPACE_H3_AVAILABLE}")

    def get_comprehensive_data_quality_report(self, module_name: str) -> dict[str, Any]:
        """
        Generate comprehensive data quality report for a module.

        Reads actual data files, validates GeoDataFrames, and generates
        quality scores with recommendations. For a simpler file-existence
        check, use ``get_data_quality_report``.

        Args:
            module_name: Name of the module to analyze

        Returns:
            Comprehensive quality report
        """
        report = {
            "module": module_name,
            "timestamp": datetime.now().isoformat(),
            "h3_resolution": self.h3_resolution,
            "data_sources": {},
            "quality_metrics": {},
            "recommendations": [],
        }

        try:
            data_paths = self.get_data_structure(module_name)

            # Check each data source
            for source_type, path_key in [
                ("empirical", "empirical_data"),
                ("raw", "raw_data"),
            ]:
                path = data_paths[path_key]
                if path.exists():
                    try:
                        gdf = gpd.read_file(path)
                        validation = self._validate_geodataframe(gdf, module_name)
                        is_empirical = self._is_empirical_data(gdf, module_name)

                        report["data_sources"][source_type] = {
                            "exists": True,
                            "file_path": str(path),
                            "file_size_mb": path.stat().st_size / 1024 / 1024,
                            "feature_count": len(gdf),
                            "is_empirical": is_empirical,
                            "validation": validation,
                        }

                        # Add quality metrics
                        report["quality_metrics"][source_type] = {
                            "quality_score": validation["quality_score"],
                            "completeness": validation["data_quality_metrics"].get(
                                "completeness_score", 0
                            ),
                            "validity": validation["data_quality_metrics"].get("validity_score", 0),
                            "consistency": validation["data_quality_metrics"].get(
                                "consistency_score", 0
                            ),
                        }

                    except Exception as e:
                        report["data_sources"][source_type] = {
                            "exists": True,
                            "file_path": str(path),
                            "error": str(e),
                            "is_empirical": False,
                        }

                else:
                    report["data_sources"][source_type] = {
                        "exists": False,
                        "file_path": str(path),
                        "is_empirical": False,
                    }

            # Check cache
            cache_path = data_paths["h3_cache"]
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        cache_data = json.load(f)
                    report["data_sources"]["h3_cache"] = {
                        "exists": True,
                        "file_path": str(cache_path),
                        "hexagon_count": len(cache_data.get("hexagons", {})),
                        "file_size_mb": cache_path.stat().st_size / 1024 / 1024,
                    }
                except Exception as e:
                    report["data_sources"]["h3_cache"] = {
                        "exists": True,
                        "file_path": str(cache_path),
                        "error": str(e),
                    }
            else:
                report["data_sources"]["h3_cache"] = {
                    "exists": False,
                    "file_path": str(cache_path),
                }

            # Generate recommendations
            empirical_sources = [
                k for k, v in report["data_sources"].items() if v.get("is_empirical", False)
            ]
            if len(empirical_sources) == 0:
                report["recommendations"].append(
                    "No empirical data sources found - consider acquiring real data"
                )
            elif len(empirical_sources) > 1:
                report["recommendations"].append(
                    f"Multiple empirical sources available: {empirical_sources}"
                )

            # Check cache status
            if not report["data_sources"].get("h3_cache", {}).get("exists", False):
                report["recommendations"].append(
                    "No H3 cache found - data processing may be slower"
                )

            # Quality score analysis
            quality_scores = [
                v["quality_score"]
                for v in report["quality_metrics"].values()
                if "quality_score" in v
            ]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                if avg_quality < 0.7:
                    report["recommendations"].append(
                        f"Low average data quality ({avg_quality:.2f}) - review data sources"
                    )
                elif avg_quality > 0.9:
                    report["recommendations"].append(
                        f"High data quality ({avg_quality:.2f}) - excellent data sources"
                    )

        except Exception as e:
            logger.error(f"Error generating data quality report for {module_name}: {e}")
            report["error"] = str(e)

        return report

    def benchmark_performance(self, module_name: str) -> dict[str, Any]:
        """
        Benchmark performance for a module's data processing operations.

        Args:
            module_name: Name of the module to benchmark

        Returns:
            Performance benchmark results
        """
        import time
        import psutil
        import gc

        benchmark_results = {
            "module": module_name,
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "benchmarks": {},
            "recommendations": [],
        }

        try:
            # Benchmark data loading
            data_paths = self.get_data_structure(module_name)

            for data_type, path_key in [
                ("empirical", "empirical_data"),
                ("raw", "raw_data"),
            ]:
                path = data_paths[path_key]
                if path.exists():
                    try:
                        start_time = time.time()
                        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

                        # Load data
                        gdf = gpd.read_file(path)

                        end_time = time.time()
                        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                        load_time = end_time - start_time
                        memory_increase = end_memory - start_memory

                        benchmark_results["benchmarks"][f"{data_type}_loading"] = {
                            "load_time_seconds": load_time,
                            "memory_increase_mb": memory_increase,
                            "file_size_mb": path.stat().st_size / 1024 / 1024,
                            "feature_count": len(gdf),
                            "performance_score": self._calculate_performance_score(
                                load_time, memory_increase, len(gdf)
                            ),
                        }

                        # Clean up memory
                        del gdf
                        gc.collect()

                    except Exception as e:
                        benchmark_results["benchmarks"][f"{data_type}_loading"] = {
                            "error": str(e),
                            "performance_score": 0,
                        }

            # Benchmark H3 processing if cache exists
            cache_path = data_paths["h3_cache"]
            if cache_path.exists():
                try:
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024

                    with open(cache_path) as f:
                        cache_data = json.load(f)

                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                    load_time = end_time - start_time
                    memory_increase = end_memory - start_memory
                    hexagon_count = len(cache_data.get("hexagons", {}))

                    benchmark_results["benchmarks"]["h3_cache_loading"] = {
                        "load_time_seconds": load_time,
                        "memory_increase_mb": memory_increase,
                        "file_size_mb": cache_path.stat().st_size / 1024 / 1024,
                        "hexagon_count": hexagon_count,
                        "performance_score": self._calculate_performance_score(
                            load_time, memory_increase, hexagon_count
                        ),
                    }

                except Exception as e:
                    benchmark_results["benchmarks"]["h3_cache_loading"] = {
                        "error": str(e),
                        "performance_score": 0,
                    }

            # Generate recommendations based on benchmarks
            self._generate_performance_recommendations(benchmark_results)

        except Exception as e:
            logger.error(f"Error during performance benchmarking for {module_name}: {e}")
            benchmark_results["error"] = str(e)

        return benchmark_results

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for benchmarking context."""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            }
        except Exception:
            return {"error": "Could not retrieve system info"}

    def _calculate_performance_score(
        self, load_time: float, memory_mb: float, data_size: int
    ) -> float:
        """
        Calculate a performance score based on time, memory, and data size.

        Args:
            load_time: Time to load data in seconds
            memory_mb: Memory increase in MB
            data_size: Size of data (features or hexagons)

        Returns:
            Performance score (0-1, higher is better)
        """
        # Score based on time (faster is better)
        time_score = max(0, 1.0 - (load_time / 10.0))  # Assume 10 seconds is poor performance

        # Score based on memory efficiency (lower memory increase is better)
        memory_score = max(0, 1.0 - (memory_mb / 1000.0))  # Assume 1GB increase is poor

        # Score based on data throughput (more data per second is better)
        throughput_score = min(1.0, data_size / load_time / 1000.0) if load_time > 0 else 1.0

        # Weighted average
        return time_score * 0.4 + memory_score * 0.3 + throughput_score * 0.3

    def _generate_performance_recommendations(self, benchmark_results: dict[str, Any]):
        """Generate performance optimization recommendations."""
        recommendations = []

        system_info = benchmark_results["system_info"]
        benchmarks = benchmark_results["benchmarks"]

        # Memory recommendations
        memory_available_gb = system_info.get("memory_available_gb", 0)
        if memory_available_gb < 2:
            recommendations.append(
                "Low available memory detected - consider increasing system RAM or reducing batch sizes"
            )
        elif memory_available_gb > 16:
            recommendations.append(
                "High memory available - consider increasing parallel workers for better performance"
            )

        # Performance score analysis
        for benchmark_name, results in benchmarks.items():
            if "performance_score" in results:
                score = results["performance_score"]
                if score < 0.3:
                    recommendations.append(
                        f"Poor performance in {benchmark_name} (score: {score:.2f}) - consider optimization"
                    )
                elif score > 0.8:
                    recommendations.append(
                        f"Excellent performance in {benchmark_name} (score: {score:.2f})"
                    )

        # Data size recommendations
        for benchmark_name, results in benchmarks.items():
            if "file_size_mb" in results:
                size_mb = results["file_size_mb"]
                if size_mb > 100:
                    recommendations.append(
                        f"Large {benchmark_name} file ({size_mb:.1f} MB) - consider data compression or chunked processing"
                    )
                elif size_mb < 1:
                    recommendations.append(
                        f"Small {benchmark_name} file ({size_mb:.1f} MB) - consider batching multiple operations"
                    )

        benchmark_results["recommendations"] = recommendations

    def get_data_structure(self, module_name: str) -> dict[str, Path]:
        """
        Get the standardized data structure for a module.

        Args:
            module_name: Name of the analysis module

        Returns:
            Dictionary with standardized data paths
        """
        # Use module-specific directory structure
        module_data_dir = self.base_data_dir.parent / module_name / "data"
        module_data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        empirical_dir = module_data_dir / "empirical"
        cache_dir = module_data_dir / "cache"
        processed_dir = module_data_dir / "processed"
        raw_dir = module_data_dir / "raw"

        for subdir in [empirical_dir, cache_dir, processed_dir, raw_dir]:
            subdir.mkdir(parents=True, exist_ok=True)

        return {
            "module_dir": module_data_dir,
            "empirical_data": empirical_dir / f"empirical_{module_name}_data.geojson",
            "raw_data": raw_dir / f"raw_{module_name}_data.geojson",
            "h3_cache": cache_dir / f"{module_name}_h3_res{self.h3_resolution}.json",
            "processed_data": processed_dir / f"processed_{module_name}_data.geojson",
            "metadata": module_data_dir / f"{module_name}_metadata.json",
            "validation_report": module_data_dir / f"{module_name}_validation_report.json",
        }

    def acquire_data_with_caching(
        self, module_name: str, data_source_func, force_refresh: bool = False
    ) -> Path:
        """
        Acquire data with intelligent caching and validation.

        Args:
            module_name: Name of the module
            data_source_func: Function that returns raw data path
            force_refresh: Force refresh of cached data

        Returns:
            Path to the acquired data file
        """
        start_time = time.time()
        self.processing_logger.log_processing_start(
            "Data Acquisition", {"module": module_name, "force_refresh": force_refresh}
        )

        data_paths = self.get_data_structure(module_name)

        # Check for empirical data first
        if data_paths["empirical_data"].exists() and not force_refresh:
            file_size_mb = data_paths["empirical_data"].stat().st_size / 1024 / 1024
            self.data_logger.log_real_data_acquisition(
                source_url="Cached empirical data",
                file_path=data_paths["empirical_data"],
                data_type="Empirical",
                row_count=0,  # Will be updated after loading
                file_size_mb=file_size_mb,
                geometry_types=["Unknown"],
                crs="Unknown",
            )
            self.processing_logger.log_processing_complete(
                "Data Acquisition",
                {"source": "cached_empirical", "module": module_name},
                time.time() - start_time,
            )
            return data_paths["empirical_data"]

        # Check for raw data
        if data_paths["raw_data"].exists() and not force_refresh:
            file_size_mb = data_paths["raw_data"].stat().st_size / 1024 / 1024
            self.data_logger.log_real_data_acquisition(
                source_url="Cached raw data",
                file_path=data_paths["raw_data"],
                data_type="Raw",
                row_count=0,  # Will be updated after loading
                file_size_mb=file_size_mb,
                geometry_types=["Unknown"],
                crs="Unknown",
            )
            self.processing_logger.log_processing_complete(
                "Data Acquisition",
                {"source": "cached_raw", "module": module_name},
                time.time() - start_time,
            )
            return data_paths["raw_data"]

        # Acquire new data
        self.processing_logger.log_processing_step(
            "Data Acquisition",
            {"action": "calling_data_source_func", "module": module_name},
        )

        try:
            raw_data_path = data_source_func()
            if raw_data_path and Path(raw_data_path).exists():
                # Validate and copy to appropriate location
                validated_path = self._validate_and_store_data(
                    raw_data_path, module_name, data_paths
                )
                self.processing_logger.log_processing_complete(
                    "Data Acquisition",
                    {
                        "source": "new_data",
                        "module": module_name,
                        "path": str(validated_path),
                    },
                    time.time() - start_time,
                )
                return validated_path
            else:
                raise FileNotFoundError(
                    f"Data source did not return an existing path for {module_name}"
                )
        except Exception as e:
            raise RuntimeError(f"Data acquisition failed for {module_name}") from e

    def _validate_and_store_data(
        self, raw_data_path: Path, module_name: str, data_paths: dict[str, Path]
    ) -> Path:
        """
        Validate data and store in appropriate location.

        Args:
            raw_data_path: Path to raw data file
            module_name: Name of the module
            data_paths: Data structure paths

        Returns:
            Path to validated data file
        """
        start_time = time.time()
        self.processing_logger.log_processing_start(
            "Data Validation and Storage",
            {"module": module_name, "raw_path": str(raw_data_path)},
        )

        try:
            # Load and validate data
            gdf = gpd.read_file(raw_data_path)
            # Ensure CRS is WGS84 if missing
            if gdf.crs is None:
                logger.warning(f"[{module_name}] Missing CRS, assuming EPSG:4326")
                gdf.set_crs(epsg=4326, inplace=True)
            # Repair invalid geometries
            try:
                invalid_mask = ~gdf.geometry.is_valid
                if invalid_mask.any():
                    gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
            except Exception as exc:
                logger.warning("Data quality step failed; continuing without it: %s", exc)

            # Log comprehensive data summary
            log_geodataframe_summary(logger, gdf, f"{module_name}_raw_data")

            # Calculate file size and performance metrics
            file_size_mb = raw_data_path.stat().st_size / 1024 / 1024

            # Perform validation
            validation_result = self._validate_geodataframe(gdf, module_name)

            # Log validation results
            self.data_logger.log_data_validation(
                validation_results=validation_result,
                quality_score=validation_result.get("quality_score", 0.0),
                issues=validation_result.get("issues", []),
            )

            if validation_result["is_valid"]:
                # Require the source-attribution and quality checks for accepted data.
                is_empirical = self._is_empirical_data(gdf, module_name)

                if is_empirical:
                    target_path = data_paths["empirical_data"]

                    # Log real data acquisition with comprehensive details
                    bbox = gdf.total_bounds if not gdf.empty else None
                    geometry_types = list(gdf.geometry.geom_type.unique()) if not gdf.empty else []
                    attributes = list(gdf.columns) if not gdf.empty else []

                    self.data_logger.log_real_data_acquisition(
                        source_url=str(raw_data_path),
                        file_path=target_path,
                        data_type="Empirical",
                        row_count=len(gdf),
                        file_size_mb=file_size_mb,
                        geometry_types=geometry_types,
                        crs=str(gdf.crs),
                        bbox=bbox,
                        attributes=attributes,
                    )
                else:
                    raise ValueError(
                        f"Dataset for {module_name} failed the authoritative-source check"
                    )

                # Save validated data
                gdf.to_file(target_path, driver="GeoJSON")

                # Save validation report
                with open(data_paths["validation_report"], "w") as f:
                    json.dump(validation_result, f, indent=2)

                # Log performance metrics
                duration = time.time() - start_time
                self.processing_logger.log_processing_complete(
                    "Data Validation and Storage",
                    {
                        "module": module_name,
                        "target_path": str(target_path),
                        "is_empirical": is_empirical,
                        "row_count": len(gdf),
                        "file_size_mb": file_size_mb,
                    },
                    duration,
                )

                return target_path
            else:
                raise ValueError(
                    f"Data validation failed for {module_name}: {validation_result['errors']}"
                )

        except Exception as e:
            raise RuntimeError(f"Data validation failed for {module_name}") from e

    def _validate_geodataframe(self, gdf: gpd.GeoDataFrame, module_name: str) -> dict[str, Any]:
        """
        Validate a GeoDataFrame for quality and consistency.

        Args:
            gdf: GeoDataFrame to validate
            module_name: Name of the module for context

        Returns:
            Validation result dictionary
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_score": 0.0,
            "feature_count": len(gdf),
            "geometry_types": (gdf.geometry.geom_type.unique().tolist() if not gdf.empty else []),
            "crs": str(gdf.crs) if gdf.crs else "None",
            "attribute_summary": {},
            "spatial_summary": {},
            "data_quality_metrics": {},
        }

        # Basic validation checks
        if len(gdf) == 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Empty dataset")
            return validation_result

        # Geometry validation
        valid_geometry_types = set(self.validation_settings["required_geometry_types"])
        actual_geometry_types = set(gdf.geometry.geom_type.unique())
        if not actual_geometry_types.intersection(valid_geometry_types):
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Invalid geometry types: {actual_geometry_types}")

        # CRS validation
        if not gdf.crs:
            validation_result["errors"].append("Missing coordinate reference system")
            validation_result["is_valid"] = False
        elif "EPSG:4326" not in str(gdf.crs):
            validation_result["warnings"].append(f"Non-standard CRS: {gdf.crs}")

        # Geometry quality checks
        null_geometries = gdf.geometry.isna().sum()
        invalid_geometries = 0

        try:
            # Check for invalid geometries
            invalid_mask = ~gdf.geometry.is_valid
            invalid_geometries = invalid_mask.sum()
        except Exception as exc:
            logger.warning("Data quality step failed; continuing without it: %s", exc)

        if null_geometries > 0:
            validation_result["warnings"].append(f"{null_geometries} null geometries found")
        if invalid_geometries > 0:
            validation_result["warnings"].append(f"{invalid_geometries} invalid geometries found")

        # Attribute quality analysis
        attribute_quality = {}
        for col in gdf.columns:
            if col != "geometry":
                null_count = gdf[col].isna().sum()
                unique_count = gdf[col].nunique()
                null_percentage = null_count / len(gdf)

                attribute_quality[col] = {
                    "null_count": int(null_count),
                    "null_percentage": float(null_percentage),
                    "unique_count": int(unique_count),
                    "data_type": str(gdf[col].dtype),
                }

                # Flag quality issues
                if null_percentage > 0.8:
                    validation_result["warnings"].append(
                        f"Column '{col}' has {null_percentage:.1%} missing values"
                    )
                if unique_count == 1:
                    validation_result["warnings"].append(
                        f"Column '{col}' has only one unique value"
                    )

        validation_result["attribute_summary"] = attribute_quality

        # Spatial quality analysis
        if not gdf.empty:
            bounds = gdf.total_bounds
            spatial_summary = {
                "bounds": bounds.tolist() if bounds is not None else None,
                "total_area": (
                    float(gdf.geometry.area.sum()) if gdf.geometry.area.sum() > 0 else 0
                ),
                "mean_area": (
                    float(gdf.geometry.area.mean()) if gdf.geometry.area.mean() > 0 else 0
                ),
                "null_geometries": int(null_geometries),
                "invalid_geometries": int(invalid_geometries),
                "geometry_type_distribution": gdf.geometry.geom_type.value_counts().to_dict(),
            }
            validation_result["spatial_summary"] = spatial_summary

            # Check for unrealistic coordinates (outside reasonable bounds)
            if bounds is not None:
                if not (-180 <= bounds[0] <= 180) or not (-180 <= bounds[2] <= 180):
                    validation_result["warnings"].append("Longitude values outside valid range")
                if not (-90 <= bounds[1] <= 90) or not (-90 <= bounds[3] <= 90):
                    validation_result["warnings"].append("Latitude values outside valid range")

        # Data quality metrics
        quality_metrics = {
            "completeness_score": (1.0 - (null_geometries / len(gdf)) if len(gdf) > 0 else 0),
            "validity_score": (1.0 - (invalid_geometries / len(gdf)) if len(gdf) > 0 else 0),
            "consistency_score": 1.0,
            "accuracy_score": 0.9,  # Default assumption, could be improved with ground truth
        }

        # Check for duplicate geometries
        try:
            duplicate_geoms = gdf.geometry.duplicated().sum()
            if duplicate_geoms > 0:
                validation_result["warnings"].append(
                    f"{duplicate_geoms} duplicate geometries found"
                )
                quality_metrics["consistency_score"] = 1.0 - (duplicate_geoms / len(gdf))
        except Exception as exc:
            logger.warning("Data quality step failed; continuing without it: %s", exc)

        validation_result["data_quality_metrics"] = quality_metrics

        # Calculate comprehensive quality score
        quality_factors = [
            quality_metrics["completeness_score"],
            quality_metrics["validity_score"],
            quality_metrics["consistency_score"],
            quality_metrics["accuracy_score"],
        ]

        # Additional quality factors
        if len(gdf) > 0:
            quality_factors.append(1.0)  # Has data
        if actual_geometry_types.intersection(valid_geometry_types):
            quality_factors.append(1.0)  # Valid geometry types
        if gdf.crs:
            quality_factors.append(0.9)  # Has CRS

        validation_result["quality_score"] = (
            sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
        )

        # Overall validity determination
        validation_result["is_valid"] = (
            len(validation_result["errors"]) == 0
            and validation_result["quality_score"]
            >= self.validation_settings["data_quality_threshold"]
        )

        return validation_result

    def _is_empirical_data(self, gdf: gpd.GeoDataFrame, module_name: str) -> bool:
        """
        Determine whether data has sufficient source attribution and quality.

        Args:
            gdf: GeoDataFrame to analyze
            module_name: Name of the module for context

        Returns:
            True if data appears to be empirical
        """
        if gdf.empty:
            return False

        # Check for empirical indicators
        empirical_indicators = 0
        total_checks = 0

        # 1. Check for realistic coordinate ranges (Del Norte County area)
        total_checks += 1
        bounds = gdf.total_bounds
        if bounds is not None:
            # Del Norte County bounds: ~[-124.5, 41.4, -123.5, 42.0]
            # Check if data falls within broader California/Oregon area
            if (
                -130 < bounds[0] < -115
                and -130 < bounds[2] < -115
                and 35 < bounds[1] < 50
                and 35 < bounds[3] < 50
            ):
                empirical_indicators += 1

        # 2. Check for realistic attribute values and patterns
        total_checks += 1
        empirical_score = 0

        # Check for common agricultural attributes with realistic values
        common_attributes = {
            "acres": lambda x: 0.1 <= float(x) <= 10000,
            "area": lambda x: 0.0001 <= float(x) <= 1000,
            "value": lambda x: 1000 <= float(x) <= 10000000,
            "year": lambda x: 1900 <= int(x) <= 2030,
            "parcel_id": lambda x: len(str(x)) >= 5,
            "owner_name": lambda x: (
                len(str(x)) >= 3 and not str(x).lower().startswith(("test", "generated", "sample"))
            ),
        }

        for col in gdf.columns:
            col_lower = col.lower()
            for attr, validator in common_attributes.items():
                if attr in col_lower or col_lower in attr:
                    try:
                        # Sample first few non-null values
                        sample_values = gdf[col].dropna().head(5)
                        valid_count = 0
                        for val in sample_values:
                            if validator(val):
                                valid_count += 1
                        if valid_count >= 3:  # At least 3 out of 5 values should be realistic
                            empirical_score += 0.5
                            break
                    except (ValueError, TypeError):
                        continue

        if empirical_score >= 0.5:
            empirical_indicators += 1

        # 3. Check for source attribution (real data usually has source info)
        total_checks += 1
        source_indicators = ["source", "data_year", "agency", "county", "state", "fips"]
        source_cols = [
            col
            for col in gdf.columns
            if any(indicator in col.lower() for indicator in source_indicators)
        ]
        if len(source_cols) > 0:
            empirical_indicators += 1

        # 4. Check for realistic feature count
        total_checks += 1
        if 5 <= len(gdf) <= 50000:  # Reasonable range for agricultural data
            empirical_indicators += 1

        # 5. Check for data quality indicators (real data often has some missing values but not too many)
        total_checks += 1
        null_percentages = gdf.isnull().sum() / len(gdf)
        if (null_percentages < 0.8).all():  # Less than 80% missing data
            empirical_indicators += 1

        # Calculate final score
        final_score = empirical_indicators / total_checks if total_checks > 0 else 0

        # Log the analysis for debugging
        logger.debug(
            f"[{module_name}] Empirical data analysis: {empirical_indicators}/{total_checks} indicators "
            f"(score: {final_score:.2f})"
        )

        return final_score >= 0.6  # Require 60% of indicators to be positive

    def process_to_h3_with_caching(
        self, data_path: Path, module_name: str, target_hexagons: list[str]
    ) -> dict[str, Any]:
        """
        Process data to H3 format with intelligent caching.

        Args:
            data_path: Path to input data file
            module_name: Name of the module
            target_hexagons: List of target H3 hexagons

        Returns:
            Dictionary of H3-indexed data
        """
        start_time = time.time()
        self.processing_logger.log_processing_start(
            "H3 Processing",
            {
                "module": module_name,
                "data_path": str(data_path),
                "target_hexagons_count": len(target_hexagons),
            },
        )

        data_paths = self.get_data_structure(module_name)
        cache_path = data_paths["h3_cache"]

        # Check for existing cache
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached_data = json.load(f)
                # Accept both normalized and prior cache formats
                cached_hex_map = cached_data.get(
                    "hexagons", cached_data if isinstance(cached_data, dict) else {}
                )

                # Validate cache against target hexagons
                if self._validate_h3_cache(cached_hex_map, target_hexagons):
                    self.data_logger.log_h3_processing(
                        input_features=(
                            int(cached_data.get("input_features", 0))
                            if isinstance(cached_data, dict)
                            else 0
                        ),
                        output_hexagons=len(cached_hex_map),
                        coverage_percentage=(
                            float(cached_data.get("coverage_percentage", 0.0))
                            if isinstance(cached_data, dict)
                            else 0.0
                        ),
                        processing_time=0.0,  # Cached, so no processing time
                    )
                    self.processing_logger.log_processing_complete(
                        "H3 Processing",
                        {"source": "cached", "module": module_name},
                        time.time() - start_time,
                    )
                    return (
                        cached_data
                        if isinstance(cached_data, dict)
                        else {"hexagons": cached_hex_map}
                    )
                else:
                    self.processing_logger.log_processing_step(
                        "H3 Processing",
                        {"action": "cache_invalid", "module": module_name},
                    )
            except Exception as e:
                self.processing_logger.log_processing_step(
                    "H3 Processing",
                    {
                        "action": "cache_load_failed",
                        "error": str(e),
                        "module": module_name,
                    },
                )

        # Process data to H3 format
        self.processing_logger.log_processing_step(
            "H3 Processing", {"action": "processing_to_h3", "module": module_name}
        )

        try:
            # Load data
            gdf = gpd.read_file(data_path)

            # Process to H3 using SPACE utilities
            h3_data = self._process_geodataframe_to_h3(gdf, target_hexagons, module_name)

            # Normalize to a dict with 'hexagons' for downstream compatibility
            normalized = {"hexagons": h3_data, "input_features": int(len(gdf))}
            # Cache the results
            with open(cache_path, "w") as f:
                json.dump(normalized, f, indent=2)

            # Log H3 processing results
            coverage_pct = (len(h3_data) / len(target_hexagons) * 100.0) if target_hexagons else 0.0
            self.data_logger.log_h3_processing(
                input_features=len(gdf),
                output_hexagons=len(h3_data),
                coverage_percentage=coverage_pct,
                processing_time=time.time() - start_time,
            )

            self.processing_logger.log_processing_complete(
                "H3 Processing",
                {
                    "source": "new_processing",
                    "module": module_name,
                    "input_features": len(gdf),
                    "output_hexagons": len(h3_data),
                },
                time.time() - start_time,
            )

            return normalized

        except Exception as e:
            raise RuntimeError(f"H3 processing failed for {module_name}") from e

    def _process_geodataframe_to_h3(
        self, gdf: gpd.GeoDataFrame, target_hexagons: list[str], module_name: str
    ) -> dict[str, Any]:
        """
        Process GeoDataFrame to H3 format using SPACE utilities.

        Args:
            gdf: Input GeoDataFrame
            target_hexagons: List of target H3 hexagons
            module_name: Name of the module for context

        Returns:
            Dictionary of H3-indexed data
        """
        h3_data: dict[str, Any] = {}

        # Convert target hexagons to set for efficient lookup
        target_hex_set = set(target_hexagons)

        for idx, row in gdf.iterrows():
            try:
                geometry = row.geometry

                # Convert geometry to H3 cells using SPACE utilities
                if geometry is None or geometry.is_empty:
                    continue
                if geometry.geom_type in ("Polygon", "MultiPolygon"):
                    # Convert polygon to GeoJSON format for H3 processing
                    if geometry.geom_type == "Polygon":
                        exterior = list(geometry.exterior.coords)
                        # Ensure ring closure
                        if exterior and exterior[0] != exterior[-1]:
                            exterior.append(exterior[0])
                        # Expect [lng, lat] ordering for geo_to_cells
                        coords = [[float(x), float(y)] for (x, y) in exterior]
                        geojson_geom = {"type": "Polygon", "coordinates": [coords]}
                    else:
                        # Use unary union to dissolve multipolygon into polygon pieces
                        parts = []
                        for poly in geometry.geoms:
                            ext = list(poly.exterior.coords)
                            if ext and ext[0] != ext[-1]:
                                ext.append(ext[0])
                            parts.append([[[float(x), float(y)] for (x, y) in ext]])
                        geojson_geom = {"type": "MultiPolygon", "coordinates": parts}

                    # Use SPACE geo_to_cells for polygon processing
                    hexagons = geo_to_cells(geojson_geom, self.h3_resolution)

                elif geometry.geom_type == "Point":
                    # Convert point to H3 cell
                    lat, lng = geometry.y, geometry.x
                    hexagon = latlng_to_cell(lat, lng, self.h3_resolution)
                    hexagons = [hexagon]

                else:
                    # Handle other geometry types
                    logger.warning(
                        f"[{module_name}] ⚠️ Unsupported geometry type: {geometry.geom_type}"
                    )
                    continue

                # Filter to target hexagons and add to results
                for hex_id in hexagons:
                    if hex_id in target_hex_set:
                        if hex_id not in h3_data:
                            h3_data[hex_id] = []

                        # Convert row to dictionary, excluding geometry
                        feature_data = row.drop("geometry").to_dict()
                        feature_data["feature_id"] = idx

                        h3_data[hex_id].append(feature_data)

            except Exception as e:
                logger.warning(f"[{module_name}] ⚠️ Error processing feature {idx}: {e}")
                continue

        logger.info(
            f"[{module_name}] ✅ Processed {len(gdf)} features to {len(h3_data)} H3 hexagons"
        )
        return h3_data

    def _validate_h3_cache(
        self, cached_hex_map: dict[str, Any], target_hexagons: list[str]
    ) -> bool:
        """
        Validate cached H3 data against target hexagons.

        Args:
            cached_data: Cached H3 data
            target_hexagons: List of target hexagons

        Returns:
            True if cache is valid
        """
        if not cached_hex_map:
            return False

        # Check if all target hexagons are covered
        cached_hexagons = set(cached_hex_map.keys())
        target_hex_set = set(target_hexagons)

        # Require at least 60% coverage (more reasonable threshold)
        coverage = len(cached_hexagons.intersection(target_hex_set)) / len(target_hex_set)

        return coverage >= 0.6

    def get_data_quality_report(self, module_name: str) -> dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Args:
            module_name: Name of the module

        Returns:
            Data quality report
        """
        data_paths = self.get_data_structure(module_name)

        report = {
            "module_name": module_name,
            "timestamp": datetime.now().isoformat(),
            "data_sources": {},
            "h3_processing": {},
            "quality_metrics": {},
        }

        # Check data sources
        for source_name, source_path in data_paths.items():
            if source_path.exists():
                report["data_sources"][source_name] = {
                    "exists": True,
                    "size_bytes": source_path.stat().st_size,
                    "last_modified": datetime.fromtimestamp(
                        source_path.stat().st_mtime
                    ).isoformat(),
                }
            else:
                report["data_sources"][source_name] = {"exists": False}

        # Check H3 processing
        if data_paths["h3_cache"].exists():
            try:
                with open(data_paths["h3_cache"]) as f:
                    h3_data = json.load(f)
                report["h3_processing"] = {
                    "cached_hexagons": len(h3_data),
                    "cache_size_bytes": data_paths["h3_cache"].stat().st_size,
                }
            except Exception as e:
                report["h3_processing"]["error"] = str(e)

        # Check validation report
        if data_paths["validation_report"].exists():
            try:
                with open(data_paths["validation_report"]) as f:
                    validation = json.load(f)
                report["quality_metrics"] = validation
            except Exception as e:
                report["quality_metrics"]["error"] = str(e)

        return report

    def cleanup_old_cache(self, max_age_days: int = 30) -> int:
        """
        Clean up old cache files.

        Args:
            max_age_days: Maximum age in days for cache files

        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)

        for cache_file in self.cache_dir.rglob("*.json"):
            if cache_file.stat().st_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Cleaned up old cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up cache file {cache_file}: {e}")

        return cleaned_count


def create_enhanced_data_manager(
    base_data_dir: Path, h3_resolution: int = 8
) -> EnhancedDataManager:
    """
    Factory function to create an enhanced data manager.

    Args:
        base_data_dir: Base directory for data storage
        h3_resolution: H3 resolution for spatial indexing

    Returns:
        Configured EnhancedDataManager instance
    """
    return EnhancedDataManager(base_data_dir, h3_resolution)
