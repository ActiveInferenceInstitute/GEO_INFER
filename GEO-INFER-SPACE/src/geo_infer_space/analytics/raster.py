"""
Raster operations module for advanced spatial analysis.

This module provides comprehensive raster-based spatial operations including
terrain analysis, map algebra, focal statistics, and image processing
using rasterio, numpy, and scipy.
"""

import logging
import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import generic_filter
from typing import Any
import geopandas as gpd

logger = logging.getLogger(__name__)


def terrain_analysis(
    dem_path: str, output_dir: str, analyses: list[str] | None = None
) -> dict[str, str]:
    """
    Perform comprehensive terrain analysis on a Digital Elevation Model.

    Args:
        dem_path: Path to input DEM raster
        output_dir: Directory for output files
        analyses: List of analyses to perform ('slope', 'aspect', 'hillshade', 'curvature', 'tpi')

    Returns:
        Dictionary mapping analysis type to output file path
    """
    if analyses is None:
        analyses = ["slope", "aspect", "hillshade"]

    results = {}

    with rasterio.open(dem_path) as src:
        elevation = src.read(1)
        profile = src.profile

        # Calculate gradients
        dy, dx = np.gradient(elevation, src.res[1], src.res[0])

        if "slope" in analyses:
            slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
            slope_path = f"{output_dir}/slope.tif"
            _write_raster(slope, slope_path, profile)
            results["slope"] = slope_path

        if "aspect" in analyses:
            aspect = np.arctan2(-dx, dy) * 180 / np.pi
            aspect = (aspect + 360) % 360  # Convert to 0-360 degrees
            aspect_path = f"{output_dir}/aspect.tif"
            _write_raster(aspect, aspect_path, profile)
            results["aspect"] = aspect_path

        if "hillshade" in analyses:
            # Default sun position (azimuth=315°, altitude=45°)
            azimuth = 315 * np.pi / 180
            altitude = 45 * np.pi / 180

            hillshade = np.sin(altitude) * np.cos(
                np.arctan(np.sqrt(dx**2 + dy**2))
            ) + np.cos(altitude) * np.sin(np.arctan(np.sqrt(dx**2 + dy**2))) * np.cos(
                azimuth - np.arctan2(-dx, dy)
            )

            hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)
            hillshade_path = f"{output_dir}/hillshade.tif"
            _write_raster(hillshade, hillshade_path, profile)
            results["hillshade"] = hillshade_path

        if "curvature" in analyses:
            # Calculate second derivatives for curvature
            dxx = np.gradient(dx, src.res[0], axis=1)
            dyy = np.gradient(dy, src.res[1], axis=0)
            dxy = np.gradient(dx, src.res[1], axis=0)

            # Profile curvature
            curvature = -(dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) / (
                dx**2 + dy**2
            ) ** (3 / 2)

            curvature_path = f"{output_dir}/curvature.tif"
            _write_raster(curvature, curvature_path, profile)
            results["curvature"] = curvature_path

        if "tpi" in analyses:
            # Topographic Position Index (mean elevation in neighborhood)
            kernel_size = 3
            mean_elevation = generic_filter(elevation, np.mean, size=kernel_size)
            tpi = elevation - mean_elevation

            tpi_path = f"{output_dir}/tpi.tif"
            _write_raster(tpi, tpi_path, profile)
            results["tpi"] = tpi_path

    logger.info(f"Terrain analysis completed: {len(results)} outputs")
    return results


def _evaluate_expression(expression: str, bands: dict[str, Any], nodata: float) -> Any:
    """Interpret a bounded arithmetic AST without exposing Python objects."""
    import ast
    import functools

    if (
        not isinstance(expression, str)
        or not expression.strip()
        or len(expression) > 4096
    ):
        raise ValueError("Invalid raster expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid raster expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Raster expression is too complex")
    binary = {
        ast.Add: np.add,
        ast.Sub: np.subtract,
        ast.Mult: np.multiply,
        ast.Div: np.divide,
        ast.Pow: np.power,
        ast.Mod: np.mod,
        ast.BitAnd: np.logical_and,
        ast.BitOr: np.logical_or,
    }
    unary = {
        ast.UAdd: np.positive,
        ast.USub: np.negative,
        ast.Invert: np.logical_not,
        ast.Not: np.logical_not,
    }
    comparisons = {
        ast.Eq: np.equal,
        ast.NotEq: np.not_equal,
        ast.Lt: np.less,
        ast.LtE: np.less_equal,
        ast.Gt: np.greater,
        ast.GtE: np.greater_equal,
    }
    functions = {
        name: getattr(np, name)
        for name in (
            "where",
            "sqrt",
            "log",
            "log10",
            "exp",
            "abs",
            "minimum",
            "maximum",
            "clip",
            "isfinite",
            "logical_and",
            "logical_or",
            "logical_not",
        )
    }

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            if abs(node.value) > 1e308:
                raise ValueError("Raster expression constant is out of range")
            return (
                np.float64(node.value) if type(node.value) is not bool else node.value
            )
        if isinstance(node, ast.Name) and node.id in bands:
            return bands[node.id]
        if isinstance(node, ast.Name) and node.id == "nodata":
            return nodata
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            op = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            return functools.reduce(op, (evaluate(value) for value in node.values))
        if isinstance(node, ast.Compare) and all(
            type(op) in comparisons for op in node.ops
        ):
            values = [evaluate(node.left), *(evaluate(v) for v in node.comparators)]
            return functools.reduce(
                np.logical_and,
                (
                    comparisons[type(op)](a, b)
                    for op, a, b in zip(node.ops, values, values[1:])
                ),
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in functions
            and not node.keywords
        ):
            arity = (
                3
                if node.func.attr in ("where", "clip")
                else 2
                if node.func.attr in ("minimum", "maximum", "logical_and", "logical_or")
                else 1
            )
            if len(node.args) != arity:
                raise ValueError(
                    "Raster expression function has unsupported argument count"
                )
            return functions[node.func.attr](*(evaluate(arg) for arg in node.args))
        raise ValueError(
            "Unsupported raster expression; only arithmetic and approved numpy functions are allowed"
        )

    try:
        with np.errstate(
            divide="ignore", invalid="ignore", over="ignore", under="ignore"
        ):
            return evaluate(tree)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Invalid raster expression arguments") from exc


def map_algebra(
    raster_paths: list[str],
    expression: str,
    output_path: str,
    nodata_value: float = -9999,
) -> str:
    """Evaluate arithmetic over aligned rasters, preserving missing-data masks.

    Expressions accept b1, b2, ... and nodata, numeric constants, arithmetic,
    comparisons, Boolean operators and np.where/sqrt/log/log10/exp/abs/minimum/
    maximum/clip/isfinite/logical_and/logical_or/logical_not. Attribute access,
    arbitrary calls, indexing and comprehensions are rejected.
    """
    if not raster_paths:
        raise ValueError("At least one raster path required")
    if not np.isfinite(nodata_value) or abs(nodata_value) > np.finfo(np.float32).max:
        raise ValueError("nodata_value must be representable as finite float32")
    bands: dict[str, Any] = {}
    profile = None
    invalid = None
    grid = None
    for index, path in enumerate(raster_paths, 1):
        with rasterio.open(path) as source:
            current = (source.shape, source.transform, source.crs)
            if grid is not None and grid != current:
                raise ValueError(
                    "Input rasters must be spatially aligned (shape, transform and CRS)"
                )
            grid = current
            band = source.read(1, masked=True).astype(np.float64)
            mask = np.ma.getmaskarray(band) | ~np.isfinite(band.data)
            invalid = mask if invalid is None else invalid | mask
            bands[f"b{index}"] = band.filled(np.nan)
            if profile is None:
                profile = source.profile.copy()
    assert profile is not None and invalid is not None and grid is not None
    value = _evaluate_expression(expression, bands, nodata_value)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.broadcast_to(
                np.asarray(value, dtype=np.float32), grid[0]
            ).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Raster expression must produce a numeric scalar or aligned grid"
        ) from exc
    result[invalid | ~np.isfinite(result)] = nodata_value
    profile.update(dtype="float32", nodata=nodata_value, count=1)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(result, 1)
    logger.info("Map algebra completed: %s", output_path)
    return output_path


def focal_statistics(
    raster_path: str,
    output_path: str,
    statistic: str = "mean",
    window_size: int = 3,
    circular: bool = False,
) -> str:
    """
    Calculate focal statistics for a raster.

    Args:
        raster_path: Input raster path
        output_path: Output raster path
        statistic: Statistic to calculate ('mean', 'sum', 'std', 'min', 'max', 'median')
        window_size: Size of focal window (odd number)
        circular: Whether to use circular window

    Returns:
        Path to output raster
    """
    stat_functions = {
        "mean": np.mean,
        "sum": np.sum,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "median": np.median,
    }

    if statistic not in stat_functions:
        raise ValueError(f"Statistic must be one of {list(stat_functions.keys())}")

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
        profile.update(dtype=rasterio.float32)

        # Create footprint for circular window if requested
        if circular:
            y, x = np.ogrid[
                -window_size // 2 : window_size // 2 + 1,
                -window_size // 2 : window_size // 2 + 1,
            ]
            footprint = x**2 + y**2 <= (window_size // 2) ** 2
        else:
            footprint = None

        # Apply focal function
        result = generic_filter(
            data, stat_functions[statistic], size=window_size, footprint=footprint
        )

        # Write output
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(result, 1)

    logger.info(f"Focal {statistic} completed: {output_path}")
    return output_path


def zonal_statistics(
    raster_path: str,
    zones_gdf: gpd.GeoDataFrame,
    statistics: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Calculate zonal statistics for raster values within polygon zones.

    Args:
        raster_path: Input raster path
        zones_gdf: GeoDataFrame with polygon zones
        statistics: List of statistics to calculate

    Returns:
        GeoDataFrame with zonal statistics
    """
    if statistics is None:
        statistics = ["mean", "sum", "count", "std", "min", "max"]

    result_gdf = zones_gdf.copy()

    with rasterio.open(raster_path) as src:
        # Ensure CRS match
        if zones_gdf.crs != src.crs:
            zones_gdf = zones_gdf.to_crs(src.crs)

        for idx, zone in zones_gdf.iterrows():
            try:
                # Mask raster to zone
                from rasterio.mask import mask

                masked_data, _ = mask(
                    src, [zone.geometry], crop=True, nodata=src.nodata
                )
                values = masked_data[masked_data != src.nodata]

                if len(values) > 0:
                    for stat in statistics:
                        if stat == "mean":
                            result_gdf.loc[idx, f"raster_{stat}"] = np.mean(values)
                        elif stat == "sum":
                            result_gdf.loc[idx, f"raster_{stat}"] = np.sum(values)
                        elif stat == "count":
                            result_gdf.loc[idx, f"raster_{stat}"] = len(values)
                        elif stat == "std":
                            result_gdf.loc[idx, f"raster_{stat}"] = np.std(values)
                        elif stat == "min":
                            result_gdf.loc[idx, f"raster_{stat}"] = np.min(values)
                        elif stat == "max":
                            result_gdf.loc[idx, f"raster_{stat}"] = np.max(values)
                else:
                    # No data in zone
                    for stat in statistics:
                        result_gdf.loc[idx, f"raster_{stat}"] = np.nan

            except Exception as e:
                logger.warning(f"Failed to process zone {idx}: {e}")
                for stat in statistics:
                    result_gdf.loc[idx, f"raster_{stat}"] = np.nan

    logger.info(f"Zonal statistics completed for {len(zones_gdf)} zones")
    return result_gdf


def raster_overlay(
    raster_paths: list[str],
    output_path: str,
    method: str = "sum",
    weights: list[float] | None = None,
) -> str:
    """
    Overlay multiple rasters using specified method.

    Args:
        raster_paths: List of input raster paths
        output_path: Output raster path
        method: Overlay method ('sum', 'mean', 'max', 'min', 'weighted_sum')
        weights: Weights for weighted operations

    Returns:
        Path to output raster
    """
    if not raster_paths:
        raise ValueError("At least one raster required")

    if method == "weighted_sum" and (not weights or len(weights) != len(raster_paths)):
        raise ValueError(
            "Weights must be provided and match number of rasters for weighted_sum"
        )

    # Read first raster to get profile
    with rasterio.open(raster_paths[0]) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32)
        result = src.read(1).astype(np.float32)

    # Process remaining rasters
    for i, path in enumerate(raster_paths[1:], 1):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)

            if method == "sum":
                result += data
            elif method == "mean":
                result = (result * i + data) / (i + 1)
            elif method == "max":
                result = np.maximum(result, data)
            elif method == "min":
                result = np.minimum(result, data)
            elif method == "weighted_sum":
                assert weights is not None
                if i == 1:  # Reset for weighted sum
                    result = result * weights[0] + data * weights[1]
                else:
                    result += data * weights[i]

    # Apply initial weight for a single weighted raster.
    if method == "weighted_sum" and len(raster_paths) == 1:
        assert weights is not None
        result *= weights[0]

    # Write output
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result, 1)

    logger.info(f"Raster overlay ({method}) completed: {output_path}")
    return output_path


def image_processing(
    raster_path: str, output_path: str, operation: str, **kwargs: Any
) -> str:
    """
    Perform image processing operations on raster data.

    Args:
        raster_path: Input raster path
        output_path: Output raster path
        operation: Processing operation ('gaussian_filter', 'median_filter', 'edge_detection', 'histogram_equalization')
        **kwargs: Additional parameters for specific operations

    Returns:
        Path to output raster
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile

        if operation == "gaussian_filter":
            sigma = kwargs.get("sigma", 1.0)
            result = ndimage.gaussian_filter(data, sigma=sigma)

        elif operation == "median_filter":
            size = kwargs.get("size", 3)
            result = ndimage.median_filter(data, size=size)

        elif operation == "edge_detection":
            # Sobel edge detection
            sobel_x = ndimage.sobel(data, axis=1)
            sobel_y = ndimage.sobel(data, axis=0)
            result = np.sqrt(sobel_x**2 + sobel_y**2)

        elif operation == "histogram_equalization":
            # Simple histogram equalization
            hist, bins = np.histogram(data.flatten(), bins=256)
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            result = np.interp(data.flatten(), bins[:-1], cdf_normalized).reshape(
                data.shape
            )

        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Write output
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(result.astype(profile["dtype"]), 1)

    logger.info(f"Image processing ({operation}) completed: {output_path}")
    return output_path


def _write_raster(data: np.ndarray, output_path: str, profile: dict[str, Any]) -> None:
    """Helper function to write raster data."""
    profile_copy = profile.copy()
    profile_copy.update(dtype=data.dtype)

    with rasterio.open(output_path, "w", **profile_copy) as dst:
        dst.write(data, 1)
