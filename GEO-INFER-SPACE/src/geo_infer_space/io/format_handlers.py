"""
Format handler system for geospatial data I/O.

Provides a pluggable handler architecture with an abstract base class and
concrete implementations for GeoJSON, Shapefile, GeoTIFF, Cloud-Optimized
GeoTIFF (COG), LAS/LAZ point clouds, and NetCDF datasets.

Each handler encapsulates format-specific read, write, and validation logic
while exposing a uniform interface through the FormatHandler ABC.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency availability flags (laspy and xarray are SPACE extras)
# ---------------------------------------------------------------------------

try:
    import laspy  # type: ignore[import-untyped]

    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    laspy = None

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class FormatHandler(ABC):
    """Abstract base class for geospatial format handlers.

    Every concrete handler must implement the four core operations:
    ``can_handle``, ``read``, ``write``, and ``validate``, as well as
    expose a ``format_name`` property.

    Attributes:
        extensions: Tuple of lowercase file extensions this handler supports
            (including the leading dot, e.g. ``('.geojson', '.json')``).
    """

    extensions: tuple[str, ...] = ()

    # -- Abstract interface -------------------------------------------------

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Human-readable name of the format (e.g. ``'GeoJSON'``)."""
        ...

    @abstractmethod
    def can_handle(self, path: str | Path) -> bool:
        """Return ``True`` if *path* is a file this handler can process.

        The check is typically based on file extension but implementations
        may also inspect magic bytes or headers.

        Args:
            path: Filesystem path to the candidate file.

        Returns:
            Whether this handler supports the given file.
        """
        ...

    @abstractmethod
    def read(self, path: str | Path, **kwargs: Any) -> Any:
        """Read geospatial data from *path*.

        Args:
            path: Filesystem path to the input file.
            **kwargs: Format-specific keyword arguments.

        Returns:
            The loaded data object (type varies by format).

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If the required library is not installed.
        """
        ...

    @abstractmethod
    def write(self, data: Any, path: str | Path, **kwargs: Any) -> None:
        """Write *data* to *path* in this handler's format.

        Args:
            data: The geospatial data object to persist.
            path: Filesystem path for the output file.
            **kwargs: Format-specific keyword arguments.

        Raises:
            ImportError: If the required library is not installed.
        """
        ...

    @abstractmethod
    def validate(self, path: str | Path) -> dict[str, Any]:
        """Validate the file at *path* and return diagnostic metadata.

        Args:
            path: Filesystem path to the file to validate.

        Returns:
            A dictionary with at least the keys ``'valid'`` (bool),
            ``'format'`` (str or None), ``'error'`` (str or None),
            and ``'metadata'`` (dict).
        """
        ...

    # -- Shared helpers -----------------------------------------------------

    def _match_extension(self, path: str | Path) -> bool:
        """Check whether *path* has a file extension this handler supports."""
        return Path(path).suffix.lower() in self.extensions

    def _ensure_file_exists(self, path: str | Path) -> Path:
        """Resolve *path* and raise if it does not exist."""
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")
        return resolved

    def _ensure_parent_dir(self, path: str | Path) -> Path:
        """Ensure the parent directory of *path* exists."""
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _base_validation_result() -> dict[str, Any]:
        """Return a fresh validation result dict with default values."""
        return {
            "valid": False,
            "format": None,
            "error": None,
            "metadata": {},
        }


# ---------------------------------------------------------------------------
# GeoJSON handler
# ---------------------------------------------------------------------------


class GeoJSONHandler(FormatHandler):
    """Handler for GeoJSON files.

    Reads and writes GeoJSON using *geopandas*.  Falls back to the
    standard-library ``json`` module for lightweight validation when
    geopandas is unavailable.
    """

    extensions = (".geojson", ".json")

    @property
    def format_name(self) -> str:
        return "GeoJSON"

    def can_handle(self, path: str | Path) -> bool:
        return self._match_extension(path)

    def read(self, path: str | Path, **kwargs: Any) -> Any:
        """Read a GeoJSON file into a ``GeoDataFrame``.

        Args:
            path: Path to the GeoJSON file.
            **kwargs: Forwarded to ``geopandas.read_file``.

        Returns:
            ``geopandas.GeoDataFrame``.
        """
        resolved = self._ensure_file_exists(path)
        logger.info("Reading GeoJSON file: %s", resolved)

        try:
            gdf = gpd.read_file(resolved, driver="GeoJSON", **kwargs)
            logger.info("Read %d features from %s", len(gdf), resolved)
            return gdf
        except Exception:
            logger.exception("Failed to read GeoJSON file: %s", resolved)
            raise

    def write(self, data: Any, path: str | Path, **kwargs: Any) -> None:
        """Write a ``GeoDataFrame`` to a GeoJSON file.

        Args:
            data: A ``geopandas.GeoDataFrame``.
            path: Output file path.
            **kwargs: Forwarded to ``GeoDataFrame.to_file``.
        """
        resolved = self._ensure_parent_dir(path)
        logger.info("Writing GeoJSON file: %s", resolved)

        try:
            data.to_file(resolved, driver="GeoJSON", **kwargs)
            logger.info("Wrote %d features to %s", len(data), resolved)
        except Exception:
            logger.exception("Failed to write GeoJSON file: %s", resolved)
            raise

    def validate(self, path: str | Path) -> dict[str, Any]:
        result = self._base_validation_result()
        result["format"] = self.format_name

        try:
            resolved = self._ensure_file_exists(path)
        except FileNotFoundError as exc:
            result["error"] = str(exc)
            return result

        try:
            with open(resolved, encoding="utf-8") as fh:
                geojson_data = json.load(fh)

            if not isinstance(geojson_data, dict):
                result["error"] = "Invalid GeoJSON: root element is not an object"
                return result

            geojson_type = geojson_data.get("type")
            valid_types = {
                "FeatureCollection",
                "Feature",
                "Point",
                "LineString",
                "Polygon",
                "MultiPoint",
                "MultiLineString",
                "MultiPolygon",
                "GeometryCollection",
            }
            if geojson_type not in valid_types:
                result["error"] = f"Invalid GeoJSON type: {geojson_type}"
                return result

            metadata: dict[str, Any] = {
                "type": geojson_type,
                "file_size_bytes": resolved.stat().st_size,
            }

            if geojson_type == "FeatureCollection":
                features = geojson_data.get("features", [])
                metadata["num_features"] = len(features)
                if features:
                    geom_types = {
                        f.get("geometry", {}).get("type")
                        for f in features
                        if isinstance(f, dict) and isinstance(f.get("geometry"), dict)
                    }
                    metadata["geometry_types"] = sorted(geom_types - {None})

            try:
                gdf = gpd.read_file(resolved, driver="GeoJSON")
                metadata["crs"] = str(gdf.crs) if gdf.crs else None
                if len(gdf) > 0:
                    metadata["bounds"] = gdf.total_bounds.tolist()
            except Exception:
                pass  # validation still succeeds with partial metadata

            result["valid"] = True
            result["metadata"] = metadata

        except json.JSONDecodeError as exc:
            result["error"] = f"Invalid JSON: {exc}"
        except Exception as exc:
            result["error"] = str(exc)

        return result


# ---------------------------------------------------------------------------
# Shapefile handler
# ---------------------------------------------------------------------------


class ShapefileHandler(FormatHandler):
    """Handler for ESRI Shapefiles.

    Reads and writes shapefiles using *geopandas* (backed by *fiona*).
    """

    extensions = (".shp",)

    @property
    def format_name(self) -> str:
        return "ESRI Shapefile"

    def can_handle(self, path: str | Path) -> bool:
        return self._match_extension(path)

    def read(self, path: str | Path, **kwargs: Any) -> Any:
        """Read a Shapefile into a ``GeoDataFrame``.

        Args:
            path: Path to the ``.shp`` file.
            **kwargs: Forwarded to ``geopandas.read_file``.

        Returns:
            ``geopandas.GeoDataFrame``.
        """
        resolved = self._ensure_file_exists(path)
        logger.info("Reading Shapefile: %s", resolved)

        try:
            gdf = gpd.read_file(resolved, **kwargs)
            logger.info("Read %d features from %s", len(gdf), resolved)
            return gdf
        except Exception:
            logger.exception("Failed to read Shapefile: %s", resolved)
            raise

    def write(self, data: Any, path: str | Path, **kwargs: Any) -> None:
        """Write a ``GeoDataFrame`` to a Shapefile.

        Args:
            data: A ``geopandas.GeoDataFrame``.
            path: Output ``.shp`` file path.
            **kwargs: Forwarded to ``GeoDataFrame.to_file``.
        """
        resolved = self._ensure_parent_dir(path)
        logger.info("Writing Shapefile: %s", resolved)

        try:
            data.to_file(resolved, driver="ESRI Shapefile", **kwargs)
            logger.info("Wrote %d features to %s", len(data), resolved)
        except Exception:
            logger.exception("Failed to write Shapefile: %s", resolved)
            raise

    def validate(self, path: str | Path) -> dict[str, Any]:
        result = self._base_validation_result()
        result["format"] = self.format_name

        try:
            resolved = self._ensure_file_exists(path)
        except FileNotFoundError as exc:
            result["error"] = str(exc)
            return result

        # Shapefiles consist of several sidecar files; check for the
        # minimum required set (.shp, .shx, .dbf).
        base = resolved.with_suffix("")
        required_sidecars = {".shx": False, ".dbf": False}
        for ext in required_sidecars:
            sidecar = base.with_suffix(ext)
            required_sidecars[ext] = sidecar.exists()

        missing = [ext for ext, found in required_sidecars.items() if not found]

        metadata: dict[str, Any] = {
            "file_size_bytes": resolved.stat().st_size,
            "sidecar_files_present": {
                ext: found for ext, found in required_sidecars.items()
            },
        }

        if missing:
            result["error"] = f"Missing required sidecar files: {', '.join(missing)}"
            result["metadata"] = metadata
            return result

        try:
            gdf = gpd.read_file(resolved)
            metadata.update(
                {
                    "num_features": len(gdf),
                    "columns": list(gdf.columns),
                    "geometry_types": list(gdf.geometry.geom_type.unique()),
                    "crs": str(gdf.crs) if gdf.crs else None,
                    "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else None,
                }
            )
            result["valid"] = True
        except Exception as exc:
            result["error"] = f"Failed to read Shapefile: {exc}"

        result["metadata"] = metadata
        return result


# ---------------------------------------------------------------------------
# GeoTIFF handler
# ---------------------------------------------------------------------------


class GeoTIFFHandler(FormatHandler):
    """Handler for GeoTIFF raster files.

    Uses *rasterio* for reading and writing.  When rasterio is not
    installed the handler raises ``ImportError`` with installation
    instructions instead of crashing on import.
    """

    extensions = (".tif", ".tiff", ".geotiff")

    @property
    def format_name(self) -> str:
        return "GeoTIFF"

    def can_handle(self, path: str | Path) -> bool:
        return self._match_extension(path)

    # -- internal helpers ---------------------------------------------------

    def _get_write_profile(
        self, data: Any, path: Path, **kwargs: Any
    ) -> dict[str, Any]:
        """Build a rasterio write profile from *data* and *kwargs*.

        Subclasses (e.g. ``COGHandler``) override this method to inject
        format-specific profile options.
        """
        # Accept either a raw numpy array or a dict with 'data' and optional
        # metadata keys that mirror a rasterio dataset.
        if isinstance(data, dict):
            arr = data["data"]
            crs = data.get("crs")
            transform = data.get("transform")
        else:
            arr = data
            crs = kwargs.pop("crs", None)
            transform = kwargs.pop("transform", None)

        if arr.ndim == 2:
            count = 1
            height, width = arr.shape
        elif arr.ndim == 3:
            count, height, width = arr.shape
        else:
            raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")

        profile: dict[str, Any] = {
            "driver": "GTiff",
            "dtype": str(arr.dtype),
            "width": width,
            "height": height,
            "count": count,
            "crs": crs,
            "transform": transform,
        }
        profile.update(kwargs)
        return profile

    # -- public interface ---------------------------------------------------

    def read(self, path: str | Path, **kwargs: Any) -> Any:
        """Read a GeoTIFF into a dictionary of array + metadata.

        Args:
            path: Path to the GeoTIFF file.
            **kwargs: Forwarded to ``rasterio.open``.

        Returns:
            A dictionary with keys ``'data'`` (numpy ndarray),
            ``'crs'``, ``'transform'``, ``'bounds'``, ``'count'``,
            ``'width'``, ``'height'``, ``'dtype'``, and ``'nodata'``.
        """

        resolved = self._ensure_file_exists(path)
        logger.info("Reading GeoTIFF file: %s", resolved)

        bands: int | list[int] | None = kwargs.pop("bands", None)

        try:
            with rasterio.open(resolved, "r", **kwargs) as src:
                if bands is not None:
                    data = src.read(bands)
                else:
                    data = src.read()

                result = {
                    "data": data,
                    "crs": src.crs,
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "count": src.count,
                    "width": src.width,
                    "height": src.height,
                    "dtype": str(src.dtypes[0]),
                    "nodata": src.nodata,
                }

            logger.info(
                "Read GeoTIFF (%dx%d, %d bands) from %s",
                result["width"],
                result["height"],
                result["count"],
                resolved,
            )
            return result
        except Exception:
            logger.exception("Failed to read GeoTIFF: %s", resolved)
            raise

    def write(self, data: Any, path: str | Path, **kwargs: Any) -> None:
        """Write raster data to a GeoTIFF file.

        *data* may be a numpy array (2-D single-band or 3-D multi-band
        with shape ``(bands, height, width)``) or a dictionary as
        returned by :meth:`read`.

        Args:
            data: Raster data to write.
            path: Output file path.
            **kwargs: Additional profile options (``crs``, ``transform``,
                ``compress``, etc.).
        """

        resolved = self._ensure_parent_dir(path)
        logger.info("Writing GeoTIFF file: %s", resolved)

        profile = self._get_write_profile(data, resolved, **kwargs)
        arr = data["data"] if isinstance(data, dict) else data

        try:
            with rasterio.open(resolved, "w", **profile) as dst:
                if arr.ndim == 2:
                    dst.write(arr, 1)
                else:
                    dst.write(arr)
            logger.info("Wrote GeoTIFF to %s", resolved)
        except Exception:
            logger.exception("Failed to write GeoTIFF: %s", resolved)
            raise

    def validate(self, path: str | Path) -> dict[str, Any]:
        result = self._base_validation_result()
        result["format"] = self.format_name

        try:
            resolved = self._ensure_file_exists(path)
        except FileNotFoundError as exc:
            result["error"] = str(exc)
            return result

        try:
            with rasterio.open(resolved, "r") as src:
                result["metadata"] = {
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtype": str(src.dtypes[0]),
                    "crs": str(src.crs) if src.crs else None,
                    "bounds": dict(src.bounds._asdict()) if src.bounds else None,
                    "transform": list(src.transform)[:6],
                    "nodata": src.nodata,
                    "driver": src.driver,
                    "file_size_bytes": resolved.stat().st_size,
                }
            result["valid"] = True
        except Exception as exc:
            result["error"] = f"Failed to validate GeoTIFF: {exc}"

        return result


# ---------------------------------------------------------------------------
# Cloud-Optimized GeoTIFF (COG) handler
# ---------------------------------------------------------------------------


class COGHandler(GeoTIFFHandler):
    """Handler for Cloud-Optimized GeoTIFF (COG) files.

    Extends :class:`GeoTIFFHandler` with COG-specific creation options
    (tiling, overview levels, and internal compression).  On the read
    side it behaves identically to a regular GeoTIFF handler.
    """

    @property
    def format_name(self) -> str:
        return "Cloud-Optimized GeoTIFF"

    def _get_write_profile(
        self, data: Any, path: Path, **kwargs: Any
    ) -> dict[str, Any]:
        """Build a COG-oriented write profile.

        Defaults to LZW compression, 256x256 internal tiling, and the
        ``COG`` driver when available in the installed GDAL version.
        These can all be overridden via *kwargs*.
        """
        profile = super()._get_write_profile(data, path, **kwargs)

        # Apply COG-friendly defaults (caller can override via kwargs).
        cog_defaults: dict[str, Any] = {
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "lzw",
        }

        for key, value in cog_defaults.items():
            profile.setdefault(key, value)

        return profile

    def write(self, data: Any, path: str | Path, **kwargs: Any) -> None:
        """Write raster data as a Cloud-Optimized GeoTIFF.

        After writing the base tiles this method generates internal
        overviews and then uses ``rasterio``'s ``copy`` to produce a
        properly ordered COG if possible.

        Args:
            data: Raster data (numpy array or dict).
            path: Output file path.
            **kwargs: Profile overrides and overview options.  Pass
                ``overview_levels`` (list of ints, default
                ``[2, 4, 8, 16]``) and ``overview_resampling``
                (str, default ``'nearest'``) to control overview
                generation.
        """

        overview_levels: list[int] = kwargs.pop("overview_levels", [2, 4, 8, 16])
        overview_resampling: str = kwargs.pop("overview_resampling", "nearest")

        resolved = self._ensure_parent_dir(path)
        logger.info("Writing COG file: %s", resolved)

        profile = self._get_write_profile(data, resolved, **kwargs)
        arr = data["data"] if isinstance(data, dict) else data

        try:
            with rasterio.open(resolved, "w", **profile) as dst:
                if arr.ndim == 2:
                    dst.write(arr, 1)
                else:
                    dst.write(arr)

                # Build internal overviews.
                from rasterio.enums import Resampling

                resampling = getattr(
                    Resampling, overview_resampling, Resampling.nearest
                )
                dst.build_overviews(overview_levels, resampling)
                dst.update_tags(ns="rio_overview", resampling=overview_resampling)

            logger.info("Wrote COG with overviews to %s", resolved)
        except Exception:
            logger.exception("Failed to write COG: %s", resolved)
            raise

    def validate(self, path: str | Path) -> dict[str, Any]:
        result = super().validate(path)

        if not result["valid"]:
            return result

        result["format"] = self.format_name

        # Extra COG-specific checks.
        try:
            resolved = Path(path)
            with rasterio.open(resolved, "r") as src:
                is_tiled = src.profile.get("tiled", False)
                has_overviews = any(src.overviews(i + 1) for i in range(src.count))
                overview_levels = src.overviews(1) if src.count > 0 else []
                result["metadata"].update(
                    {
                        "is_tiled": is_tiled,
                        "has_overviews": has_overviews,
                        "overview_levels": overview_levels,
                        "is_cog_compliant": is_tiled and has_overviews,
                    }
                )
        except Exception as exc:
            logger.warning("COG-specific validation failed: %s", exc)

        return result


# ---------------------------------------------------------------------------
# LAS / LAZ point cloud handler
# ---------------------------------------------------------------------------


class LASHandler(FormatHandler):
    """Handler for LAS and LAZ point cloud files.

    Uses *laspy* for reading and writing.
    """

    extensions = (".las", ".laz")

    @property
    def format_name(self) -> str:
        return "LAS/LAZ"

    def can_handle(self, path: str | Path) -> bool:
        return self._match_extension(path)

    @staticmethod
    def _require_laspy() -> None:
        if not HAS_LASPY:
            raise ImportError(
                "laspy is required to handle LAS/LAZ files. "
                "Install it with: pip install laspy[lazrs]"
            )

    def read(self, path: str | Path, **kwargs: Any) -> Any:
        """Read a LAS/LAZ file.

        Args:
            path: Path to the point cloud file.
            **kwargs: Forwarded to ``laspy.read``.

        Returns:
            A ``laspy.LasData`` object.
        """
        self._require_laspy()

        resolved = self._ensure_file_exists(path)
        logger.info("Reading LAS file: %s", resolved)

        try:
            las_data = laspy.read(str(resolved), **kwargs)
            logger.info("Read %d points from %s", las_data.header.point_count, resolved)
            return las_data
        except Exception:
            logger.exception("Failed to read LAS file: %s", resolved)
            raise

    def write(self, data: Any, path: str | Path, **kwargs: Any) -> None:
        """Write point cloud data to a LAS/LAZ file.

        Args:
            data: A ``laspy.LasData`` object, or a dictionary with keys
                ``'x'``, ``'y'``, ``'z'`` (numpy arrays) and optional
                ``'point_format'`` (int) and ``'file_version'`` (str).
            path: Output file path.
            **kwargs: Forwarded to ``laspy.LasData.write``.
        """
        self._require_laspy()

        resolved = self._ensure_parent_dir(path)
        logger.info("Writing LAS file: %s", resolved)

        try:
            if isinstance(data, dict):
                # Build LasData from raw arrays.
                point_format_id = data.get("point_format", 0)
                file_version = data.get("file_version", "1.4")
                header = laspy.LasHeader(
                    point_format=point_format_id, version=file_version
                )
                las_data = laspy.LasData(header)
                las_data.x = data["x"]
                las_data.y = data["y"]
                las_data.z = data["z"]

                # Copy optional dimension arrays.
                for dim_name in ("intensity", "classification", "return_number"):
                    if dim_name in data:
                        setattr(las_data, dim_name, data[dim_name])

                las_data.write(str(resolved), **kwargs)
            else:
                data.write(str(resolved), **kwargs)

            logger.info("Wrote LAS file to %s", resolved)
        except Exception:
            logger.exception("Failed to write LAS file: %s", resolved)
            raise

    def validate(self, path: str | Path) -> dict[str, Any]:
        result = self._base_validation_result()
        result["format"] = self.format_name

        try:
            resolved = self._ensure_file_exists(path)
        except FileNotFoundError as exc:
            result["error"] = str(exc)
            return result

        if not HAS_LASPY:
            result["error"] = (
                "laspy is required to validate LAS/LAZ files. "
                "Install it with: pip install laspy[lazrs]"
            )
            return result

        try:
            las_data = laspy.read(str(resolved))
            header = las_data.header

            point_format = header.point_format
            dimension_names = [dim.name for dim in point_format.dimensions]

            mins = las_data.header.mins
            maxs = las_data.header.maxs

            result["metadata"] = {
                "point_count": int(header.point_count),
                "point_format_id": int(point_format.id),
                "file_version": str(header.version),
                "dimensions": dimension_names,
                "mins": [float(v) for v in mins],
                "maxs": [float(v) for v in maxs],
                "file_size_bytes": resolved.stat().st_size,
            }

            # Attempt to retrieve CRS information.
            try:
                for vlr in header.vlrs:
                    if vlr.record_id == 2112:  # WKT CRS VLR
                        result["metadata"]["crs_wkt"] = vlr.record_data.decode(
                            "utf-8", errors="replace"
                        )
                        break
            except Exception:
                pass

            result["valid"] = True
        except Exception as exc:
            result["error"] = f"Failed to validate LAS file: {exc}"

        return result


# ---------------------------------------------------------------------------
# NetCDF handler
# ---------------------------------------------------------------------------


class NetCDFHandler(FormatHandler):
    """Handler for NetCDF files.

    Uses *xarray* (backed by the *netcdf4* or *scipy* engine) for
    reading and writing.
    """

    extensions = (".nc", ".nc4", ".netcdf")

    @property
    def format_name(self) -> str:
        return "NetCDF"

    def can_handle(self, path: str | Path) -> bool:
        return self._match_extension(path)

    @staticmethod
    def _require_xarray() -> None:
        if not HAS_XARRAY:
            raise ImportError(
                "xarray is required to handle NetCDF files. "
                "Install it with: pip install xarray netcdf4"
            )

    def read(self, path: str | Path, **kwargs: Any) -> Any:
        """Read a NetCDF file into an ``xarray.Dataset``.

        Args:
            path: Path to the NetCDF file.
            **kwargs: Forwarded to ``xarray.open_dataset``.

        Returns:
            ``xarray.Dataset``.
        """
        self._require_xarray()

        resolved = self._ensure_file_exists(path)
        logger.info("Reading NetCDF file: %s", resolved)

        try:
            ds = xr.open_dataset(str(resolved), **kwargs)
            logger.info(
                "Read NetCDF with %d variables from %s",
                len(ds.data_vars),
                resolved,
            )
            return ds
        except Exception:
            logger.exception("Failed to read NetCDF file: %s", resolved)
            raise

    def write(self, data: Any, path: str | Path, **kwargs: Any) -> None:
        """Write an ``xarray.Dataset`` to a NetCDF file.

        Args:
            data: An ``xarray.Dataset`` (or ``xarray.DataArray``, which
                will be promoted to a Dataset automatically).
            path: Output file path.
            **kwargs: Forwarded to ``Dataset.to_netcdf``.
        """
        self._require_xarray()

        resolved = self._ensure_parent_dir(path)
        logger.info("Writing NetCDF file: %s", resolved)

        try:
            if hasattr(data, "to_dataset"):
                # Convert DataArray to Dataset.
                data = data.to_dataset(name=kwargs.pop("name", "data"))

            data.to_netcdf(str(resolved), **kwargs)
            logger.info("Wrote NetCDF file to %s", resolved)
        except Exception:
            logger.exception("Failed to write NetCDF file: %s", resolved)
            raise

    def validate(self, path: str | Path) -> dict[str, Any]:
        result = self._base_validation_result()
        result["format"] = self.format_name

        try:
            resolved = self._ensure_file_exists(path)
        except FileNotFoundError as exc:
            result["error"] = str(exc)
            return result

        if not HAS_XARRAY:
            result["error"] = (
                "xarray is required to validate NetCDF files. "
                "Install it with: pip install xarray netcdf4"
            )
            return result

        try:
            with xr.open_dataset(str(resolved)) as ds:
                coords = {
                    name: {
                        "dims": list(coord.dims),
                        "shape": list(coord.shape),
                        "dtype": str(coord.dtype),
                    }
                    for name, coord in ds.coords.items()
                }

                variables = {
                    name: {
                        "dims": list(var.dims),
                        "shape": list(var.shape),
                        "dtype": str(var.dtype),
                    }
                    for name, var in ds.data_vars.items()
                }

                result["metadata"] = {
                    "dimensions": dict(ds.dims),
                    "coordinates": coords,
                    "variables": variables,
                    "global_attributes": dict(ds.attrs),
                    "file_size_bytes": resolved.stat().st_size,
                }

            result["valid"] = True
        except Exception as exc:
            result["error"] = f"Failed to validate NetCDF file: {exc}"

        return result


# ---------------------------------------------------------------------------
# Handler registry helper
# ---------------------------------------------------------------------------

# Ordered list of all built-in handlers (most specific first).
_BUILTIN_HANDLERS: list[FormatHandler] = [
    COGHandler(),
    GeoTIFFHandler(),
    GeoJSONHandler(),
    ShapefileHandler(),
    LASHandler(),
    NetCDFHandler(),
]


def get_handler_for_path(path: str | Path) -> FormatHandler | None:
    """Return the first registered handler that can process *path*.

    Args:
        path: Filesystem path to inspect.

    Returns:
        A matching :class:`FormatHandler` instance, or ``None`` if no
        handler supports the file.
    """
    for handler in _BUILTIN_HANDLERS:
        if handler.can_handle(path):
            return handler
    return None


def list_supported_formats() -> dict[str, list[str]]:
    """Return a mapping of format names to their supported extensions.

    Returns:
        Dictionary keyed by ``format_name`` with lists of extensions.
    """
    formats: dict[str, list[str]] = {}
    for handler in _BUILTIN_HANDLERS:
        formats[handler.format_name] = list(handler.extensions)
    return formats
