"""
Raster data I/O operations for various geospatial formats.

This module provides reading and writing capabilities for raster data
including GeoTIFF, Cloud Optimized GeoTIFF (COG), NetCDF, HDF5,
JPEG2000, PNG, and JPEG formats using rasterio.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine, from_bounds
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning(
        "rasterio is not installed. Raster I/O functionality will be limited. "
        "Install with: pip install rasterio"
    )

# Supported raster formats
SUPPORTED_RASTER_FORMATS = {
    ".tif": "GeoTIFF",
    ".tiff": "GeoTIFF",
    ".cog": "COG",
    ".nc": "NetCDF",
    ".hdf": "HDF5",
    ".h5": "HDF5",
    ".hdf5": "HDF5",
    ".jp2": "JPEG2000",
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
}

# Map format names to rasterio driver names
FORMAT_DRIVERS = {
    "GeoTIFF": "GTiff",
    "COG": "COG",
    "NetCDF": "netCDF",
    "HDF5": "HDF5",
    "JPEG2000": "JP2OpenJPEG",
    "PNG": "PNG",
    "JPEG": "JPEG",
}


def _check_rasterio() -> None:
    """Raise ImportError if rasterio is not available."""
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for raster I/O operations. "
            "Install with: pip install rasterio"
        )


class RasterReader:
    """Reader class for raster geospatial data."""

    def __init__(self) -> None:
        self.supported_formats = SUPPORTED_RASTER_FORMATS

    def read(
        self,
        file_path: str | Path,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
        masked: bool = True,
        overview_level: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Read raster data from file.

        Args:
            file_path: Path to raster data file.
            bands: List of 1-based band indices to read. None reads all bands.
            window: Optional read window as (col_off, row_off, width, height).
            masked: If True, return a numpy masked array honoring nodata values.
            overview_level: If set, read from the specified overview level.
            **kwargs: Additional parameters passed to rasterio.open.

        Returns:
            Dictionary with keys:
                - data: numpy ndarray of shape (bands, height, width)
                - metadata: dict of rasterio profile / tags
                - crs: coordinate reference system as a string
                - transform: affine transform as a tuple
                - bounds: bounding box as (left, bottom, right, top)
                - nodata: nodata value or None
                - dtype: numpy dtype string for the data
                - shape: tuple (bands, height, width)

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If file format is not supported.
            FileNotFoundError: If file does not exist.
        """
        _check_rasterio()

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported raster format: {file_ext}")

        try:
            format_name = self.supported_formats[file_ext]

            if format_name == "NetCDF":
                return self._read_netcdf(
                    file_path, bands=bands, masked=masked, **kwargs
                )
            elif format_name == "HDF5":
                return self._read_hdf5(file_path, bands=bands, masked=masked, **kwargs)
            else:
                return self._read_standard(
                    file_path,
                    bands=bands,
                    window=window,
                    masked=masked,
                    overview_level=overview_level,
                    **kwargs,
                )

        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Failed to read raster file {file_path}: {e}")
            raise

    def _read_standard(
        self,
        file_path: Path,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
        masked: bool = True,
        overview_level: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Read raster using standard rasterio open."""
        read_window = None
        if window is not None:
            col_off, row_off, width, height = window
            read_window = Window(col_off, row_off, width, height)

        with rasterio.open(file_path, **kwargs) as src:
            # Determine which bands to read
            if bands is None:
                bands = list(range(1, src.count + 1))

            # Handle overview level
            if overview_level is not None and src.overviews(bands[0]):
                overviews = src.overviews(bands[0])
                if overview_level < len(overviews):
                    decimation = overviews[overview_level]
                    if read_window is not None:
                        out_shape = (
                            len(bands),
                            max(1, read_window.height // decimation),
                            max(1, read_window.width // decimation),
                        )
                    else:
                        out_shape = (
                            len(bands),
                            max(1, src.height // decimation),
                            max(1, src.width // decimation),
                        )
                    data = src.read(
                        bands,
                        window=read_window,
                        out_shape=out_shape,
                        masked=masked,
                        resampling=Resampling.nearest,
                    )
                else:
                    logger.warning(
                        f"Overview level {overview_level} not available. "
                        f"Reading at full resolution."
                    )
                    data = src.read(bands, window=read_window, masked=masked)
            else:
                data = src.read(bands, window=read_window, masked=masked)

            # Build transform for the window if applicable
            transform = src.transform
            if read_window is not None:
                transform = src.window_transform(read_window)

            # Collect tags and profile as metadata
            metadata = dict(src.profile)
            metadata["tags"] = src.tags()
            metadata["descriptions"] = list(src.descriptions)

            # Convert non-serialisable objects in profile to strings
            if "crs" in metadata:
                metadata["crs"] = str(metadata["crs"])
            if "transform" in metadata:
                metadata["transform"] = tuple(metadata["transform"])

            result = {
                "data": np.asarray(data) if not masked else data,
                "metadata": metadata,
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(transform),
                "bounds": (
                    src.bounds
                    if read_window is None
                    else rasterio.windows.bounds(read_window, src.transform)
                ),
                "nodata": src.nodata,
                "dtype": str(src.dtypes[0]),
                "shape": data.shape,
            }

            return result

    def _read_netcdf(
        self,
        file_path: Path,
        bands: list[int] | None = None,
        masked: bool = True,
        subdataset: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Read NetCDF raster data.

        NetCDF files can contain multiple subdatasets. If *subdataset* is
        provided, open that specific subdataset directly. Otherwise, try
        opening the file as a single rasterio dataset; if subdatasets are
        detected, read the first one and log the available options.
        """
        open_path: str | Path = file_path

        if subdataset is not None:
            open_path = f"netcdf:{file_path}:{subdataset}"
        else:
            # Probe for subdatasets
            with rasterio.open(file_path) as probe:
                subdatasets = probe.subdatasets
                if subdatasets:
                    logger.info(
                        f"NetCDF contains {len(subdatasets)} subdatasets: "
                        f"{subdatasets}. Reading the first one."
                    )
                    open_path = subdatasets[0]

        return self._read_standard(
            open_path,  # type: ignore[arg-type]
            bands=bands,
            masked=masked,
            **kwargs,
        )

    def _read_hdf5(
        self,
        file_path: Path,
        bands: list[int] | None = None,
        masked: bool = True,
        subdataset: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Read HDF5 raster data.

        HDF5 files can contain multiple subdatasets. Behaviour mirrors the
        NetCDF reader: honour *subdataset* if given, otherwise probe and
        fall back to the first subdataset.
        """
        open_path: str | Path = file_path

        if subdataset is not None:
            open_path = f"HDF5:{file_path}:{subdataset}"
        else:
            with rasterio.open(file_path) as probe:
                subdatasets = probe.subdatasets
                if subdatasets:
                    logger.info(
                        f"HDF5 contains {len(subdatasets)} subdatasets: "
                        f"{subdatasets}. Reading the first one."
                    )
                    open_path = subdatasets[0]

        return self._read_standard(
            open_path,  # type: ignore[arg-type]
            bands=bands,
            masked=masked,
            **kwargs,
        )

    def read_metadata(
        self,
        file_path: str | Path,
    ) -> dict[str, Any]:
        """
        Read only metadata without loading pixel data.

        Args:
            file_path: Path to raster file.

        Returns:
            Dictionary with metadata, crs, transform, bounds, and shape info.
        """
        _check_rasterio()

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with rasterio.open(file_path) as src:
            metadata = dict(src.profile)
            if "crs" in metadata:
                metadata["crs"] = str(metadata["crs"])
            if "transform" in metadata:
                metadata["transform"] = tuple(metadata["transform"])

            return {
                "metadata": metadata,
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform),
                "bounds": tuple(src.bounds),
                "nodata": src.nodata,
                "dtype": str(src.dtypes[0]),
                "shape": (src.count, src.height, src.width),
                "band_count": src.count,
                "overviews": [src.overviews(i) for i in range(1, src.count + 1)],
                "tags": src.tags(),
                "descriptions": list(src.descriptions),
            }


class RasterWriter:
    """Writer class for raster geospatial data."""

    def __init__(self) -> None:
        self.supported_formats = SUPPORTED_RASTER_FORMATS

    def write(
        self,
        data: np.ndarray,
        file_path: str | Path,
        crs: str | None = None,
        transform: Union[tuple, "Affine"] | None = None,
        nodata: int | float | None = None,
        dtype: str | None = None,
        driver: str | None = None,
        compress: str | None = None,
        tags: dict[str, str] | None = None,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Write numpy array as raster file.

        Args:
            data: Numpy array of shape (bands, height, width) or
                  (height, width) for single-band rasters.
            file_path: Output file path.
            crs: Coordinate reference system (e.g. 'EPSG:4326').
            transform: Affine transform. Can be a tuple of 6 or 9 values
                       or a rasterio Affine object.
            nodata: Nodata value to encode in the file.
            dtype: Data type string (e.g. 'float32'). Defaults to the
                   array's dtype.
            driver: Rasterio driver name. Auto-detected from extension
                    when None.
            compress: Compression method (e.g. 'lzw', 'deflate', 'zstd').
            tags: Optional dict of metadata tags to attach.
            overwrite: If False, raise FileExistsError when file exists.
            **kwargs: Additional profile parameters forwarded to rasterio.

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If file format is not supported or data shape is wrong.
            FileExistsError: If file exists and overwrite is False.
        """
        _check_rasterio()

        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported raster format: {file_ext}")

        if not overwrite and file_path.exists():
            raise FileExistsError(f"File already exists: {file_path}")

        # Ensure output directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalise data shape to (bands, height, width)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D array, got {data.ndim}D array with shape {data.shape}"
            )

        band_count, height, width = data.shape

        # Resolve driver
        if driver is None:
            format_name = self.supported_formats[file_ext]
            driver = FORMAT_DRIVERS.get(format_name, "GTiff")

        # Resolve dtype
        if dtype is None:
            dtype = str(data.dtype)

        # Resolve transform
        affine_transform = None
        if transform is not None:
            if isinstance(transform, Affine):
                affine_transform = transform
            else:
                affine_transform = Affine(*transform[:6])
        elif HAS_RASTERIO:
            # A non-identity affine transform makes synthetic rasters valid
            # geospatial datasets and avoids Rasterio's warning on write.
            affine_transform = from_bounds(0, 0, width, height, width, height)

        try:
            if self.supported_formats.get(file_ext) == "COG":
                self._write_cog(
                    data,
                    file_path,
                    crs=crs,
                    transform=affine_transform,
                    nodata=nodata,
                    dtype=dtype,
                    compress=compress,
                    tags=tags,
                    **kwargs,
                )
            else:
                self._write_standard(
                    data,
                    file_path,
                    driver=driver,
                    crs=crs,
                    transform=affine_transform,
                    nodata=nodata,
                    dtype=dtype,
                    compress=compress,
                    tags=tags,
                    **kwargs,
                )

            logger.info(
                f"Successfully wrote raster to {file_path} "
                f"({band_count} bands, {width}x{height}, {dtype})"
            )

        except Exception as e:
            logger.error(f"Failed to write raster file {file_path}: {e}")
            raise

    def _write_standard(
        self,
        data: np.ndarray,
        file_path: Path,
        driver: str,
        crs: str | None,
        transform: Optional["Affine"],
        nodata: int | float | None,
        dtype: str,
        compress: str | None,
        tags: dict[str, str] | None,
        **kwargs: Any,
    ) -> None:
        """Write raster with a standard rasterio driver."""
        band_count, height, width = data.shape

        profile = {
            "driver": driver,
            "width": width,
            "height": height,
            "count": band_count,
            "dtype": dtype,
        }

        if crs is not None:
            profile["crs"] = CRS.from_user_input(crs)
        if transform is not None:
            profile["transform"] = transform
        if nodata is not None:
            profile["nodata"] = nodata
        if compress is not None:
            profile["compress"] = compress

        # Merge any extra kwargs into profile
        profile.update(kwargs)

        with rasterio.open(file_path, "w", **profile) as dst:
            dst.write(data.astype(dtype))
            if tags:
                dst.update_tags(**tags)

    def _write_cog(
        self,
        data: np.ndarray,
        file_path: Path,
        crs: str | None,
        transform: Optional["Affine"],
        nodata: int | float | None,
        dtype: str,
        compress: str | None,
        tags: dict[str, str] | None,
        blocksize: int = 512,
        overview_resampling: str = "nearest",
        **kwargs: Any,
    ) -> None:
        """Write Cloud Optimized GeoTIFF.

        Uses the COG driver when available (GDAL >= 3.1), otherwise writes
        a tiled GeoTIFF with overviews manually.
        """
        band_count, height, width = data.shape

        profile = {
            "driver": "COG",
            "width": width,
            "height": height,
            "count": band_count,
            "dtype": dtype,
            "blocksize": blocksize,
        }

        if crs is not None:
            profile["crs"] = CRS.from_user_input(crs)
        if transform is not None:
            profile["transform"] = transform
        if nodata is not None:
            profile["nodata"] = nodata
        if compress is not None:
            profile["compress"] = compress
        else:
            profile["compress"] = "deflate"

        profile["overview_resampling"] = overview_resampling
        profile.update(kwargs)

        try:
            with rasterio.open(file_path, "w", **profile) as dst:
                dst.write(data.astype(dtype))
                if tags:
                    dst.update_tags(**tags)
        except Exception:
            # Fallback: write a tiled GeoTIFF with overviews
            logger.info("COG driver unavailable, writing tiled GeoTIFF with overviews.")
            fallback_profile = {
                "driver": "GTiff",
                "width": width,
                "height": height,
                "count": band_count,
                "dtype": dtype,
                "tiled": True,
                "blockxsize": blocksize,
                "blockysize": blocksize,
            }

            if crs is not None:
                fallback_profile["crs"] = CRS.from_user_input(crs)
            if transform is not None:
                fallback_profile["transform"] = transform
            if nodata is not None:
                fallback_profile["nodata"] = nodata
            if compress is not None:
                fallback_profile["compress"] = compress
            else:
                fallback_profile["compress"] = "deflate"

            with rasterio.open(file_path, "w", **fallback_profile) as dst:
                dst.write(data.astype(dtype))
                if tags:
                    dst.update_tags(**tags)

                # Build overviews
                overview_factors = []
                factor = 2
                while factor <= min(height, width):
                    overview_factors.append(factor)
                    factor *= 2

                if overview_factors:
                    resampling = getattr(
                        Resampling, overview_resampling, Resampling.nearest
                    )
                    dst.build_overviews(overview_factors, resampling)
                    dst.update_tags(ns="rio_overview", resampling=overview_resampling)

    def write_from_dict(
        self,
        raster_dict: dict[str, Any],
        file_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """
        Write raster from a dictionary produced by RasterReader.read().

        Args:
            raster_dict: Dictionary with 'data', 'crs', 'transform', etc.
            file_path: Output file path.
            **kwargs: Additional parameters forwarded to write().
        """
        self.write(
            data=raster_dict["data"],
            file_path=file_path,
            crs=raster_dict.get("crs"),
            transform=raster_dict.get("transform"),
            nodata=raster_dict.get("nodata"),
            dtype=raster_dict.get("dtype"),
            **kwargs,
        )


# Convenience functions


def read_raster_file(
    file_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Read raster data from file using appropriate reader.

    Args:
        file_path: Path to raster data file.
        **kwargs: Additional parameters for reading (bands, window, masked, etc.).

    Returns:
        Dictionary with 'data', 'metadata', 'crs', 'transform', 'bounds',
        'nodata', 'dtype', and 'shape'.
    """
    reader = RasterReader()
    return reader.read(file_path, **kwargs)


def write_raster_file(
    data: np.ndarray,
    file_path: str | Path,
    **kwargs: Any,
) -> None:
    """
    Write numpy array to raster file using appropriate writer.

    Args:
        data: Numpy array of shape (bands, height, width) or (height, width).
        file_path: Output file path.
        **kwargs: Additional parameters for writing (crs, transform, nodata, etc.).
    """
    writer = RasterWriter()
    writer.write(data, file_path, **kwargs)


def supported_raster_formats() -> dict[str, str]:
    """
    Get dictionary of supported raster formats.

    Returns:
        Dictionary mapping file extensions to format names.
    """
    return SUPPORTED_RASTER_FORMATS.copy()


def detect_raster_format(file_path: str | Path) -> str | None:
    """
    Detect raster format from file extension.

    Args:
        file_path: Path to raster file.

    Returns:
        Format name or None if not supported.
    """
    file_ext = Path(file_path).suffix.lower()
    return SUPPORTED_RASTER_FORMATS.get(file_ext)


def validate_raster_file(file_path: str | Path) -> dict[str, Any]:
    """
    Validate raster file and return metadata.

    Args:
        file_path: Path to raster file.

    Returns:
        Dictionary with validation results and metadata.
    """
    file_path = Path(file_path)

    result: dict[str, Any] = {
        "valid": False,
        "format": None,
        "error": None,
        "metadata": {},
    }

    try:
        if not file_path.exists():
            result["error"] = "File does not exist"
            return result

        format_name = detect_raster_format(file_path)
        if not format_name:
            result["error"] = f"Unsupported format: {file_path.suffix}"
            return result

        result["format"] = format_name

        reader = RasterReader()
        info = reader.read_metadata(file_path)

        result["metadata"] = {
            "band_count": info["band_count"],
            "shape": info["shape"],
            "dtype": info["dtype"],
            "crs": info["crs"],
            "bounds": info["bounds"],
            "nodata": info["nodata"],
            "overviews": info["overviews"],
        }

        result["valid"] = True

    except Exception as e:
        result["error"] = str(e)

    return result
