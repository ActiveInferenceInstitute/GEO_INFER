"""
I/O module for GEO-INFER-SPACE geospatial data handling.

Reading and writing for vector, raster, and point cloud data. LAS/LAZ point
clouds require the optional ``laspy`` extra and NetCDF requires ``xarray``;
the corresponding callables raise ImportError when the extra is missing.
"""

from .format_handlers import (
    COGHandler,
    FormatHandler,
    GeoJSONHandler,
    GeoTIFFHandler,
    LASHandler,
    NetCDFHandler,
    ShapefileHandler,
)
from .point_cloud_io import (
    PointCloudReader,
    PointCloudWriter,
    read_point_cloud_file,
    supported_point_cloud_formats,
    write_point_cloud_file,
)
from .raster_io import (
    RasterReader,
    RasterWriter,
    read_raster_file,
    supported_raster_formats,
    write_raster_file,
)
from .vector_io import (
    VectorReader,
    VectorWriter,
    read_vector_file,
    supported_vector_formats,
    write_vector_file,
)

__all__ = [
    # Vector I/O
    "VectorReader",
    "VectorWriter",
    "read_vector_file",
    "write_vector_file",
    "supported_vector_formats",
    # Raster I/O
    "RasterReader",
    "RasterWriter",
    "read_raster_file",
    "write_raster_file",
    "supported_raster_formats",
    # Point cloud I/O
    "PointCloudReader",
    "PointCloudWriter",
    "read_point_cloud_file",
    "write_point_cloud_file",
    "supported_point_cloud_formats",
    # Format handlers
    "FormatHandler",
    "GeoJSONHandler",
    "ShapefileHandler",
    "GeoTIFFHandler",
    "COGHandler",
    "LASHandler",
    "NetCDFHandler",
]
