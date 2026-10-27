# Geospatial data formats

`geo_infer_space.io` reads and writes vector, raster, and point-cloud files.
Its exported format registries describe the extensions routed by the package.
Actual codec support also depends on the installed I/O engine, GDAL drivers,
and optional dependencies; a registered extension alone does not establish a
successful read, write, or performance result.

## Current I/O interfaces

| Data | Public functions | Input and result |
| --- | --- | --- |
| Vector | `read_vector_file`, `write_vector_file`, `supported_vector_formats` | Readers return a GeoDataFrame; writers accept a GeoDataFrame. |
| Raster | `read_raster_file`, `write_raster_file`, `supported_raster_formats` | Readers return a dictionary containing `data`, `crs`, `transform`, `nodata`, and metadata. Writers accept the array and explicit spatial metadata. |
| Point cloud | `read_point_cloud_file`, `write_point_cloud_file`, `supported_point_cloud_formats` | Use the point-cloud reader/writer contracts; LAS/LAZ requires the `laspy` extra. |

The vector registry includes GeoJSON, Shapefile, GeoPackage, KML, GML, CSV,
Excel, Parquet, and Feather. The raster registry includes GeoTIFF, COG,
NetCDF, HDF5, JPEG2000, PNG, and JPEG. Query the corresponding
`supported_*_formats()` function for the current extension mapping. Preserve
geometry, CRS, masks, nodata, and coordinate transforms when changing formats.

## Local vector and raster round trips

This example creates its own small inputs and checks the copied values and
spatial metadata. It uses the declared SPACE dependencies, performs no network
access, and removes its files when the temporary directory closes.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Point

from geo_infer_space.io import (
    read_raster_file,
    read_vector_file,
    supported_raster_formats,
    supported_vector_formats,
    write_raster_file,
    write_vector_file,
)

assert supported_vector_formats()[".geojson"] == "GeoJSON"
assert supported_raster_formats()[".tif"] == "GeoTIFF"

with TemporaryDirectory(prefix="geo-formats-") as temporary:
    directory = Path(temporary)
    points = gpd.GeoDataFrame(
        {"site": ["coast"], "measurement": [3.5]},
        geometry=[Point(-124.2, 41.75)],
        crs="EPSG:4326",
    )
    vector_path = directory / "sites.geojson"
    write_vector_file(points, vector_path)
    restored_points = read_vector_file(vector_path, crs="EPSG:4326")
    assert restored_points["site"].tolist() == ["coast"]
    assert restored_points["measurement"].tolist() == [3.5]
    assert restored_points.geometry.iloc[0].equals(points.geometry.iloc[0])
    assert restored_points.crs == points.crs

    values = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype="float32")
    transform = from_origin(-124.2, 41.75, 0.01, 0.01)
    raster_path = directory / "surface.tif"
    write_raster_file(
        values, raster_path, crs="EPSG:4326", transform=transform, nodata=-9999.0
    )
    raster = read_raster_file(raster_path, masked=False)
    copy_path = directory / "surface_copy.tif"
    write_raster_file(
        raster["data"],
        copy_path,
        crs=raster["crs"],
        transform=raster["transform"],
        nodata=raster["nodata"],
    )
    restored_raster = read_raster_file(copy_path, masked=False)
    np.testing.assert_array_equal(restored_raster["data"][0], values)
    assert restored_raster["transform"] == transform
    assert restored_raster["crs"] == "EPSG:4326"
    assert restored_raster["nodata"] == -9999.0
```

The default GeoJSON reader builds a GeoDataFrame from its features. Supply
its known CRS explicitly as `crs=...`, as above. An explicit `engine=...`
selects the installed GeoPandas reader instead. Assigning a CRS labels the
existing coordinates; use `GeoDataFrame.to_crs()` when reprojection is required.

For raster arrays, `read_raster_file` returns a band-first array shaped
`(bands, height, width)`. Pass its `data` member to `write_raster_file` and
retain the associated CRS, affine transform, and nodata value as above.
A format conversion does not itself reproject coordinates or resample pixels.
Rasterization and polygonization require separately chosen operations and
resolution, extent, attribute, and mask policies.

## Choosing an interchange format

- Use GeoJSON for small vector interchange where readable coordinates help.
- Use GeoPackage when a driver-backed vector workflow needs layers in one file.
- Use GeoTIFF for georeferenced raster exchange; consider COG when the actual
  reader and storage service support its tiled access pattern.
- Choose multidimensional and point-cloud formats from the owning reader's
  contract and installed codec capabilities.

Measure storage size, memory use, and read/write time on the actual workload
before making performance claims. Check a conversion's values, geometry, CRS,
transform, nodata, and any required attributes against the source.

## Related guides

- [H3 integration](h3/index.md)
- [Spatial data models](../concepts/spatial_data_models.md)
- [Temporal dimensions](../concepts/temporal_dimensions.md)
- [Performance guidance](../../advanced/performance_optimization.md)
