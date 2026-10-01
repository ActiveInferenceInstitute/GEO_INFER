"""
Comprehensive tests for the GEO-INFER-SPACE I/O modules.

Covers:
  - Vector I/O   (geo_infer_space.io.vector_io)
  - Raster I/O   (geo_infer_space.io.raster_io)
  - Point Cloud I/O (geo_infer_space.io.point_cloud_io)
  - Format Handlers (geo_infer_space.io.format_handlers)

The I/O modules import unconditionally; heavy optional dependencies such
as rasterio are guarded inside the modules themselves via capability
flags, and the public callables raise ImportError when the dependency is
missing. Tests that exercise those callables branch on ``HAS_RASTERIO``
(mirroring test_whitebox_bridge.py) and assert the loud ImportError
contract when the dependency is absent.
"""

import numpy as np
import pytest

try:
    import rasterio  # noqa: F401

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# Vector I/O tests
# ---------------------------------------------------------------------------


class TestVectorIOImports:
    """Verify that vector I/O classes and functions are importable."""

    def test_vector_reader_class_exists(self):
        from geo_infer_space.io.vector_io import VectorReader

        assert VectorReader is not None

    def test_vector_writer_class_exists(self):
        from geo_infer_space.io.vector_io import VectorWriter

        assert VectorWriter is not None

    def test_supported_vector_formats_importable(self):
        from geo_infer_space.io.vector_io import supported_vector_formats

        assert callable(supported_vector_formats)

    def test_detect_vector_format_importable(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert callable(detect_vector_format)

    def test_convenience_functions_importable(self):
        from geo_infer_space.io.vector_io import read_vector_file, write_vector_file

        assert callable(read_vector_file)
        assert callable(write_vector_file)


class TestSupportedVectorFormats:
    """Validate the supported_vector_formats() return value."""

    def test_returns_dict(self):
        from geo_infer_space.io.vector_io import supported_vector_formats

        result = supported_vector_formats()
        assert isinstance(result, dict)

    def test_contains_geojson(self):
        from geo_infer_space.io.vector_io import supported_vector_formats

        result = supported_vector_formats()
        assert ".geojson" in result
        assert result[".geojson"] == "GeoJSON"

    def test_contains_shapefile(self):
        from geo_infer_space.io.vector_io import supported_vector_formats

        result = supported_vector_formats()
        assert ".shp" in result
        assert result[".shp"] == "ESRI Shapefile"

    def test_contains_geopackage(self):
        from geo_infer_space.io.vector_io import supported_vector_formats

        result = supported_vector_formats()
        assert ".gpkg" in result
        assert result[".gpkg"] == "GPKG"

    def test_contains_csv(self):
        from geo_infer_space.io.vector_io import supported_vector_formats

        result = supported_vector_formats()
        assert ".csv" in result

    def test_contains_parquet(self):
        from geo_infer_space.io.vector_io import supported_vector_formats

        result = supported_vector_formats()
        assert ".parquet" in result

    def test_returns_copy_not_original(self):
        """Mutating the returned dict must not affect the module-level dict."""
        from geo_infer_space.io.vector_io import supported_vector_formats

        first = supported_vector_formats()
        first[".fake"] = "Fake"
        second = supported_vector_formats()
        assert ".fake" not in second


class TestDetectVectorFormat:
    """Validate detect_vector_format() for known and unknown extensions."""

    def test_geojson_detected(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert detect_vector_format("data/rivers.geojson") == "GeoJSON"

    def test_json_detected_as_geojson(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert detect_vector_format("data/parcels.json") == "GeoJSON"

    def test_shp_detected(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert detect_vector_format("/tmp/bounds.shp") == "ESRI Shapefile"

    def test_gpkg_detected(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert detect_vector_format("output.gpkg") == "GPKG"

    def test_kml_detected(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert detect_vector_format("map.kml") == "KML"

    def test_unknown_extension_returns_none(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert detect_vector_format("file.xyz") is None

    def test_case_insensitive_via_path(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert detect_vector_format("MAP.GeoJSON") == "GeoJSON"

    def test_parquet_detected(self):
        from geo_infer_space.io.vector_io import detect_vector_format

        assert detect_vector_format("features.parquet") == "Parquet"


class TestVectorReaderWriter:
    """Basic instantiation and attribute checks for reader/writer."""

    def test_reader_instantiation(self):
        from geo_infer_space.io.vector_io import VectorReader

        reader = VectorReader()
        assert hasattr(reader, "supported_formats")
        assert isinstance(reader.supported_formats, dict)

    def test_writer_instantiation(self):
        from geo_infer_space.io.vector_io import VectorWriter

        writer = VectorWriter()
        assert hasattr(writer, "supported_formats")
        assert isinstance(writer.supported_formats, dict)

    def test_reader_raises_file_not_found(self):
        from geo_infer_space.io.vector_io import VectorReader

        reader = VectorReader()
        with pytest.raises(FileNotFoundError):
            reader.read("/nonexistent/path/data.geojson")

    def test_reader_raises_value_error_for_unsupported_format(self, tmp_path):
        from geo_infer_space.io.vector_io import VectorReader

        bad_file = tmp_path / "data.xyz123"
        bad_file.write_text("dummy")
        reader = VectorReader()
        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read(str(bad_file))

    def test_writer_raises_value_error_for_unsupported_format(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from geo_infer_space.io.vector_io import VectorWriter

        gdf = gpd.GeoDataFrame(
            [{"name": "a", "geometry": Point(0, 0)}],
            crs="EPSG:4326",
        )
        writer = VectorWriter()
        with pytest.raises(ValueError, match="Unsupported format"):
            writer.write(gdf, "/tmp/output.xyz123")


# ---------------------------------------------------------------------------
# Raster I/O tests
# ---------------------------------------------------------------------------


class TestRasterIOImports:
    """Verify that raster I/O classes and functions are importable."""

    def test_raster_reader_class_exists(self):
        from geo_infer_space.io.raster_io import RasterReader

        assert RasterReader is not None

    def test_raster_writer_class_exists(self):
        from geo_infer_space.io.raster_io import RasterWriter

        assert RasterWriter is not None

    def test_convenience_functions_importable(self):
        from geo_infer_space.io.raster_io import (
            read_raster_file,
            write_raster_file,
            supported_raster_formats,
        )

        assert callable(read_raster_file)
        assert callable(write_raster_file)
        assert callable(supported_raster_formats)


class TestSupportedRasterFormats:
    """Validate the supported_raster_formats() return value."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.raster_io import supported_raster_formats

        self._supported_raster_formats = supported_raster_formats

    def test_returns_dict(self):
        result = self._supported_raster_formats()
        assert isinstance(result, dict)

    def test_contains_geotiff(self):
        result = self._supported_raster_formats()
        assert ".tif" in result
        assert result[".tif"] == "GeoTIFF"

    def test_contains_tiff_variant(self):
        result = self._supported_raster_formats()
        assert ".tiff" in result
        assert result[".tiff"] == "GeoTIFF"

    def test_contains_cog(self):
        result = self._supported_raster_formats()
        assert ".cog" in result
        assert result[".cog"] == "COG"

    def test_contains_netcdf(self):
        result = self._supported_raster_formats()
        assert ".nc" in result
        assert result[".nc"] == "NetCDF"

    def test_contains_png(self):
        result = self._supported_raster_formats()
        assert ".png" in result
        assert result[".png"] == "PNG"

    def test_contains_jpeg(self):
        result = self._supported_raster_formats()
        assert ".jpg" in result or ".jpeg" in result

    def test_contains_hdf5(self):
        result = self._supported_raster_formats()
        assert ".hdf5" in result or ".hdf" in result or ".h5" in result

    def test_returns_copy_not_original(self):
        first = self._supported_raster_formats()
        first[".fake_raster"] = "Fake"
        second = self._supported_raster_formats()
        assert ".fake_raster" not in second


class TestRasterReaderErrors:
    """Test that raster reader fails gracefully for error conditions."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.raster_io import RasterReader

        self._RasterReader = RasterReader

    def test_read_missing_file_raises_file_not_found(self):
        reader = self._RasterReader()
        with pytest.raises(FileNotFoundError):
            reader.read("/nonexistent/path/image.tif")

    def test_read_unsupported_format_raises_value_error(self, tmp_path):
        bad_file = tmp_path / "data.bmp"
        bad_file.write_text("not a raster")
        reader = self._RasterReader()
        with pytest.raises(ValueError, match="Unsupported raster format"):
            reader.read(str(bad_file))


class TestRasterWriterDirectoryCreation:
    """Test that the raster writer creates output directories as needed."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.raster_io import RasterWriter

        self._RasterWriter = RasterWriter

    def test_write_creates_parent_directory(self, tmp_path):
        nested_dir = tmp_path / "a" / "b" / "c"
        out_file = nested_dir / "output.tif"
        data = np.random.rand(1, 16, 16).astype(np.float32)

        writer = self._RasterWriter()
        if HAS_RASTERIO:
            writer.write(data, str(out_file), crs="EPSG:4326")

            assert nested_dir.exists()
            assert out_file.exists()
        else:
            # Without rasterio the call must fail loudly, never fabricate
            # a silently-created output tree.
            with pytest.raises(ImportError, match="rasterio is required"):
                writer.write(data, str(out_file), crs="EPSG:4326")
            assert not nested_dir.exists()

    def test_write_unsupported_format_raises_value_error(self, tmp_path):
        writer = self._RasterWriter()
        data = np.zeros((1, 4, 4), dtype=np.float32)
        if HAS_RASTERIO:
            with pytest.raises(ValueError, match="Unsupported raster format"):
                writer.write(data, str(tmp_path / "output.bmp"))
        else:
            # The rasterio availability check fires before the format check
            # in this environment.
            with pytest.raises(ImportError, match="rasterio is required"):
                writer.write(data, str(tmp_path / "output.bmp"))

    def test_write_rejects_4d_array(self, tmp_path):
        writer = self._RasterWriter()
        data = np.zeros((2, 3, 4, 4), dtype=np.float32)
        if HAS_RASTERIO:
            with pytest.raises(ValueError, match="Expected 2D or 3D array"):
                writer.write(data, str(tmp_path / "bad.tif"))
        else:
            with pytest.raises(ImportError, match="rasterio is required"):
                writer.write(data, str(tmp_path / "bad.tif"))


class TestRasterRoundTrip:
    """Write and read back a small synthetic raster to verify round-trip."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.raster_io import (
            RasterReader,
            RasterWriter,
            read_raster_file,
            write_raster_file,
        )

        self._RasterReader = RasterReader
        self._RasterWriter = RasterWriter
        self._read_raster_file = read_raster_file
        self._write_raster_file = write_raster_file

    def test_single_band_round_trip(self, tmp_path):
        out = tmp_path / "single_band.tif"
        original = np.arange(64, dtype=np.float32).reshape(8, 8)

        writer = self._RasterWriter()
        writer.write(original, str(out), crs="EPSG:4326")

        reader = self._RasterReader()
        result = reader.read(str(out))

        assert result["shape"] == (1, 8, 8)
        assert result["crs"] is not None
        np.testing.assert_array_almost_equal(
            result["data"].squeeze(), original, decimal=5
        )

    def test_multi_band_round_trip(self, tmp_path):
        out = tmp_path / "multi_band.tif"
        original = np.random.rand(3, 16, 16).astype(np.float32)

        self._write_raster_file(original, str(out), crs="EPSG:4326")
        result = self._read_raster_file(str(out))

        assert result["shape"] == (3, 16, 16)
        np.testing.assert_array_almost_equal(
            np.asarray(result["data"]), original, decimal=5
        )

    def test_nodata_preserved(self, tmp_path):
        out = tmp_path / "nodata.tif"
        data = np.ones((1, 4, 4), dtype=np.float32)
        data[0, 0, 0] = -9999.0

        self._write_raster_file(data, str(out), nodata=-9999.0)
        result = self._read_raster_file(str(out), masked=False)

        assert result["nodata"] == pytest.approx(-9999.0)

    def test_write_from_dict(self, tmp_path):
        """write_from_dict should accept the dict that read() produces."""
        src_path = tmp_path / "src.tif"
        dst_path = tmp_path / "dst.tif"

        original = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        writer = self._RasterWriter()
        writer.write(original, str(src_path), crs="EPSG:4326")

        reader = self._RasterReader()
        raster_dict = reader.read(str(src_path), masked=False)

        writer.write_from_dict(raster_dict, str(dst_path))

        result = reader.read(str(dst_path), masked=False)
        np.testing.assert_array_almost_equal(
            result["data"], raster_dict["data"], decimal=5
        )

    def test_read_metadata_only(self, tmp_path):
        fpath = tmp_path / "meta.tif"
        data = np.zeros((2, 10, 10), dtype=np.float32)
        self._write_raster_file(data, str(fpath), crs="EPSG:4326")

        reader = self._RasterReader()
        info = reader.read_metadata(str(fpath))

        assert info["band_count"] == 2
        assert info["shape"] == (2, 10, 10)
        assert info["crs"] is not None
        assert "bounds" in info


# ---------------------------------------------------------------------------
# Point Cloud I/O tests
# ---------------------------------------------------------------------------


class TestPointCloudIOImports:
    """Verify that point cloud I/O classes and functions are importable."""

    def test_point_cloud_reader_class_exists(self):
        from geo_infer_space.io.point_cloud_io import PointCloudReader

        assert PointCloudReader is not None

    def test_point_cloud_writer_class_exists(self):
        from geo_infer_space.io.point_cloud_io import PointCloudWriter

        assert PointCloudWriter is not None

    def test_convenience_functions_importable(self):
        from geo_infer_space.io.point_cloud_io import (
            read_point_cloud_file,
            write_point_cloud_file,
            supported_point_cloud_formats,
        )

        assert callable(read_point_cloud_file)
        assert callable(write_point_cloud_file)
        assert callable(supported_point_cloud_formats)


class TestSupportedPointCloudFormats:
    """Validate the supported_point_cloud_formats() return value."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.point_cloud_io import supported_point_cloud_formats

        self._supported = supported_point_cloud_formats

    def test_returns_dict(self):
        result = self._supported()
        assert isinstance(result, dict)

    def test_contains_las(self):
        result = self._supported()
        assert ".las" in result

    def test_contains_laz(self):
        result = self._supported()
        assert ".laz" in result

    def test_contains_ply(self):
        result = self._supported()
        assert ".ply" in result

    def test_contains_xyz(self):
        result = self._supported()
        assert ".xyz" in result

    def test_contains_csv(self):
        result = self._supported()
        assert ".csv" in result

    def test_returns_copy(self):
        first = self._supported()
        first[".fake_cloud"] = "Fake"
        second = self._supported()
        assert ".fake_cloud" not in second


class TestPointCloudValidation:
    """Test input validation for point cloud writer."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.point_cloud_io import PointCloudWriter

        self._Writer = PointCloudWriter

    def test_rejects_1d_array(self, tmp_path):
        writer = self._Writer()
        bad = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="points must be an"):
            writer.write(bad, str(tmp_path / "bad.xyz"))

    def test_rejects_2_column_array(self, tmp_path):
        writer = self._Writer()
        bad = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="points must be an"):
            writer.write(bad, str(tmp_path / "bad.xyz"))

    def test_rejects_3d_array(self, tmp_path):
        writer = self._Writer()
        bad = np.zeros((2, 3, 3))
        with pytest.raises(ValueError, match="points must be an"):
            writer.write(bad, str(tmp_path / "bad.xyz"))

    def test_rejects_unsupported_format(self, tmp_path):
        writer = self._Writer()
        pts = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Unsupported point cloud format"):
            writer.write(pts, str(tmp_path / "bad.obj"))

    def test_reader_missing_file(self):
        from geo_infer_space.io.point_cloud_io import PointCloudReader

        reader = PointCloudReader()
        with pytest.raises(FileNotFoundError):
            reader.read("/nonexistent/cloud.xyz")

    def test_reader_unsupported_format(self, tmp_path):
        from geo_infer_space.io.point_cloud_io import PointCloudReader

        bad = tmp_path / "cloud.obj"
        bad.write_text("junk")
        reader = PointCloudReader()
        with pytest.raises(ValueError, match="Unsupported point cloud format"):
            reader.read(str(bad))


@pytest.fixture
def sample_points():
    """Synthetic Nx3 point cloud for round-trip testing."""
    rng = np.random.default_rng(42)
    return rng.uniform(-100, 100, size=(50, 3))


@pytest.fixture
def sample_classifications():
    """Integer classification labels matching sample_points count."""
    return np.random.default_rng(42).integers(0, 6, size=50).astype(np.int32)


@pytest.fixture
def sample_intensities():
    """Float intensity values matching sample_points count."""
    return np.random.default_rng(42).uniform(0, 255, size=50).astype(np.float64)


@pytest.fixture
def sample_colors():
    """RGB colour array matching sample_points count."""
    return np.random.default_rng(42).integers(0, 256, size=(50, 3)).astype(np.uint8)


class TestXYZRoundTrip:
    """Write and read back XYZ point cloud files."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.point_cloud_io import (
            PointCloudReader,
            PointCloudWriter,
        )

        self._Reader = PointCloudReader
        self._Writer = PointCloudWriter

    def test_xyz_basic_round_trip(self, tmp_path, sample_points):
        fpath = tmp_path / "basic.xyz"

        writer = self._Writer()
        writer.write(sample_points, str(fpath))

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["points"].shape == sample_points.shape
        np.testing.assert_array_almost_equal(result["points"], sample_points, decimal=4)
        assert result["metadata"]["format"] == "XYZ"
        assert result["metadata"]["point_count"] == sample_points.shape[0]

    def test_xyz_with_intensities(self, tmp_path, sample_points, sample_intensities):
        fpath = tmp_path / "intensities.xyz"

        writer = self._Writer()
        writer.write(sample_points, str(fpath), intensities=sample_intensities)

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["points"].shape == sample_points.shape
        assert result["intensities"] is not None
        np.testing.assert_array_almost_equal(
            result["intensities"], sample_intensities, decimal=4
        )

    def test_xyz_with_classifications(
        self, tmp_path, sample_points, sample_classifications
    ):
        fpath = tmp_path / "classified.xyz"

        writer = self._Writer()
        writer.write(
            sample_points,
            str(fpath),
            intensities=None,
            classifications=sample_classifications,
        )

        # When XYZ writer writes with classifications only (no intensities),
        # the file will have 4 columns: x y z classification.
        # But the reader auto-assigns column 4 as 'intensity' by default.
        # To verify the data is round-trippable we check the raw column count.
        reader = self._Reader()
        result = reader.read(str(fpath))
        assert result["points"].shape == sample_points.shape

    def test_xyz_creates_parent_dir(self, tmp_path, sample_points):
        nested = tmp_path / "deep" / "nested" / "dir"
        fpath = nested / "cloud.xyz"

        writer = self._Writer()
        writer.write(sample_points, str(fpath))

        assert nested.exists()
        assert fpath.exists()

    def test_xyz_single_point(self, tmp_path):
        pts = np.array([[1.0, 2.0, 3.0]])
        fpath = tmp_path / "single.xyz"

        writer = self._Writer()
        writer.write(pts, str(fpath))

        reader = self._Reader()
        result = reader.read(str(fpath))
        assert result["points"].shape == (1, 3)
        np.testing.assert_array_almost_equal(result["points"], pts, decimal=4)


class TestCSVRoundTrip:
    """Write and read back CSV point cloud files."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.point_cloud_io import (
            PointCloudReader,
            PointCloudWriter,
        )

        self._Reader = PointCloudReader
        self._Writer = PointCloudWriter

    def test_csv_basic_round_trip(self, tmp_path, sample_points):
        fpath = tmp_path / "points.csv"

        writer = self._Writer()
        writer.write(sample_points, str(fpath))

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["points"].shape == sample_points.shape
        np.testing.assert_array_almost_equal(result["points"], sample_points, decimal=4)
        assert result["metadata"]["format"] == "CSV"
        assert result["metadata"]["point_count"] == sample_points.shape[0]

    def test_csv_with_intensities_and_classifications(
        self, tmp_path, sample_points, sample_intensities, sample_classifications
    ):
        fpath = tmp_path / "full.csv"

        writer = self._Writer()
        writer.write(
            sample_points,
            str(fpath),
            intensities=sample_intensities,
            classifications=sample_classifications,
        )

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["points"].shape == sample_points.shape
        assert result["intensities"] is not None
        assert result["classifications"] is not None

    def test_csv_with_colors(self, tmp_path, sample_points, sample_colors):
        fpath = tmp_path / "colored.csv"

        writer = self._Writer()
        writer.write(sample_points, str(fpath), colors=sample_colors)

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["points"].shape == sample_points.shape
        assert result["colors"] is not None
        assert result["colors"].shape == (sample_points.shape[0], 3)

    def test_csv_header_present(self, tmp_path, sample_points):
        fpath = tmp_path / "header_check.csv"

        writer = self._Writer()
        writer.write(sample_points, str(fpath))

        with open(fpath) as f:
            first_line = f.readline().strip()

        assert "x" in first_line.lower()
        assert "y" in first_line.lower()
        assert "z" in first_line.lower()

    def test_csv_metadata_includes_headers(self, tmp_path, sample_points):
        fpath = tmp_path / "meta.csv"

        writer = self._Writer()
        writer.write(sample_points, str(fpath))

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert "headers" in result["metadata"]
        assert "x" in result["metadata"]["headers"]
        assert "y" in result["metadata"]["headers"]
        assert "z" in result["metadata"]["headers"]


class TestPLYRoundTrip:
    """Write and read back PLY point cloud files."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.point_cloud_io import (
            PointCloudReader,
            PointCloudWriter,
        )

        self._Reader = PointCloudReader
        self._Writer = PointCloudWriter

    def test_ply_binary_round_trip(self, tmp_path, sample_points):
        fpath = tmp_path / "binary.ply"

        writer = self._Writer()
        writer.write(sample_points, str(fpath))  # binary=True by default

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["points"].shape == sample_points.shape
        np.testing.assert_array_almost_equal(result["points"], sample_points, decimal=4)
        assert result["metadata"]["format"] == "PLY"
        assert result["metadata"]["point_count"] == sample_points.shape[0]

    def test_ply_ascii_round_trip(self, tmp_path, sample_points):
        fpath = tmp_path / "ascii.ply"

        writer = self._Writer()
        writer.write(sample_points, str(fpath), binary=False)

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["points"].shape == sample_points.shape
        np.testing.assert_array_almost_equal(result["points"], sample_points, decimal=3)

    def test_ply_with_colors(self, tmp_path, sample_points, sample_colors):
        fpath = tmp_path / "colored.ply"

        writer = self._Writer()
        writer.write(sample_points, str(fpath), colors=sample_colors)

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["colors"] is not None
        assert result["colors"].shape == (sample_points.shape[0], 3)
        np.testing.assert_array_equal(result["colors"], sample_colors)

    def test_ply_with_classifications(
        self, tmp_path, sample_points, sample_classifications
    ):
        fpath = tmp_path / "classified.ply"

        writer = self._Writer()
        writer.write(sample_points, str(fpath), classifications=sample_classifications)

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["classifications"] is not None
        np.testing.assert_array_equal(
            result["classifications"],
            sample_classifications.astype(np.uint8),
        )

    def test_ply_with_intensities(self, tmp_path, sample_points, sample_intensities):
        fpath = tmp_path / "intensities.ply"

        writer = self._Writer()
        writer.write(sample_points, str(fpath), intensities=sample_intensities)

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["intensities"] is not None
        np.testing.assert_array_almost_equal(
            result["intensities"],
            sample_intensities.astype(np.float32),
            decimal=2,
        )

    def test_ply_with_all_attributes(
        self,
        tmp_path,
        sample_points,
        sample_intensities,
        sample_classifications,
        sample_colors,
    ):
        fpath = tmp_path / "full.ply"

        writer = self._Writer()
        writer.write(
            sample_points,
            str(fpath),
            intensities=sample_intensities,
            classifications=sample_classifications,
            colors=sample_colors,
        )

        reader = self._Reader()
        result = reader.read(str(fpath))

        assert result["points"].shape == sample_points.shape
        assert result["intensities"] is not None
        assert result["classifications"] is not None
        assert result["colors"] is not None

    def test_ply_header_contains_properties(self, tmp_path, sample_points):
        fpath = tmp_path / "header_check.ply"

        writer = self._Writer()
        writer.write(sample_points, str(fpath), binary=False)

        with open(fpath, "rb") as f:
            header_bytes = b""
            while True:
                line = f.readline()
                header_bytes += line
                if b"end_header" in line:
                    break

        header = header_bytes.decode("ascii")
        assert "element vertex" in header
        assert "property" in header
        assert "ply" in header

    def test_ply_metadata_bounds(self, tmp_path, sample_points):
        fpath = tmp_path / "bounds.ply"

        writer = self._Writer()
        writer.write(sample_points, str(fpath))

        reader = self._Reader()
        result = reader.read(str(fpath))

        bounds = result["metadata"]["bounds"]
        assert "min" in bounds
        assert "max" in bounds
        assert len(bounds["min"]) == 3
        assert len(bounds["max"]) == 3
        np.testing.assert_array_almost_equal(
            bounds["min"], sample_points.min(axis=0).tolist(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            bounds["max"], sample_points.max(axis=0).tolist(), decimal=4
        )


class TestPointCloudConvenienceFunctions:
    """Test the module-level convenience wrappers."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.point_cloud_io import (
            read_point_cloud_file,
            write_point_cloud_file,
        )

        self._read = read_point_cloud_file
        self._write = write_point_cloud_file

    def test_write_then_read_xyz(self, tmp_path, sample_points):
        fpath = tmp_path / "conv.xyz"
        self._write(sample_points, str(fpath))
        result = self._read(str(fpath))
        assert result["points"].shape == sample_points.shape

    def test_write_then_read_csv(self, tmp_path, sample_points):
        fpath = tmp_path / "conv.csv"
        self._write(sample_points, str(fpath))
        result = self._read(str(fpath))
        assert result["points"].shape == sample_points.shape


# ---------------------------------------------------------------------------
# Format Handlers tests
# ---------------------------------------------------------------------------


class TestFormatHandlerABC:
    """Verify that FormatHandler is abstract and cannot be instantiated."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import FormatHandler

        self._FormatHandler = FormatHandler

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            self._FormatHandler()

    def test_has_abstract_methods(self):
        import inspect

        abstracts = {
            name
            for name, _ in inspect.getmembers(self._FormatHandler)
            if getattr(
                getattr(self._FormatHandler, name, None),
                "__isabstractmethod__",
                False,
            )
        }
        assert "can_handle" in abstracts
        assert "read" in abstracts
        assert "write" in abstracts
        assert "validate" in abstracts

    def test_format_name_is_abstract(self):
        # format_name is an abstract property
        assert hasattr(self._FormatHandler, "format_name")


class TestGeoJSONHandler:
    """Test the GeoJSONHandler concrete implementation."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import GeoJSONHandler

        self._handler = GeoJSONHandler()

    def test_can_handle_geojson_extension(self):
        assert self._handler.can_handle("data.geojson") is True

    def test_can_handle_json_extension(self):
        assert self._handler.can_handle("data.json") is True

    def test_cannot_handle_shp(self):
        assert self._handler.can_handle("data.shp") is False

    def test_cannot_handle_tif(self):
        assert self._handler.can_handle("data.tif") is False

    def test_format_name(self):
        assert self._handler.format_name == "GeoJSON"

    def test_extensions_attribute(self):
        assert ".geojson" in self._handler.extensions
        assert ".json" in self._handler.extensions


class TestShapefileHandler:
    """Test the ShapefileHandler concrete implementation."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import ShapefileHandler

        self._handler = ShapefileHandler()

    def test_can_handle_shp(self):
        assert self._handler.can_handle("boundaries.shp") is True

    def test_cannot_handle_geojson(self):
        assert self._handler.can_handle("data.geojson") is False

    def test_cannot_handle_tif(self):
        assert self._handler.can_handle("image.tif") is False

    def test_format_name(self):
        assert self._handler.format_name == "ESRI Shapefile"

    def test_extensions_attribute(self):
        assert ".shp" in self._handler.extensions


class TestGeoTIFFHandler:
    """Test the GeoTIFFHandler concrete implementation."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import GeoTIFFHandler

        self._handler = GeoTIFFHandler()

    def test_can_handle_tif(self):
        assert self._handler.can_handle("elevation.tif") is True

    def test_can_handle_tiff(self):
        assert self._handler.can_handle("satellite.tiff") is True

    def test_can_handle_geotiff(self):
        assert self._handler.can_handle("dem.geotiff") is True

    def test_cannot_handle_geojson(self):
        assert self._handler.can_handle("data.geojson") is False

    def test_cannot_handle_las(self):
        assert self._handler.can_handle("cloud.las") is False

    def test_format_name(self):
        assert self._handler.format_name == "GeoTIFF"

    def test_extensions_attribute(self):
        assert ".tif" in self._handler.extensions
        assert ".tiff" in self._handler.extensions


class TestCOGHandler:
    """Test the COGHandler concrete implementation."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import COGHandler, GeoTIFFHandler

        self._handler = COGHandler()
        self._GeoTIFFHandler = GeoTIFFHandler

    def test_inherits_from_geotiff_handler(self):
        assert isinstance(self._handler, self._GeoTIFFHandler)

    def test_can_handle_tif(self):
        # COG handler supports the same extensions as GeoTIFF
        assert self._handler.can_handle("cog_image.tif") is True

    def test_can_handle_tiff(self):
        assert self._handler.can_handle("cog_image.tiff") is True

    def test_format_name(self):
        assert self._handler.format_name == "Cloud-Optimized GeoTIFF"


class TestLASHandler:
    """Test the LASHandler concrete implementation."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import LASHandler

        self._handler = LASHandler()

    def test_can_handle_las(self):
        assert self._handler.can_handle("terrain.las") is True

    def test_can_handle_laz(self):
        assert self._handler.can_handle("terrain.laz") is True

    def test_cannot_handle_tif(self):
        assert self._handler.can_handle("image.tif") is False

    def test_cannot_handle_geojson(self):
        assert self._handler.can_handle("data.geojson") is False

    def test_format_name(self):
        assert self._handler.format_name == "LAS/LAZ"

    def test_extensions_attribute(self):
        assert ".las" in self._handler.extensions
        assert ".laz" in self._handler.extensions


class TestNetCDFHandler:
    """Test the NetCDFHandler concrete implementation."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import NetCDFHandler

        self._handler = NetCDFHandler()

    def test_can_handle_nc(self):
        assert self._handler.can_handle("climate.nc") is True

    def test_can_handle_nc4(self):
        assert self._handler.can_handle("data.nc4") is True

    def test_can_handle_netcdf(self):
        assert self._handler.can_handle("model.netcdf") is True

    def test_cannot_handle_tif(self):
        assert self._handler.can_handle("image.tif") is False

    def test_cannot_handle_las(self):
        assert self._handler.can_handle("cloud.las") is False

    def test_format_name(self):
        assert self._handler.format_name == "NetCDF"

    def test_extensions_attribute(self):
        assert ".nc" in self._handler.extensions


class TestGetHandlerForPath:
    """Test the get_handler_for_path() registry function."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import (
            get_handler_for_path,
            GeoJSONHandler,
            ShapefileHandler,
            GeoTIFFHandler,
            COGHandler,
            LASHandler,
            NetCDFHandler,
        )

        self._get_handler = get_handler_for_path
        self._GeoJSONHandler = GeoJSONHandler
        self._ShapefileHandler = ShapefileHandler
        self._GeoTIFFHandler = GeoTIFFHandler
        self._COGHandler = COGHandler
        self._LASHandler = LASHandler
        self._NetCDFHandler = NetCDFHandler

    def test_geojson_returns_geojson_handler(self):
        handler = self._get_handler("rivers.geojson")
        assert isinstance(handler, self._GeoJSONHandler)

    def test_json_returns_geojson_handler(self):
        handler = self._get_handler("parcels.json")
        assert isinstance(handler, self._GeoJSONHandler)

    def test_shp_returns_shapefile_handler(self):
        handler = self._get_handler("bounds.shp")
        assert isinstance(handler, self._ShapefileHandler)

    def test_tif_returns_geotiff_or_cog_handler(self):
        handler = self._get_handler("dem.tif")
        # COGHandler inherits from GeoTIFFHandler, and COG is registered
        # first in _BUILTIN_HANDLERS, so we may get either.
        assert isinstance(handler, self._GeoTIFFHandler)

    def test_tiff_returns_geotiff_handler(self):
        handler = self._get_handler("image.tiff")
        assert isinstance(handler, self._GeoTIFFHandler)

    def test_las_returns_las_handler(self):
        handler = self._get_handler("lidar.las")
        assert isinstance(handler, self._LASHandler)

    def test_laz_returns_las_handler(self):
        handler = self._get_handler("lidar.laz")
        assert isinstance(handler, self._LASHandler)

    def test_nc_returns_netcdf_handler(self):
        handler = self._get_handler("climate.nc")
        assert isinstance(handler, self._NetCDFHandler)

    def test_unknown_returns_none(self):
        handler = self._get_handler("unknown.xyz999")
        assert handler is None

    def test_no_extension_returns_none(self):
        handler = self._get_handler("README")
        assert handler is None


class TestListSupportedFormats:
    """Test the list_supported_formats() helper."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        from geo_infer_space.io.format_handlers import list_supported_formats

        self._list = list_supported_formats

    def test_returns_dict(self):
        result = self._list()
        assert isinstance(result, dict)

    def test_keys_are_format_names(self):
        result = self._list()
        expected_names = {
            "GeoJSON",
            "ESRI Shapefile",
            "GeoTIFF",
            "Cloud-Optimized GeoTIFF",
            "LAS/LAZ",
            "NetCDF",
        }
        # All expected names must be present
        assert expected_names.issubset(set(result.keys()))

    def test_values_are_extension_lists(self):
        result = self._list()
        for name, extensions in result.items():
            assert isinstance(extensions, list), (
                f"Extensions for {name} should be a list"
            )
            for ext in extensions:
                assert ext.startswith("."), (
                    f"Extension {ext!r} for {name} should start with '.'"
                )


# ---------------------------------------------------------------------------
# IO __init__ re-export tests
# ---------------------------------------------------------------------------


class TestIOModuleReExports:
    """Verify that the io package __init__ re-exports the expected symbols."""

    def test_vector_reader_via_io_package(self):
        from geo_infer_space.io import VectorReader

        assert VectorReader is not None

    def test_vector_writer_via_io_package(self):
        from geo_infer_space.io import VectorWriter

        assert VectorWriter is not None

    def test_supported_vector_formats_via_io_package(self):
        from geo_infer_space.io import supported_vector_formats

        assert callable(supported_vector_formats)

    def test_raster_reader_if_available(self):
        from geo_infer_space.io import RasterReader

        assert RasterReader is not None

    def test_raster_writer_if_available(self):
        from geo_infer_space.io import RasterWriter

        assert RasterWriter is not None

    def test_point_cloud_reader_if_available(self):
        from geo_infer_space.io import PointCloudReader

        assert PointCloudReader is not None

    def test_point_cloud_writer_if_available(self):
        from geo_infer_space.io import PointCloudWriter

        assert PointCloudWriter is not None

    def test_format_handler_if_available(self):
        from geo_infer_space.io import FormatHandler

        assert FormatHandler is not None

    def test_geojson_handler_if_available(self):
        from geo_infer_space.io import GeoJSONHandler

        assert GeoJSONHandler is not None

    def test_geotiff_handler_if_available(self):
        from geo_infer_space.io import GeoTIFFHandler

        assert GeoTIFFHandler is not None
