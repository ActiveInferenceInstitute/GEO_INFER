"""Tests for the cloud-native vector reader with DuckDB-Spatial fallback."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
import shapely

from geo_infer_data.utils.duckdb_spatial import (
    HAS_DUCKDB,
    duckdb_status,
    read_cloud_native_vector,
)


def test_required_duckdb_failure_cannot_run_fallback(geojson_file, monkeypatch):
    """An explicit backend requirement preserves the original failure boundary."""
    import geo_infer_data.utils.duckdb_spatial as backend

    def failed_read(*args, **kwargs):
        raise RuntimeError("backend rejected fixture")

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("explicit backend silently fell back")

    monkeypatch.setattr(backend, "_duckdb_read_vector", failed_read)
    monkeypatch.setattr(backend, "_fallback_read_vector", forbidden_fallback)
    with pytest.raises(backend.DuckDBSpatialError) as raised:
        backend.read_cloud_native_vector(geojson_file, require_duckdb=True)
    assert str(raised.value.__cause__) == "backend rejected fixture"


def test_required_duckdb_rejects_absence_and_unsupported_options(
    geojson_file, monkeypatch
):
    """Missing extras and dropped reader parameters must be explicit failures."""
    import geo_infer_data.utils.duckdb_spatial as backend

    with pytest.raises(ValueError, match="layer"):
        backend.read_cloud_native_vector(
            geojson_file, require_duckdb=True, layer="points"
        )
    monkeypatch.setattr(backend, "HAS_DUCKDB", False)
    with pytest.raises(backend.DuckDBSpatialError, match="not installed"):
        backend.read_cloud_native_vector(geojson_file, require_duckdb=True)


@pytest.fixture
def geojson_file(tmp_path: Path) -> Path:
    """A tiny GeoJSON feature collection on disk."""
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[shapely.Point(0, 0), shapely.Point(1, 1)],
        crs="EPSG:4326",
    )
    path = tmp_path / "points.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


def test_read_cloud_native_vector_fallback(geojson_file: Path) -> None:
    """Reads a GeoJSON file via the fallback path regardless of duckdb."""
    gdf = read_cloud_native_vector(geojson_file, use_duckdb=False)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2
    assert list(gdf.columns) == ["name", "geometry"]


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_cloud_native_vector(tmp_path / "nope.geojson")


def test_duckdb_status_reports_availability() -> None:
    status = duckdb_status()
    if HAS_DUCKDB:
        assert "available" in status
    else:
        assert "fallback" in status


def test_has_duckdb_is_boolean() -> None:
    assert isinstance(HAS_DUCKDB, bool)


def test_read_layer_arg(tmp_path: Path) -> None:
    """The fallback path forwards the layer kwarg without error for GeoJSON."""
    gdf = gpd.GeoDataFrame({"x": [1]}, geometry=[shapely.Point(0, 0)], crs="EPSG:4326")
    path = tmp_path / "single.geojson"
    gdf.to_file(path, driver="GeoJSON")
    # layer=None is safe on the fallback path.
    out = read_cloud_native_vector(path, use_duckdb=False, layer=None)
    assert len(out) == 1


def test_read_quote_containing_path(tmp_path: Path) -> None:
    """A path containing a single quote must round-trip, not inject SQL."""
    gdf = gpd.GeoDataFrame(
        {"name": ["it's"]},
        geometry=[shapely.Point(2, 2)],
        crs="EPSG:4326",
    )
    tricky_dir = tmp_path / "bob's files"
    tricky_dir.mkdir()
    path = tricky_dir / "quote's test.geojson"
    gdf.to_file(path, driver="GeoJSON")
    out = read_cloud_native_vector(path)
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) == 1
    assert out["name"].tolist() == ["it's"]


def test_read_directory_raises(tmp_path: Path) -> None:
    """A path that exists but is not a regular file is rejected."""
    with pytest.raises(FileNotFoundError, match="Not a regular file"):
        read_cloud_native_vector(tmp_path)


def test_duckdb_and_fallback_agree_on_projected_crs(tmp_path: Path) -> None:
    """GS-109: the DuckDB fast path must preserve the file's CRS, not
    hardcode EPSG:4326. A projected (EPSG:32610) FlatGeobuf must yield the
    identical CRS through both engines."""
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[
            shapely.Point(500000, 4649776),
            shapely.Point(500100, 4649800),
        ],
        crs="EPSG:32610",
    )
    path = tmp_path / "projected.fgb"
    gdf.to_file(path, driver="FlatGeobuf")

    fast = read_cloud_native_vector(path, require_duckdb=True)
    slow = read_cloud_native_vector(path, use_duckdb=False)

    assert fast.crs == slow.crs
    assert fast.crs is not None and fast.crs.to_epsg() == 32610
    assert fast["name"].tolist() == slow["name"].tolist()
    assert fast.geometry.iloc[0].x == slow.geometry.iloc[0].x


def test_duckdb_fast_path_reports_wgs84_for_wgs84_file(tmp_path: Path) -> None:
    """A WGS84 GeoParquet still resolves to EPSG:4326 via spec-default metadata."""
    gdf = gpd.GeoDataFrame(
        {"name": ["x"]}, geometry=[shapely.Point(1, 2)], crs="EPSG:4326"
    )
    path = tmp_path / "wgs.parquet"
    gdf.to_parquet(path)

    from geo_infer_data.utils.duckdb_spatial import _resolve_crs
    import duckdb as _duckdb

    conn = _duckdb.connect()
    try:
        conn.execute("LOAD spatial;")
        crs = _resolve_crs(conn, path.as_posix())
    finally:
        conn.close()
    assert crs is not None and crs.to_epsg() == 4326


@pytest.mark.parametrize(
    "failure", ["native_import_error", "transitive_module_missing"]
)
def test_installed_duckdb_import_failure_propagates(tmp_path, failure):
    """Only absent DuckDB is optional; failures inside an installed backend are fatal."""
    import subprocess
    import sys

    script = f"""
import builtins, importlib
original = builtins.__import__
def import_with_failure(name, *args, **kwargs):
    if name == 'duckdb':
        if {failure!r} == 'native_import_error':
            raise ImportError('installed DuckDB failed its native import')
        raise ModuleNotFoundError('installed DuckDB has a missing transitive module', name='broken_transitive_dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = import_with_failure
try:
    importlib.import_module('geo_infer_data.utils.duckdb_spatial')
except ImportError as error:
    assert str(error).startswith('installed DuckDB'), str(error)
    if {failure!r} == 'transitive_module_missing':
        assert isinstance(error, ModuleNotFoundError)
        assert error.name == 'broken_transitive_dependency'
    else:
        assert type(error) is ImportError
else:
    raise AssertionError('Installed DuckDB failure was suppressed')
print('original installed-DuckDB failure propagated')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "original installed-DuckDB failure propagated" in result.stdout


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:32610", None])
def test_native_geoparquet_preserves_declared_geometry_and_nulls(
    tmp_path, monkeypatch, crs
):
    """Native Parquet preserves a quoted primary name, zeros and missing geometry."""
    import geo_infer_data.utils.duckdb_spatial as backend

    name = 'shape"; DROP TABLE pretend; --'
    points = [shapely.Point(500000, 4649776), None, shapely.Point(500100, 4649800)]
    source = gpd.GeoDataFrame(
        {"identity": ["z", "m", "a"], "value": [0, 4, 7]}, geometry=points, crs=crs
    ).rename_geometry(name)
    original = source.copy(deep=True)
    path = tmp_path / "quoted'file.parquet"
    source.to_parquet(path)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("required native Parquet entered fallback")

    monkeypatch.setattr(backend, "_fallback_read_vector", forbidden)
    result = backend.read_cloud_native_vector(path, require_duckdb=True)
    assert result.geometry.name == name
    assert result.crs == source.crs
    assert result["identity"].tolist() == ["z", "m", "a"]
    assert result["value"].tolist() == [0, 4, 7]
    assert result.geometry.iloc[0].x == 500000
    assert result.geometry.iloc[0].y == 4649776
    assert result.geometry.iloc[1] is None
    assert result.geometry.iloc[2].x == 500100
    assert result.geometry.iloc[2].y == 4649800
    assert path.read_bytes() == before
    result.loc[0, "value"] = 99
    assert source.equals(original)


def test_base_geoparquet_fallback_reads_parquet(tmp_path):
    source = gpd.GeoDataFrame(
        {"value": [0, 7]}, geometry=[shapely.Point(1, 2), None], crs="EPSG:4326"
    )
    path = tmp_path / "tiny.parquet"
    source.to_parquet(path)
    result = read_cloud_native_vector(path, use_duckdb=False)
    assert result.equals(source)


def test_native_geoparquet_requires_geometry_metadata(tmp_path, monkeypatch):
    import pandas as pd
    import geo_infer_data.utils.duckdb_spatial as backend

    path = tmp_path / "plain.parquet"
    pd.DataFrame({"value": [0, 7]}).to_parquet(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("required native Parquet entered fallback")

    monkeypatch.setattr(backend, "_fallback_read_vector", forbidden)
    with pytest.raises(backend.DuckDBSpatialError) as raised:
        backend.read_cloud_native_vector(path, require_duckdb=True)
    assert "requires geo metadata" in str(raised.value.__cause__)


def test_native_geoparquet_omitted_crs_is_crs84(tmp_path, monkeypatch):
    import json
    import pyarrow.parquet as pq
    import geo_infer_data.utils.duckdb_spatial as backend

    source = gpd.GeoDataFrame(
        {"value": [0]}, geometry=[shapely.Point(1, 2)], crs="EPSG:4326"
    )
    path = tmp_path / "default-crs.parquet"
    source.to_parquet(path)
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    geo = json.loads(metadata[b"geo"])
    del geo["columns"][geo["primary_column"]]["crs"]
    metadata[b"geo"] = json.dumps(geo).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)

    def forbidden(*args, **kwargs):
        raise AssertionError("native default CRS entered fallback")

    monkeypatch.setattr(backend, "_fallback_read_vector", forbidden)
    result = backend.read_cloud_native_vector(path, require_duckdb=True)
    assert result.crs.to_string() == "OGC:CRS84"
    assert [axis.direction for axis in result.crs.axis_info] == ["east", "north"]
    assert (result.geometry.iloc[0].x, result.geometry.iloc[0].y) == (1, 2)
    assert result.crs.equals(gpd.read_parquet(path).crs)


def test_native_geoparquet_empty_and_secondary_geometries(tmp_path, monkeypatch):
    import geo_infer_data.utils.duckdb_spatial as backend

    source = gpd.GeoDataFrame(
        {"value": [0, 4, 7]},
        geometry=[shapely.Point(500000, 4649776), None, shapely.Point()],
        crs="EPSG:32610",
    ).rename_geometry("projected")
    source["geographic"] = gpd.GeoSeries(
        [shapely.Point(1, 2), None, shapely.Point(3, 4)], crs="EPSG:4326"
    )
    path = tmp_path / "secondary.parquet"
    source.to_parquet(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("native secondary geometry entered fallback")

    monkeypatch.setattr(backend, "_fallback_read_vector", forbidden)
    result = backend.read_cloud_native_vector(path, require_duckdb=True)
    assert result.geometry.name == "projected" and result.crs.to_epsg() == 32610
    assert result.geometry.iloc[1] is None and result.geometry.iloc[2].is_empty
    assert result["geographic"].crs.to_epsg() == 4326
    assert result["geographic"].iloc[1] is None
    assert (result["geographic"].iloc[2].x, result["geographic"].iloc[2].y) == (3, 4)


def test_native_geoparquet_zero_rows(tmp_path, monkeypatch):
    import geo_infer_data.utils.duckdb_spatial as backend

    source = gpd.GeoDataFrame({"value": []}, geometry=[], crs="EPSG:4326")
    path = tmp_path / "zero-rows.parquet"
    source.to_parquet(path)
    result = backend.read_cloud_native_vector(path, require_duckdb=True)
    assert len(result) == 0 and result.crs.to_epsg() == 4326
    assert result.columns.tolist() == ["value", "geometry"]


@pytest.mark.parametrize("index_kind", ["named", "range", "multi", "duplicate"])
def test_native_geoparquet_lossless_pandas_identity(tmp_path, monkeypatch, index_kind):
    import pandas as pd
    import geo_infer_data.utils.duckdb_spatial as backend

    timestamps = pd.DatetimeIndex(
        [
            "2026-11-01T08:30:00.123456789Z",
            "2026-11-01T09:30:00.123456790Z",
            "2026-11-01T09:30:00.123456791Z",
        ]
    )
    if index_kind == "named":
        index = pd.Index(
            ["sensor-z", "sensor-a", "sensor-q"], name="observation_identity"
        )
    elif index_kind == "range":
        index = pd.RangeIndex(10, 16, 2, name="observation_identity")
    elif index_kind == "multi":
        index = pd.MultiIndex.from_arrays(
            [["z", "a", "z"], timestamps.tz_convert("America/Los_Angeles")],
            names=["sensor", "arrival"],
        )
    else:
        index = pd.Index(["z", "z", "a"])
    source = gpd.GeoDataFrame(
        {
            "value": [0, 4, 7],
            "timestamp": timestamps,
            "local": timestamps.tz_convert("America/Los_Angeles"),
            "category": pd.Categorical(
                ["z", "a", "z"], categories=["unused", "z", "a"], ordered=True
            ),
            "label": pd.array(["first", None, "last"], dtype="string"),
            "nullable": pd.array([0.0, None, 7.0], dtype="Float64"),
        },
        geometry=[shapely.Point(1, 2), None, shapely.Point(3, -4)],
        crs="EPSG:4326",
        index=index,
    )
    path = tmp_path / "identity.parquet"
    source.to_parquet(path)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("lossless native Parquet entered fallback")

    monkeypatch.setattr(backend, "_fallback_read_vector", forbidden)
    actual = backend.read_cloud_native_vector(path, require_duckdb=True)
    pd.testing.assert_frame_equal(actual, source)
    pd.testing.assert_frame_equal(actual, gpd.read_parquet(path))
    assert actual.timestamp.iloc[1].value + 1 == actual.timestamp.iloc[2].value
    actual.iloc[0, 0] = 99
    pd.testing.assert_frame_equal(
        backend.read_cloud_native_vector(path, require_duckdb=True), source
    )
    assert path.read_bytes() == before


@pytest.mark.parametrize("require_native", [False, True])
def test_native_geoparquet_arrow_absence_names_integration_extra(
    tmp_path, monkeypatch, require_native
):
    from geo_infer_data.utils import dependencies
    import geo_infer_data.utils.duckdb_spatial as backend

    source = gpd.GeoDataFrame(
        {"value": [0]}, geometry=[shapely.Point(1, 2)], crs="EPSG:4326"
    )
    path = tmp_path / "arrow-missing.parquet"
    source.to_parquet(path)
    original_import = dependencies.import_module

    def import_without_arrow(name):
        if name == "pyarrow":
            raise ModuleNotFoundError("No module named pyarrow", name="pyarrow")
        return original_import(name)

    monkeypatch.setattr(dependencies, "import_module", import_without_arrow)
    expected_error = (
        backend.DuckDBSpatialError
        if require_native
        else dependencies.MissingOptionalDependency
    )
    with pytest.raises(expected_error) as raised:
        backend.read_cloud_native_vector(
            path, use_duckdb=require_native, require_duckdb=require_native
        )
    error = raised.value.__cause__ if require_native else raised.value
    assert "geo-infer-data[integrations]" in str(error)
