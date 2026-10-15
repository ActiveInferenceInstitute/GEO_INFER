"""Real vector files qualify GeoPandas default and explicit engine contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geo_infer_space.io.format_handlers import GeoJSONHandler, ShapefileHandler
from geo_infer_space.io.vector_io import VectorReader, VectorWriter


@pytest.fixture
def vector_features() -> gpd.GeoDataFrame:
    """Use short property names and real WGS84 coordinates in all formats."""
    return gpd.GeoDataFrame(
        {"name": ["café", "東京"], "value": [3, 7]},
        geometry=[Point(-122.4, 37.8), Point(139.7, 35.7)],
        crs="EPSG:4326",
    )


def assert_vector_features(
    actual: gpd.GeoDataFrame, expected: gpd.GeoDataFrame
) -> None:
    """Compare meaningful vector contents independently of serialization bytes."""
    assert actual.crs == expected.crs
    assert len(actual) == len(expected)
    assert actual["name"].tolist() == expected["name"].tolist()
    assert actual["value"].tolist() == expected["value"].tolist()
    assert actual.geometry.geom_equals_exact(expected.geometry, tolerance=1e-12).all()


def engine_options(engine: str | None) -> dict[str, str]:
    """Leave default engine selection to the installed GeoPandas version."""
    return {} if engine is None else {"engine": engine}


@pytest.mark.parametrize("engine", [None, "pyogrio"])
@pytest.mark.parametrize(
    ("handler_type", "extension"),
    [(GeoJSONHandler, ".geojson"), (ShapefileHandler, ".shp")],
)
def test_handler_roundtrip(
    tmp_path: Path, vector_features: gpd.GeoDataFrame, engine, handler_type, extension
) -> None:
    handler = handler_type()
    path = tmp_path / "nested" / f"features{extension}"
    options = engine_options(engine)
    handler.write(vector_features, path, encoding="UTF-8", index=False, **options)
    assert_vector_features(handler.read(path, **options), vector_features)
    validation = handler.validate(path)
    assert validation["valid"] is True
    assert validation["error"] is None
    assert validation["metadata"]["num_features"] == len(vector_features)
    assert validation["metadata"]["crs"] == str(vector_features.crs)
    if extension == ".shp":
        assert all(
            path.with_suffix(suffix).is_file()
            for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
        )


@pytest.mark.parametrize("engine", [None, "pyogrio"])
@pytest.mark.parametrize("extension", [".geojson", ".shp", ".gpkg"])
def test_reader_writer_roundtrip(
    tmp_path: Path, vector_features: gpd.GeoDataFrame, engine, extension
) -> None:
    path = tmp_path / "nested" / f"features{extension}"
    options = engine_options(engine)
    if extension == ".gpkg":
        options["layer"] = "observations"
    VectorWriter().write(
        vector_features, path, encoding="UTF-8", index=False, **options
    )
    read_options = dict(options)
    if extension == ".geojson" and engine is None:
        # The default GeoJSON reader retains its constructor-based interface.
        read_options["crs"] = "EPSG:4326"
    assert_vector_features(VectorReader().read(path, **read_options), vector_features)
    if extension == ".geojson":
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert [feature["properties"]["name"] for feature in payload["features"]] == [
            "café",
            "東京",
        ]


def test_geojson_writer_preserves_explicit_driver_options(tmp_path: Path) -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(1.234567, 2.345678)], crs="EPSG:4326")
    path = tmp_path / "precision.geojson"
    VectorWriter().write(frame, path, engine="pyogrio", coordinate_precision=3)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["features"][0]["geometry"]["coordinates"] == [1.235, 2.346]


@pytest.mark.parametrize("handler_type", [GeoJSONHandler, ShapefileHandler])
def test_handlers_reject_missing_files(tmp_path: Path, handler_type) -> None:
    handler = handler_type()
    path = tmp_path / (
        "missing.geojson" if handler_type is GeoJSONHandler else "missing.shp"
    )
    with pytest.raises(FileNotFoundError):
        handler.read(path)
    validation = handler.validate(path)
    assert validation["valid"] is False
    assert validation["error"]


@pytest.mark.parametrize("extension", [".geojson", ".shp", ".gpkg"])
def test_reader_rejects_missing_files(tmp_path: Path, extension: str) -> None:
    with pytest.raises(FileNotFoundError):
        VectorReader().read(tmp_path / f"missing{extension}")


@pytest.mark.parametrize("engine", [None, "pyogrio"])
@pytest.mark.parametrize("sidecar", [".shx", ".dbf"])
def test_shapefile_validation_rejects_missing_sidecars(
    tmp_path: Path, vector_features: gpd.GeoDataFrame, engine, sidecar: str
) -> None:
    path = tmp_path / "features.shp"
    handler = ShapefileHandler()
    handler.write(vector_features, path, index=False, **engine_options(engine))
    path.with_suffix(sidecar).unlink()

    validation = handler.validate(path)

    assert validation["valid"] is False
    assert sidecar in validation["error"]
    assert validation["metadata"]["sidecar_files_present"][sidecar] is False


@pytest.mark.parametrize("engine", [None, "pyogrio"])
@pytest.mark.parametrize(
    ("handler_type", "extension"),
    [(GeoJSONHandler, ".geojson"), (ShapefileHandler, ".shp")],
)
def test_handlers_reject_corrupt_files(
    tmp_path: Path, engine, handler_type, extension
) -> None:
    path = tmp_path / f"corrupt{extension}"
    path.write_text("invalid vector file", encoding="utf-8")
    handler = handler_type()
    with pytest.raises((ValueError, RuntimeError)):
        handler.read(path, **engine_options(engine))
    validation = handler.validate(path)
    assert validation["valid"] is False
    assert validation["error"]


@pytest.mark.parametrize("engine", [None, "pyogrio"])
@pytest.mark.parametrize("extension", [".geojson", ".shp", ".gpkg"])
def test_reader_rejects_corrupt_files(tmp_path: Path, engine, extension) -> None:
    path = tmp_path / f"corrupt{extension}"
    path.write_text("invalid vector file", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        VectorReader().read(path, **engine_options(engine))


@pytest.mark.parametrize(
    ("handler_type", "extension"),
    [(GeoJSONHandler, ".geojson"), (ShapefileHandler, ".shp")],
)
def test_explicit_fiona_selection(
    tmp_path: Path, vector_features: gpd.GeoDataFrame, handler_type, extension
) -> None:
    """Exercise the real optional engine or its actionable absence error."""
    path = tmp_path / f"features{extension}"
    handler = handler_type()
    if importlib.util.find_spec("fiona") is None:
        with pytest.raises(ImportError, match="fiona"):
            handler.write(vector_features, path, engine="fiona", index=False)
        handler.write(vector_features, path, engine="pyogrio", index=False)
        with pytest.raises(ImportError, match="fiona"):
            handler.read(path, engine="fiona")
    else:
        handler.write(
            vector_features, path, engine="fiona", encoding="UTF-8", index=False
        )
        assert_vector_features(handler.read(path, engine="fiona"), vector_features)


@pytest.mark.parametrize("extension", [".geojson", ".shp", ".gpkg"])
def test_reader_writer_explicit_fiona_selection(
    tmp_path: Path, vector_features: gpd.GeoDataFrame, extension
) -> None:
    path = tmp_path / f"features{extension}"
    writer, reader = VectorWriter(), VectorReader()
    if importlib.util.find_spec("fiona") is None:
        with pytest.raises(ImportError, match="fiona"):
            writer.write(vector_features, path, engine="fiona", index=False)
        writer.write(vector_features, path, engine="pyogrio", index=False)
        with pytest.raises(ImportError, match="fiona"):
            reader.read(path, engine="fiona")
    else:
        writer.write(
            vector_features, path, engine="fiona", encoding="UTF-8", index=False
        )
        assert_vector_features(reader.read(path, engine="fiona"), vector_features)
