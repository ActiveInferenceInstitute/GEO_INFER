"""Native rendering contracts for scientific map composition and exports."""

import json
import re
import xml.etree.ElementTree as ET

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import MultiLineString, MultiPolygon, Point, Polygon

from geo_infer_art.core.visualization.geo_art import GeoArt
from geo_infer_art.core.visualization.map_styling import MapStyle


@pytest.fixture(autouse=True)
def release_test_figures():
    """Release only the pyplot figures allocated by each native rendering test."""
    before = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


@pytest.mark.parametrize(
    "shape,layout,expected",
    [
        ((3, 8, 9), "auto", (8, 9, 3)),
        ((4, 8, 9), "band_first", (8, 9, 4)),
        ((1, 8, 9), "band_first", (8, 9)),
        ((3, 8, 3), "auto", (3, 8, 3)),
        ((8, 9, 1), "channel_last", (8, 9)),
    ],
)
def test_raster_layout_preserves_orientation_and_channels(shape, layout, expected):
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    with GeoArt(data) as art:
        art.apply_style(raster_layout=layout)
        shown = art._ax.images[0].get_array()
        assert shown.shape == expected
        normalized = (
            np.moveaxis(data, 0, -1)
            if layout == "band_first" or shape == (3, 8, 9)
            else data
        )
        if normalized.shape[-1] == 1:
            normalized = normalized[..., 0]
        np.testing.assert_array_equal(shown, normalized)


def test_band_selection_and_shared_scalar_limits():
    data = np.arange(5 * 8 * 9).reshape(5, 8, 9)
    with GeoArt(data) as art:
        art.apply_style(raster_layout="band_first", raster_band=3, vmin=0, vmax=500)
        np.testing.assert_array_equal(art._ax.images[0].get_array(), data[3])
        assert art._ax.images[0].get_clim() == (0, 500)
        art.get_colorbar("Elevation (m)")
        count = len(art._figure.axes)
        art.get_colorbar("Updated elevation (m)")
        assert len(art._figure.axes) == count
        assert art._colorbar.ax.get_ylabel() == "Updated elevation (m)"
    for options in (
        {"raster_band": 5},
        {"raster_band": True},
        {"vmin": np.nan},
        {"vmin": 1, "vmax": 0},
    ):
        with pytest.raises(ValueError):
            GeoArt(data).apply_style(**options)


def test_loaded_geotiff_retains_nodata_bounds_and_band_order(tmp_path):
    path = tmp_path / "scalar.tif"
    data = np.arange(72, dtype=np.int16).reshape(1, 8, 9)
    data[0, 2, 3] = -9999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=9,
        height=8,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=from_origin(100, 500, 10, 20),
        nodata=-9999,
    ) as dataset:
        dataset.write(data)
    with GeoArt.load_raster(str(path)) as art:
        art.apply_style()
        shown = art._ax.images[0]
        assert shown.get_array().shape == (8, 9)
        assert shown.get_array().mask[2, 3]
        assert tuple(shown.get_extent()) == (100, 190, 340, 500)
        assert art.crs == "EPSG:3857"


def test_categorical_and_scalar_vector_legends():
    data = gpd.GeoDataFrame(
        {"kind": ["Forest", "Water"], "value": [2, 8]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs=4326,
    )
    with GeoArt(data) as art:
        art.apply_style(legend=True)
        assert [text.get_text() for text in art._ax.get_legend().get_texts()] == [
            "Forest",
            "Water",
        ]
        art.apply_style(column="value", vmin=0, vmax=10).get_colorbar("Temperature (C)")
        assert art._ax.collections[0].get_clim() == (0, 10)
        assert art._colorbar.ax.get_ylabel() == "Temperature (C)"


def test_repeated_style_and_context_only_close_owned_figures():
    unrelated, _ = plt.subplots()
    # Native zero disables the warning; it must never mean "close all figures".
    with plt.rc_context({"figure.max_open_warning": 0}):
        with GeoArt(np.arange(25).reshape(5, 5)) as art:
            art.apply_style()
            previous = art._figure
            art.apply_style(style="minimal")
            assert not plt.fignum_exists(previous.number)
            assert plt.fignum_exists(unrelated.number)
            own = art._figure
        assert not plt.fignum_exists(own.number)
    assert plt.fignum_exists(unrelated.number)


def test_caller_axes_composition_survives_save_and_close(tmp_path):
    fig, axes = plt.subplots(1, 2)
    with GeoArt(np.arange(25).reshape(5, 5)) as art:
        art.apply_style(ax=axes[1], vmin=0, vmax=30, title="Observed elevation")
        assert art._figure is fig
        assert not axes[0].images
        assert len(axes[1].images) == 1
        art.save(str(tmp_path / "composed.png"), dpi=60)
    assert plt.fignum_exists(fig.number)


def test_failed_style_releases_new_figure_without_losing_previous():
    with GeoArt(np.arange(25).reshape(5, 5)) as art:
        art.apply_style()
        previous = art._figure
        before = set(plt.get_fignums())
        with pytest.raises(ValueError):
            art.apply_style(
                map_style=MapStyle(parameters={"background": "invalid-color"})
            )
        assert art._figure is previous
        assert set(plt.get_fignums()) == before


def test_svg_exports_are_repeatable_selectable_and_described(tmp_path):
    with GeoArt(np.arange(25).reshape(5, 5)) as art:
        art.apply_style(title="Observed elevation (m)")
        first, second = tmp_path / "first.svg", tmp_path / "second.svg"
        art.export_svg(
            str(first), metadata={"Description": "Twenty-five measured raster cells"}
        )
        art.export_svg(
            str(second), metadata={"Description": "Twenty-five measured raster cells"}
        )
        assert first.read_bytes() == second.read_bytes()
        svg = ET.fromstring(first.read_text())
        assert (
            svg.find("{http://www.w3.org/2000/svg}title").text
            == "Observed elevation (m)"
        )
        assert svg.findall(".//{http://www.w3.org/2000/svg}text")
        assert "Twenty-five measured raster cells" in first.read_text()
        output = tmp_path / "transparent.png"
        art.save(
            str(output),
            dpi=60,
            close=False,
            transparent=True,
            metadata={"Description": "Raster cells"},
        )
        with Image.open(output) as png:
            assert png.info["Description"] == "Raster cells"
            assert png.mode == "RGBA"


def test_animation_changes_actual_pixels_and_preserves_current_figure(tmp_path):
    unrelated, _ = plt.subplots()
    with GeoArt(np.arange(64).reshape(8, 8)) as art:
        art.apply_style(title="Current map")
        current = art._figure
        before = set(plt.get_fignums())
        path = art.create_animation(
            str(tmp_path / "styles.gif"), ["default", "neon"], duration=1, fps=4
        )
        assert art._figure is current
        assert set(plt.get_fignums()) == before
        assert plt.fignum_exists(unrelated.number)
        with Image.open(path) as gif:
            assert gif.n_frames >= 2
            gif.seek(0)
            first = np.asarray(gif.convert("RGB")).copy()
            gif.seek(gif.n_frames - 1)
            assert np.count_nonzero(first != np.asarray(gif.convert("RGB"))) > 1000


def _web_geometries(markup):
    payloads = re.findall(r"geo_json_[a-z0-9]+_add\((\{.*?\})\);", markup, re.S)
    objects = [json.loads(payload) for payload in payloads]
    return [
        feature["geometry"] if feature["type"] == "Feature" else feature
        for item in objects
        for feature in (
            item["features"] if item["type"] == "FeatureCollection" else [item]
        )
    ]


def test_web_map_reprojects_preserves_holes_multipart_and_escapes_text(tmp_path):
    polygon = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]]
    )
    line = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
    data = gpd.GeoDataFrame(
        {"name": ["<script>alert(1)</script>${alert(2)}`", "Lines"]},
        geometry=[MultiPolygon([polygon]), line],
        crs=4326,
    ).to_crs(3857)
    original = data.copy()
    path = tmp_path / "map.html"
    GeoArt(data).create_interactive_web_map(str(path), tiles=None)
    markup = path.read_text()
    geometries = _web_geometries(markup)
    assert [geometry["type"] for geometry in geometries] == [
        "MultiPolygon",
        "MultiLineString",
    ]
    assert len(geometries[0]["coordinates"][0]) == 2
    np.testing.assert_allclose(geometries[1]["coordinates"][1][1], [3, 3])
    assert "<script>alert(1)</script>" not in markup
    assert "&lt;script&gt;" in markup
    assert "${alert(2)}" not in markup
    assert "&#36;{alert(2)}&#96;" in markup
    assert data.equals(original)


def test_web_map_rejects_unsupported_data_and_missing_crs(tmp_path):
    for data in (np.ones((5, 5)), gpd.GeoDataFrame(geometry=[Point(0, 0)])):
        path = tmp_path / "invalid.html"
        with pytest.raises(ValueError):
            GeoArt(data).create_interactive_web_map(str(path))
        assert not path.exists()


def test_plotly_maps_use_geographic_coordinates_and_native_feature_ids():
    points = gpd.GeoDataFrame(
        {"value": [2, 3]}, geometry=[Point(1, 2), Point(3, 4)], crs=4326
    ).to_crs(3857)
    fig = GeoArt(points).create_plotly_visualization(
        "scatter", color_column="value", title="Points"
    )
    np.testing.assert_allclose(fig.data[0].lat, [2, 4])
    np.testing.assert_allclose(fig.data[0].lon, [1, 3])
    assert fig.layout.title.text == "Points"
    polygons = gpd.GeoDataFrame(
        {"value": [2, 3]},
        index=["duplicate", "duplicate"],
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 0), (3, 0), (3, 1)])],
        crs=4326,
    )
    trace = (
        GeoArt(polygons)
        .create_plotly_visualization("choropleth", color_column="value")
        .data[0]
    )
    assert list(trace.locations) == ["0", "1"]
    assert [feature["id"] for feature in trace.geojson["features"]] == ["0", "1"]
    with pytest.raises(ValueError, match="Point"):
        GeoArt(polygons).create_plotly_visualization("scatter")


def test_plotly_scalar_raster_uses_selected_band_and_nodata():
    data = np.ma.array(np.arange(5 * 8 * 9, dtype=float).reshape(5, 8, 9), mask=False)
    data.mask[2, 1, 1] = True
    trace = (
        GeoArt(data)
        .create_plotly_visualization(
            "heatmap", raster_layout="band_first", raster_band=2, vmin=0, vmax=400
        )
        .data[0]
    )
    assert np.isnan(trace.z[1, 1])
    assert trace.z.shape == (8, 9)
    assert (trace.zmin, trace.zmax) == (0, 400)
    surface = (
        GeoArt(data)
        .create_plotly_visualization("3d", raster_layout="band_first", raster_band=2)
        .data[0]
    )
    assert surface.x.shape == surface.z.shape == (8, 9)
    with pytest.raises(ValueError, match="scalar"):
        GeoArt(np.zeros((8, 9, 3))).create_plotly_visualization("heatmap")


def test_web_map_uses_active_geometry_column_and_rejects_invalid_latitude(tmp_path):
    data = gpd.GeoDataFrame(geometry=[Point(1, 2)], crs=4326).rename_geometry(
        "footprint"
    )
    path = tmp_path / "custom.html"
    GeoArt(data).create_interactive_web_map(str(path), tiles=None)
    assert _web_geometries(path.read_text())[0]["coordinates"] == [1, 2]
    invalid = gpd.GeoDataFrame(geometry=[Point(1, 100)], crs=4326)
    with pytest.raises(ValueError, match="WGS84"):
        GeoArt(invalid).create_interactive_web_map(str(tmp_path / "bad.html"))


def test_rotated_geotiff_uses_native_affine_coordinates(tmp_path):
    from rasterio.transform import Affine

    transform = (
        Affine.translation(100, 500) * Affine.rotation(25) * Affine.scale(10, -20)
    )
    path = tmp_path / "rotated.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=9,
        height=8,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=transform,
    ) as dataset:
        dataset.write(np.arange(72, dtype=np.int16).reshape(1, 8, 9))
    with GeoArt.load_raster(str(path)) as art:
        art.apply_style()
        image = art._ax.images[0]
        point = art._ax.transData.inverted().transform(
            image.get_transform().transform((0.5, 0.5))
        )
        np.testing.assert_allclose(point, transform * (0.5, 0.5))
        bounds = art.metadata["bounds"]
        np.testing.assert_allclose(art._ax.get_xlim(), [bounds.left, bounds.right])
        np.testing.assert_allclose(art._ax.get_ylim(), [bounds.bottom, bounds.top])
