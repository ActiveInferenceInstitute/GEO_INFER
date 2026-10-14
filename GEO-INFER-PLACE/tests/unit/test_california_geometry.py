from __future__ import annotations

import zipfile
from pathlib import Path

import h3
import numpy as np
import pytest
from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    ObservatoryConfig,
)
from geo_infer_place.locations.california.geometry import (
    County,
    county_weights,
    drought_collection,
    geometry_record,
    normalize_geometry,
    point_patch_index,
)
from pyproj import Transformer
from shapely import segmentize
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
from shapely.strtree import STRtree


def test_metric_point_attribution_rejects_spherical_boundary_misplacement() -> None:
    """Reproduce the independent boundary finding and conserve one assigned event."""
    point = Point(-124.0023686535118, 40.30565152595539)
    identifiers = sorted(["8528004bfffffff", "85280297fffffff"])
    projector = Transformer.from_crs(4326, 3310, always_xy=True)
    patches = [
        transform(
            projector.transform,
            segmentize(
                Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(identifier)]),
                0.02,
            ),
        )
        for identifier in identifiers
    ]
    metric = transform(projector.transform, point)
    spherical = identifiers.index(h3.latlng_to_cell(point.y, point.x, 5))
    assert not patches[spherical].covers(metric)
    index = point_patch_index(metric, STRtree(patches))
    assert identifiers[index] == "8528004bfffffff"
    assert patches[index].contains(metric)
    counts = np.zeros(len(patches), dtype=int)
    counts[index] += 1
    assert counts.sum() == 1 and counts[spherical] == 0


def test_point_attribution_shared_boundary_and_uncovered_cases() -> None:
    """Choose a deterministic metric boundary owner and reject uncovered or empty points."""
    tree = STRtree([box(0, 0, 1, 1), box(1, 0, 2, 1)])
    assert point_patch_index(Point(1, 0.5), tree) == 0
    with pytest.raises(CaliforniaError, match="not covered"):
        point_patch_index(Point(3, 0.5), tree)
    with pytest.raises(CaliforniaError, match="finite"):
        point_patch_index(Point(), tree)


def test_polygon_repair_is_explicit_and_retains_a_receipt() -> None:
    """Require explicit polygon repair and preserve the changed topology record."""
    crossed = geometry_record(
        Polygon([(-121, 38), (-120, 39), (-121, 39), (-120, 38), (-121, 38)])
    )
    with pytest.raises(CaliforniaError, match="repair disabled"):
        normalize_geometry(crossed, "crossed", repair=False)
    repaired = normalize_geometry(crossed, "crossed", repair=True)
    assert repaired.geometry.is_valid and repaired.geometry.area > 0
    assert repaired.repair is not None and repaired.repair.source_id == "crossed"
    valid = normalize_geometry(
        geometry_record(box(-121, 38, -120, 39)), "valid", repair=False
    )
    assert valid.repair is None


def test_empty_and_non_geographic_coordinates_are_refused() -> None:
    """Reject empty, malformed, nonfinite and nongeographic geometry coordinates."""
    with pytest.raises(CaliforniaError):
        normalize_geometry(geometry_record(Polygon()), "empty", repair=True)
    with pytest.raises(CaliforniaError):
        normalize_geometry(
            geometry_record(box(180, 38, 181, 39)), "outside", repair=True
        )
    record = geometry_record(box(-121, 38, -120, 39)).model_dump()
    for invalid in (True, "38", None, {"latitude": 38}, float("nan")):
        corrupted = dict(record)
        corrupted["coordinates"] = [[[invalid, 38], [-120, 39], [-121, 38]]]
        from geo_infer_place.locations.california.geometry import Geometry

        with pytest.raises(CaliforniaError):
            normalize_geometry(
                Geometry.model_validate(corrupted), "malformed", repair=True
            )


def test_overlapping_county_weights_conserve_allocated_area() -> None:
    """Conserve attribution through overlaps and reject gaps hidden by double counting."""
    geometries = [box(0, 0, 2, 1), box(1, 0, 3, 1)]
    counties = [
        County(
            fips=f"0600{index + 1}",
            name=f"Numerical county {index}",
            geometry=geometry_record(geometry),
            area_km2=geometry.area,
        )
        for index, geometry in enumerate(geometries)
    ]
    patch = box(0, 0, 3, 1)
    weights, ratio = county_weights(patch, counties, geometries)
    assert ratio == pytest.approx(4 / 3)
    assert tuple(item.fraction for item in weights) == (0.5, 0.5)
    assert sum(item.fraction * patch.area for item in weights) == pytest.approx(
        patch.area
    )
    with pytest.raises(CaliforniaError):
        county_weights(box(5, 5, 6, 6), counties, geometries)
    with pytest.raises(CaliforniaError, match="do not cover"):
        county_weights(box(0, 0, 4, 1), counties, geometries)


@pytest.mark.parametrize(
    "condition", ["few", "duplicate", "traversal", "symlink", "projection"]
)
def test_drought_archive_controls_precede_gis_read(
    tmp_path: Path, condition: str
) -> None:
    """Reject incomplete, duplicate, traversing and symlinked shapefile archives."""
    path = tmp_path / "unsafe.zip"
    names = ["a.shp", "a.shx", "a.dbf", "a.prj"]
    if condition == "few":
        names = names[:2]
    elif condition == "duplicate":
        names[1] = names[0]
    elif condition == "traversal":
        names[0] = "../a.shp"
    elif condition == "projection":
        names[-1] = "a.txt"
    with zipfile.ZipFile(path, "x") as archive:
        for index, name in enumerate(names):
            item = zipfile.ZipInfo(name)
            if condition == "symlink" and index == 0:
                item.external_attr = 0o120777 << 16
            if condition == "duplicate" and index == 1:
                with pytest.warns(UserWarning, match="Duplicate"):
                    archive.writestr(item, b"fixture")
            else:
                archive.writestr(item, b"fixture")
    with pytest.raises(CaliforniaError):
        drought_collection(path)


@pytest.mark.parametrize(
    "condition",
    [
        "training",
        "seeds",
        "precision",
        "coverage",
        "projection",
        "quota",
        "work",
        "unknown",
        "boolean",
    ],
)
def test_strict_design_rejects_invalid_controls(condition: str) -> None:
    """Reject invalid chronology, sensor ordering, work budgets and projection controls."""
    data = ObservatoryConfig().model_dump()
    if condition == "training":
        data["first_year"] = data["target_year"] - 1
    elif condition == "seeds":
        data["seeds"] = (0, 0)
    elif condition == "precision":
        data["detailed_accuracy"] = data["coarse_accuracy"]
    elif condition == "coverage":
        data["detailed_coverage"] = data["coarse_coverage"]
    elif condition == "projection":
        data["projection_segment_degrees"] = 0
    elif condition == "quota":
        data["budget"] = 1
    elif condition == "work":
        data["seeds"], data["rounds"], data["max_cells"] = (0, 1, 2, 3), 24, 10000
    elif condition == "unknown":
        data["unused"] = 1
    else:
        data["resolution"] = True
    with pytest.raises(ValueError):
        ObservatoryConfig.model_validate(data)
    assert np.isfinite(ObservatoryConfig().coupling)
