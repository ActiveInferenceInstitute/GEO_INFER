"""Measured, clipped H3 geometry and temporally separate empirical indicators."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import geopandas as gpd
import h3
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from pyproj import Transformer
from shapely import get_parts, make_valid, segmentize
from shapely.geometry import Point, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

from geo_infer_place.locations.california.acquisition import check_sources
from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    Digest,
    Hazard,
    ObservatoryConfig,
    StrictModel,
    digest,
    last_tuesday,
    strict_bytes,
    write_record,
)


class Geometry(StrictModel):
    type: Literal["Polygon", "MultiPolygon", "Point"]
    coordinates: list[JsonValue]


class GeoFeature(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    geometry: Geometry
    properties: dict[str, JsonValue]
    id: str | int | None = None


class Collection(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    type: Literal["FeatureCollection"]
    features: tuple[GeoFeature, ...]


class CountyProperties(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    OBJECTID: int
    County: str
    FIPS: str = Field(pattern=r"^[0-9]{3}$")


class FireProperties(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    OBJECTID: int
    YEAR_: int
    STATE: Literal["CA"]


class EarthquakeProperties(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    mag: float = Field(allow_inf_nan=False)
    time: int
    type: Literal["earthquake"]


class DroughtProperties(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    DM: int = Field(ge=0, le=4)


class Repair(StrictModel):
    source_id: str
    original_type: str
    resulting_type: str


class CountyShare(StrictModel):
    fips: str = Field(pattern=r"^06[0-9]{3}$")
    fraction: float = Field(gt=0, le=1.00000001, allow_inf_nan=False)


class County(StrictModel):
    fips: str = Field(pattern=r"^06[0-9]{3}$")
    name: str
    geometry: Geometry
    area_km2: float = Field(gt=0, allow_inf_nan=False)


class Cell(StrictModel):
    id: str = Field(pattern=r"^[0-9a-f]{15}$")
    parent: str = Field(pattern=r"^[0-9a-f]{15}$")
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    area_km2: float = Field(gt=0, allow_inf_nan=False)
    geometry: Geometry
    county_shares: tuple[CountyShare, ...]
    raw_county_coverage_ratio: float = Field(ge=0.99999999, allow_inf_nan=False)


class Indicator(StrictModel):
    hazard: Hazard
    unit: str
    years: tuple[int, ...]
    dates: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    positive: tuple[tuple[int, ...], ...]
    prior: tuple[float, ...]
    source_records: tuple[int, ...]


class Landscape(StrictModel):
    schema_version: Literal["geo-infer-place/california-landscape/1"]
    config: ObservatoryConfig
    source_receipt_sha256: Digest
    projection: Literal["EPSG:3310"]
    counties: tuple[County, ...]
    cells: tuple[Cell, ...]
    edges: tuple[tuple[int, int], ...]
    indicators: tuple[Indicator, ...]
    repairs: tuple[Repair, ...]
    administrative_area_km2: float = Field(gt=0, allow_inf_nan=False)
    uncovered_fraction: float = Field(ge=0, le=1e-7, allow_inf_nan=False)
    county_overlap_km2: float = Field(ge=0, allow_inf_nan=False)
    county_weighting: Literal["normalized-intersection-area"]

    @model_validator(mode="after")
    def carrier_integrity(self) -> Self:
        count = len(self.cells)
        county_ids = {county.fips for county in self.counties}
        if len(self.counties) != 58 or len(county_ids) != 58:
            raise CaliforniaError("landscape requires 58 distinct counties")
        if not 1 <= count <= self.config.max_cells:
            raise CaliforniaError("invalid landscape carrier size")
        identifiers = tuple(cell.id for cell in self.cells)
        if identifiers != tuple(sorted(set(identifiers))):
            raise CaliforniaError("H3 carrier must be distinct and sorted")
        expected_edges: list[tuple[int, int]] = []
        lookup = {identifier: index for index, identifier in enumerate(identifiers)}
        for index, cell in enumerate(self.cells):
            if (
                not h3.is_valid_cell(cell.id)
                or h3.get_resolution(cell.id) != self.config.resolution
                or h3.cell_to_parent(cell.id, self.config.parent_resolution)
                != cell.parent
            ):
                raise CaliforniaError("invalid H3 resolution or hierarchy")
            shares = cell.county_shares
            if (
                not shares
                or len({share.fips for share in shares}) != len(shares)
                or any(share.fips not in county_ids for share in shares)
                or not np.isclose(
                    sum(share.fraction for share in shares), 1, atol=1e-12, rtol=0
                )
            ):
                raise CaliforniaError("county allocation weights do not conserve area")
            expected_edges.extend(
                (index, lookup[neighbor])
                for neighbor in h3.grid_disk(cell.id, 1)
                if neighbor in lookup and index < lookup[neighbor]
            )
        if self.edges != tuple(sorted(expected_edges)):
            raise CaliforniaError("landscape adjacency differs from its H3 carrier")
        if tuple(item.hazard for item in self.indicators) != tuple(Hazard):
            raise CaliforniaError(
                "landscape must contain each indicator in canonical order"
            )
        years = tuple(range(self.config.first_year, self.config.target_year + 1))
        for indicator in self.indicators:
            values, positive, prior = (
                np.asarray(item)
                for item in (indicator.values, indicator.positive, indicator.prior)
            )
            if (
                indicator.years != years
                or len(indicator.dates) != len(years)
                or len(indicator.source_records) != len(years)
                or any(number < 0 for number in indicator.source_records)
                or values.shape != (len(years), count)
                or positive.shape != values.shape
                or prior.shape != (count,)
                or not np.all(np.isfinite(values))
                or np.any(values < 0)
                or not np.all(np.isin(positive, (0, 1)))
                or not np.all(np.isfinite(prior))
                or np.any(prior <= 0)
                or np.any(prior >= 1)
            ):
                raise CaliforniaError(
                    "indicator carrier, support or temporal roster is invalid"
                )
            threshold = {
                Hazard.FIRE: self.config.fire_fraction,
                Hazard.SEISMIC: 1,
                Hazard.DROUGHT: self.config.drought_fraction,
            }[indicator.hazard]
            expected = (values >= threshold).astype(int)
            training = expected[:-1].astype(float)
            expected_prior = (
                training.sum(axis=0)
                + self.config.prior_strength * training.mean()
                + 0.5
            ) / (len(training) + self.config.prior_strength + 1)
            if not np.array_equal(positive, expected) or not np.allclose(
                prior, expected_prior, atol=1e-14, rtol=0
            ):
                raise CaliforniaError(
                    "indicator labels or strictly earlier-year priors differ"
                )
            if indicator.hazard != Hazard.SEISMIC and np.any(values > 1):
                raise CaliforniaError("area fractions exceed one")
        return self


@dataclass(frozen=True, slots=True)
class NormalGeometry:
    geometry: BaseGeometry
    repair: Repair | None


def _coordinate_check(value: JsonValue) -> None:
    match value:
        case bool() | str() | None | dict():
            raise CaliforniaError("geometry coordinates must be finite numbers")
        case int() | float():
            if not np.isfinite(value):
                raise CaliforniaError("geometry coordinates must be finite numbers")
        case list():
            for child in value:
                _coordinate_check(child)


def normalize_geometry(
    raw: Geometry, source_id: str, *, repair: bool
) -> NormalGeometry:
    """Reject malformed coordinates; record every explicitly enabled polygon repair."""
    _coordinate_check(raw.coordinates)
    geometry = shape(raw.model_dump())
    if geometry.is_empty or not np.all(np.isfinite(geometry.bounds)):
        raise CaliforniaError("source geometry is empty or nonfinite")
    x1, y1, x2, y2 = geometry.bounds
    if x1 < -180 or x2 > 180 or y1 < -90 or y2 > 90:
        raise CaliforniaError("source geometry is not WGS84 longitude/latitude")
    if geometry.is_valid:
        return NormalGeometry(geometry, None)
    if not repair:
        raise CaliforniaError(f"invalid geometry {source_id}; explicit repair disabled")
    fixed = make_valid(geometry)
    pieces = [
        part
        for part in get_parts(fixed)
        if part.geom_type in {"Polygon", "MultiPolygon"}
    ]
    fixed = unary_union(pieces)
    if fixed.is_empty or not fixed.is_valid:
        raise CaliforniaError(f"geometry repair cannot retain polygons: {source_id}")
    return NormalGeometry(
        fixed,
        Repair(
            source_id=source_id,
            original_type=geometry.geom_type,
            resulting_type=fixed.geom_type,
        ),
    )


def geometry_record(geometry: BaseGeometry) -> Geometry:
    return Geometry.model_validate_json(json.dumps(mapping(geometry), allow_nan=False))


def county_weights(
    patch: BaseGeometry, counties: list[County], projected_counties: list[BaseGeometry]
) -> tuple[tuple[CountyShare, ...], float]:
    """Normalize intersected county areas; retain source overlap and projection discrepancy."""
    pieces = [patch.intersection(county) for county in projected_counties]
    intersections = [piece.area for piece in pieces]
    total = sum(intersections)
    if not np.isfinite(total) or total <= 0 or patch.area <= 0:
        raise CaliforniaError("county attribution does not intersect an H3 patch")
    if patch.difference(unary_union(pieces)).area > patch.area * 1e-8:
        raise CaliforniaError("county intersections do not cover the metric H3 patch")
    return (
        tuple(
            CountyShare(fips=county.fips, fraction=area / total)
            for county, area in zip(counties, intersections, strict=True)
            if area > 0
        ),
        total / patch.area,
    )


def drought_collection(path: Path) -> Collection:
    """Read the captured NDMC ZIP without extracting or accepting unsafe members."""
    with zipfile.ZipFile(path) as archive:
        files = archive.infolist()
        if (
            not 4 <= len(files) <= 16
            or sum(item.file_size for item in files) > 64 * 1024 * 1024
        ):
            raise CaliforniaError("drought archive exceeds member or expansion budget")
        names = [item.filename for item in files]
        if len(set(names)) != len(names) or any(
            PurePosixPath(name).name != name or "\\" in name for name in names
        ):
            raise CaliforniaError("drought archive members must be distinct flat paths")
        if any((item.external_attr >> 16) & 0o170000 == 0o120000 for item in files):
            raise CaliforniaError("drought archive contains a symlink")
        if not all(
            any(name.endswith(suffix) for name in names)
            for suffix in (".shp", ".shx", ".dbf", ".prj")
        ):
            raise CaliforniaError(
                "drought archive lacks required geometry/projection files"
            )
    frame = gpd.read_file(f"zip://{path.resolve()}")
    if frame.crs is None or "DM" not in frame:
        raise CaliforniaError("drought source lacks its CRS or classification")
    return Collection.model_validate_json(frame.to_crs("EPSG:4326").to_json())


def point_patch_index(point: BaseGeometry, tree: STRtree) -> int:
    """Use metric containment; shared boundaries select the first sorted carrier index."""
    if (
        not isinstance(point, Point)
        or point.is_empty
        or not np.all(np.isfinite(point.bounds))
    ):
        raise CaliforniaError("point attribution requires finite metric coordinates")
    candidates = tree.query(point, predicate="intersects")
    if not len(candidates):
        raise CaliforniaError("selected earthquake is not covered by an analysis patch")
    return int(np.min(candidates))


def prepare_landscape(config: ObservatoryConfig, sources: Path, output: Path) -> Path:
    """Rasterize real records and use only strictly earlier years for empirical priors."""
    receipt = check_sources(sources)
    if (
        config.first_year != receipt.config.first_year
        or config.target_year != receipt.config.target_year
        or config.earthquake_magnitude != receipt.config.earthquake_magnitude
        or config.geometry_offset_degrees != receipt.config.geometry_offset_degrees
        or config.geometry_precision != receipt.config.geometry_precision
    ):
        raise CaliforniaError(
            "analysis years/magnitude differ from captured source selection"
        )
    repairs: list[Repair] = []
    counties, projected_counties = [], []
    projector = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True)
    inverse = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)

    def normalized(raw: Geometry, identifier: str) -> BaseGeometry:
        result = normalize_geometry(raw, identifier, repair=config.repair_geometry)
        if result.repair is not None:
            repairs.append(result.repair)
        return result.geometry

    def projected_geometry(geometry: BaseGeometry) -> BaseGeometry:
        return transform(
            projector.transform,
            segmentize(geometry, config.projection_segment_degrees),
        )

    county_ids = []
    for path in sorted(sources.glob("counties_[0-9]*.geojson")):
        for feature in Collection.model_validate_json(strict_bytes(path)).features:
            props = CountyProperties.model_validate(feature.properties)
            county_ids.append(props.OBJECTID)
            geo = normalized(feature.geometry, f"county:{props.OBJECTID}")
            area = projected_geometry(geo)
            projected_counties.append(area)
            counties.append(
                County(
                    fips="06" + props.FIPS,
                    name=props.County,
                    geometry=geometry_record(geo),
                    area_km2=area.area / 1e6,
                )
            )
    if len(counties) != 58 or len({county.fips for county in counties}) != 58:
        raise CaliforniaError("require all 58 distinct California counties")
    if len(set(county_ids)) != len(county_ids) or set(county_ids) != set(
        receipt.county_ids
    ):
        raise CaliforniaError(
            "county geometry IDs differ from the selected source roster"
        )
    projected_region = unary_union(projected_counties)
    region = transform(inverse.transform, projected_region)
    selected = sorted(
        set(
            h3.polygon_to_cells_experimental(
                h3.geo_to_h3shape(mapping(region)), config.resolution, contain="overlap"
            )
        )
    )
    if not 1 <= len(selected) <= config.max_cells:
        raise CaliforniaError("H3 carrier exceeds configured cell budget")
    cells, patches = [], []
    for identifier in selected:
        full = Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(identifier)])
        projected = projected_geometry(full).intersection(projected_region)
        if projected.is_empty or projected.area <= 0:
            continue
        clipped = transform(inverse.transform, projected)
        shares, coverage_ratio = county_weights(projected, counties, projected_counties)
        lat, lng = h3.cell_to_latlng(identifier)
        cells.append(
            Cell(
                id=identifier,
                parent=h3.cell_to_parent(identifier, config.parent_resolution),
                latitude=lat,
                longitude=lng,
                area_km2=projected.area / 1e6,
                geometry=geometry_record(clipped),
                county_shares=shares,
                raw_county_coverage_ratio=coverage_ratio,
            )
        )
        patches.append(projected)
    ids = {cell.id: index for index, cell in enumerate(cells)}
    edges = tuple(
        sorted(
            (index, ids[neighbor])
            for index, cell in enumerate(cells)
            for neighbor in h3.grid_disk(cell.id, 1)
            if neighbor in ids and index < ids[neighbor]
        )
    )
    covered = unary_union(patches)
    region_area = projected_region.area
    uncovered = projected_region.difference(covered).area / region_area
    if uncovered > 1e-7:
        raise CaliforniaError(
            "selected H3 patches do not cover the captured state geometry"
        )
    years = tuple(range(config.first_year, config.target_year + 1))
    fire_years: dict[int, list[BaseGeometry]] = {year: [] for year in years}
    fire_ids = []
    for path in sorted(sources.glob("fires_[0-9]*.geojson")):
        for feature in Collection.model_validate_json(strict_bytes(path)).features:
            fire_props = FireProperties.model_validate(feature.properties)
            fire_ids.append(fire_props.OBJECTID)
            if fire_props.YEAR_ not in fire_years:
                raise CaliforniaError("fire record outside selected years")
            geometry = projected_geometry(
                normalized(feature.geometry, f"fire:{fire_props.OBJECTID}")
            ).intersection(projected_region)
            if not geometry.is_empty:
                fire_years[fire_props.YEAR_].append(geometry)
    if len(set(fire_ids)) != len(fire_ids) or set(fire_ids) != set(receipt.fire_ids):
        raise CaliforniaError(
            "fire geometry IDs differ from the selected source roster"
        )
    tree = STRtree(patches)

    def fractions(polygons: list[BaseGeometry]) -> tuple[float, ...]:
        union = unary_union(polygons)
        values = np.zeros(len(cells))
        for index in tree.query(union, predicate="intersects"):
            values[index] = (
                patches[index].intersection(union).area / patches[index].area
            )
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1 + 1e-8)
        ):
            raise CaliforniaError("invalid clipped area fractions")
        return tuple(float(min(1, value)) for value in values)

    fire_values, quake_values, drought_values = [], [], []
    quake_records, drought_records = [], []
    for year in years:
        fire_values.append(fractions(fire_years[year]))
        quake = np.zeros(len(cells))
        earthquakes = Collection.model_validate_json(
            strict_bytes(sources / f"earthquakes_{year}.geojson")
        )
        event_ids = [feature.id for feature in earthquakes.features]
        if any(
            not isinstance(identifier, str) or not identifier
            for identifier in event_ids
        ) or len(set(event_ids)) != len(event_ids):
            raise CaliforniaError("earthquake IDs must be nonempty and distinct")
        for feature in earthquakes.features:
            earthquake_props = EarthquakeProperties.model_validate(feature.properties)
            if (
                earthquake_props.mag < config.earthquake_magnitude
                or datetime.fromtimestamp(earthquake_props.time / 1000, UTC).year
                != year
            ):
                raise CaliforniaError(
                    "earthquake record outside its temporal/magnitude selection"
                )
            point = normalized(feature.geometry, f"earthquake:{feature.id}")
            if point.geom_type != "Point":
                raise CaliforniaError("earthquake geometry must be a point")
            metric_point = transform(projector.transform, point)
            if projected_region.covers(metric_point):
                index = point_patch_index(metric_point, tree)
                quake[index] += 1
        quake_values.append(tuple(map(float, quake)))
        quake_records.append(int(quake.sum()))
        drought = drought_collection(sources / f"drought_{last_tuesday(year)}.zip")
        polygons = []
        for feature in drought.features:
            drought_props = DroughtProperties.model_validate(feature.properties)
            if drought_props.DM >= config.drought_category:
                geometry = projected_geometry(
                    normalized(feature.geometry, f"drought:{year}:{feature.id}")
                ).intersection(projected_region)
                if not geometry.is_empty:
                    polygons.append(geometry)
        drought_values.append(fractions(polygons))
        drought_records.append(len(polygons))
    indicators = []
    for hazard, values, threshold, unit, dates, records in (
        (
            Hazard.FIRE,
            fire_values,
            config.fire_fraction,
            "union perimeter area / clipped cell area",
            tuple(f"{year}-01-01/{year}-12-31" for year in years),
            tuple(len(fire_years[year]) for year in years),
        ),
        (
            Hazard.SEISMIC,
            quake_values,
            1,
            "catalogued earthquakes at or above magnitude cutoff / cell / year",
            tuple(f"{year}-01-01/{year}-12-31" for year in years),
            tuple(quake_records),
        ),
        (
            Hazard.DROUGHT,
            drought_values,
            config.drought_fraction,
            "D-category-or-worse area / clipped cell area at year-end snapshot",
            tuple(last_tuesday(year) for year in years),
            tuple(drought_records),
        ),
    ):
        positive = tuple(
            tuple(int(value >= threshold) for value in row) for row in values
        )
        training = np.asarray(positive[:-1], dtype=float)
        statewide = float(training.mean())
        prior = (training.sum(axis=0) + config.prior_strength * statewide + 0.5) / (
            len(training) + config.prior_strength + 1
        )
        indicators.append(
            Indicator(
                hazard=hazard,
                unit=unit,
                years=years,
                dates=dates,
                values=tuple(values),
                positive=positive,
                prior=tuple(map(float, prior)),
                source_records=records,
            )
        )
    landscape = Landscape(
        schema_version="geo-infer-place/california-landscape/1",
        config=config,
        source_receipt_sha256=digest(sources / "sources_receipt.json"),
        projection="EPSG:3310",
        counties=tuple(counties),
        cells=tuple(cells),
        edges=edges,
        indicators=tuple(indicators),
        repairs=tuple(repairs),
        administrative_area_km2=region_area / 1e6,
        uncovered_fraction=uncovered,
        county_overlap_km2=max(
            0,
            (
                sum(county.area for county in projected_counties)
                - unary_union(projected_counties).area
            )
            / 1e6,
        ),
        county_weighting="normalized-intersection-area",
    )
    output.mkdir(parents=True, exist_ok=False)
    path = output / "landscape.json"
    write_record(path, landscape)
    return path
