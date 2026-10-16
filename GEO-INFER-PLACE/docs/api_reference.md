# GEO-INFER-PLACE API Reference

Complete API reference for the `geo_infer_place` package.

## PlaceInterface

**Module**: `geo_infer_place.core.place_interface`

Unified entry point for place-based analysis. Orchestrates location-specific analyzers, data acquisition, temporal analysis, and quality management.

### Constructor

```python
PlaceInterface(
    location: str = "del_norte",
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    counties: Optional[List[str]] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `location` | `str` | `"del_norte"` | Location key (`"del_norte"` or `"cascadia"`) |
| `config` | `Optional[Dict]` | `None` | Override configuration; replaces preset defaults |
| `output_dir` | `Optional[str]` | `None` | Output directory path; defaults to `locations/<location>/output/` |
| `counties` | `Optional[List[str]]` | `None` | County filter for Cascadia (format: `"STATE:County"`) |

**Raises**: `ValueError` if location key is not in `LOCATION_PRESETS`.

**Attributes**:
- `location` (`str`): Location key.
- `location_name` (`str`): Human-readable location name.
- `config` (`Dict`): Active configuration dictionary.
- `output_dir` (`Path`): Output directory path.
- `counties` (`Optional[List[str]]`): County filter list.

### Properties (Lazy-Initialized)

#### `integrator`

Data integrator with wrapped API clients (CAL FIRE, NOAA, USGS). Initialized on first access.

#### `data_manager`

`PlaceDataManager` instance for data quality validation and provenance tracking. Bridges GEO-INFER-DATA when available.

#### `temporal`

`PlaceTemporalAnalyzer` instance for trend detection and anomaly analysis. Bridges GEO-INFER-TIME when available.

### Methods

#### `get_analyzer(name: str) -> Any`

Get or create a named analyzer. Supported analyzers:

| Name | Class | Description |
|------|-------|-------------|
| `"forest_health"` | `ForestHealthMonitor` | Forest canopy, NDVI, biomass health |
| `"coastal_resilience"` | `CoastalResilienceAnalyzer` | Coastal vulnerability and erosion |
| `"fire_risk"` | `FireRiskAssessor` | Wildfire risk scoring |
| `"seismic_hazard"` | `SeismicHazardAnalyzer` | Earthquake and fault exposure |

**Returns**: Analyzer instance, or `None` if not implemented for the location.

#### `run_full_analysis(analyzers: Optional[List[str]] = None, include_temporal: bool = True) -> Dict[str, Any]`

Run all configured analyzers and return unified results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyzers` | `Optional[List[str]]` | `None` | Analyzer names to run; defaults to all for location |
| `include_temporal` | `bool` | `True` | Run temporal analysis on applicable data |

**Returns**: Dictionary with keys:
- `location` (`str`): Location name.
- `timestamp` (`str`): ISO 8601 timestamp.
- `config` (`Dict`): Analysis configuration.
- `analyses` (`Dict[str, Any]`): Per-analyzer results.
- `temporal_analysis` (`Dict[str, Any]`): Trend and anomaly results.
- `data_quality` (`Dict[str, Any]`): Per-analyzer quality reports.
- `provenance` (`List`): Data provenance log.

Results are saved to `<output_dir>/<location>_full_analysis_<timestamp>.json`.

#### `get_earthquakes(bbox: Optional[tuple] = None) -> Dict[str, Any]`

Fetch recent earthquakes for the location from USGS.

#### `get_cascadia_seismicity(days: int = 30) -> Dict[str, Any]`

Fetch Cascadia-wide seismicity data from USGS.

#### `get_tide_data(stations: Optional[List[str]] = None, time_range: Optional[tuple] = None) -> Dict[str, Any]`

Fetch tide gauge data from NOAA.

#### `get_fire_perimeters(bbox: Optional[tuple] = None, start_year: Optional[int] = None) -> Dict[str, Any]`

Fetch fire perimeter data from CAL FIRE.

#### `get_weather(station_id: str = "KCEC") -> Dict[str, Any]`

Fetch current weather observations from NOAA.

#### `status() -> Dict[str, Any]`

Return status of all components including module availability, initialized analyzers, and cache statistics.

---

## PlaceDataManager

**Module**: `geo_infer_place.core.module_bridge`

Data quality validation and provenance tracking. Bridges GEO-INFER-DATA module when available; provides standalone fallback otherwise.

**Attributes**:
- `has_data_module` (`bool`): Whether GEO-INFER-DATA is installed.

### Methods

#### `validate_dataset(data: Any, name: str = "") -> Dict`

Validate dataset quality. Returns completeness and anomaly metrics.

#### `log_provenance(name: str, metadata: Dict) -> None`

Record provenance metadata for a data processing step.

#### `get_provenance() -> List`

Return accumulated provenance log.

---

## PlaceTemporalAnalyzer

**Module**: `geo_infer_place.core.module_bridge`

Temporal trend detection and anomaly analysis. Bridges GEO-INFER-TIME module when available.

**Attributes**:
- `has_time_module` (`bool`): Whether GEO-INFER-TIME is installed.

### Methods

#### `analyze_tide_trends(tide_data: Dict) -> Dict`

Analyze tide gauge data for sea-level trends.

#### `analyze_seismic_rates(seismic_data: Dict) -> Dict`

Analyze seismic event rates for temporal patterns.

---

## InteractiveVisualizationEngine

**Module**: `geo_infer_place.core.visualization_engine`

H3 hexagonal map generation and interactive dashboard creation.

---

## API Clients

**Module**: `geo_infer_place.core.api_clients`

### CaliforniaAPIManager

Manages connections to all California-specific data APIs.

### NOAAClient

Client for NOAA weather and tide data. Includes retry logic and response caching.

### USGSClient / USGSEarthquakeClient

Client for USGS earthquake catalog and geological data.

### CALFIREClient

Client for CAL FIRE wildfire perimeter and incident data.

### CDECClient

Client for California Data Exchange Center hydrological data.

---

## H3 Utility Functions

**Module**: `geo_infer_place.utils.h3_operations`

Re-exported H3 v4 functions for convenience:

| Function | Description |
|----------|-------------|
| `latlng_to_cell(lat, lng, resolution)` | Convert point to H3 cell |
| `cell_to_latlng(cell)` | Get cell centroid |
| `cell_to_latlng_boundary(cell)` | Get cell boundary polygon |
| `geo_to_cells(geojson, resolution)` | Fill GeoJSON with cells |
| `polygon_to_cells(polygon, resolution)` | Fill polygon with cells |
| `grid_disk(cell, k)` | K-ring neighborhood |
| `grid_distance(cell1, cell2)` | Grid distance between cells |
| `grid_ring(cell, k)` | Hollow ring at distance k |
| `cell_area(cell, unit)` | Cell area in km2 or m2 |
| `get_resolution(cell)` | Get cell resolution |
| `is_valid_cell(cell)` | Validate cell string |
| `are_neighbor_cells(cell1, cell2)` | Check adjacency |
| `cells_to_geodataframe(cells)` | Convert to GeoDataFrame |
| `cell_to_parent(cell, resolution)` | Get parent cell |
| `cell_to_children(cell, resolution)` | Get child cells |
| `compact_cells(cells)` | Compact cell set |
| `uncompact_cells(cells, resolution)` | Uncompact to resolution |
| `estimate_cell_count(polygon, resolution)` | Estimate fill count |

---

## Configuration

### Location Presets

Loaded from `config/location_presets.yaml` or hardcoded fallback:

```yaml
del_norte:
  name: "Del Norte County, California"
  bounds:
    west: -124.408
    south: 41.458
    east: -123.536
    north: 42.006
  h3_resolution: 8
  analyzers:
    - forest_health
    - coastal_resilience
    - fire_risk
    - seismic_hazard
  data_sources:
    - calfire
    - noaa
    - usgs

cascadia:
  name: "Cascadia Bioregion (BC, WA, OR, CA)"
  bounds:
    west: -124.8
    south: 40.0
    east: -114.5
    north: 49.0
  h3_resolution: 7
  analyzers:
    - seismic_hazard
    - forest_health
    - salmon_habitat
    - volcanic_hazard
  data_sources:
    - usgs
    - noaa
    - calfire
```

## Hydrography acquisition and network analysis

**Module**: `geo_infer_place.hydrography`

See the [hydrography guide](../src/geo_infer_place/hydrography/GUIDE.md) for
bounded USGS ingestion, resumable page checksums, native NHDPlusID aliases,
full-network topology before filtering, and installed sample access.

```python
from geo_infer_place.hydrography import (
    HydrographySelection, NHDPlusHRIngestor, CascadianSurfaceWaterDataSources,
    SMITH_RIVER_HUC8, SMITH_RIVER_PILOT_BBOX,
)

path = NHDPlusHRIngestor().ingest(
    HydrographySelection(SMITH_RIVER_PILOT_BBOX, SMITH_RIVER_HUC8),
    "smith-river-data",
)
network = CascadianSurfaceWaterDataSources(path).get_flowline_network(4)
```

The pilot is a bounded lower Smith River excerpt. Whole-watershed and regional
acquisition require explicit selections and resource budgets. Unknown native
orders remain unknown; network H3 indexing reports its sampling approximation.

## Bioregion layer rendering

**Module**: `geo_infer_place.core.bioregion_visualization`

```python
create_bioregion_map(
    config_dir, h3_data, output_path,
    integration_results=None, *, allow_missing_layers=False,
)
```

Provide an explicit directory containing WGS84 GeoJSON:
`cascadia_bioregion_boundary.geojson` and `cascadia_major_watersheds.geojson`
(Polygon/MultiPolygon), `cascadia_subduction_zone.geojson`
(LineString/MultiLineString), and `cascadia_volcanoes.geojson` (Point).
Features preserve their supplied names and properties. Supplied H3 IDs render
with actual cell boundaries. Invalid CRS, coordinates and geometry are rejected;
source text is escaped before HTML interpolation. No earthquake probability or
volcano threat level is inferred from geometry alone.

Missing required files raise `FileNotFoundError`. Explicit
`allow_missing_layers=True` produces a partial map with a visible unavailable
layer notice. A sibling `.layers.json` records loaded/unavailable layer status.
The packaged snapshot includes Aquila Flower's complete watershed-based
bioregion boundary and three narrower USGS display layers. Captured boundary
metadata, geometry, attribution and receipt hashes are verified before rendering;
the map fits the full source extent and displays its attribution/license. See
[boundary capture and scope](cascadia_boundary.md) and
[USGS display-layer scope](usgs_regional_layers.md). Constructed geometry fixtures
remain separate behavior controls. Legacy location imports remain compatibility shims.
