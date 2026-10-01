# Urban Analytics: Multi-Layered City Analysis

> **Illustrative example notice.** This page contains historical or
> conceptual integration sketches. Names such as `SpatialAnalyzer` and
> domain-specific facade classes are not public GEO-INFER exports in the
> current checkout; verify imports against each module's `src/` package
> and use the module README/tests for executable examples.

> **Illustrative guide.** The code in this page is illustrative: it sketches
> how the module APIs compose for this use case. Some identifiers shown are
> conceptual; always import from the current package exports (see the module
> `__init__.py` and `SKILL.md`) and prefer the runnable scripts under
> `GEO-INFER-*/examples/` for verified behavior. Any numeric results shown
> are illustrative and must be reproduced against your own data before use.


This guide demonstrates multi-layered urban analysis using GEO-INFER modules for population density mapping, infrastructure exposure assessment, multi-hazard risk scoring, and civic engagement analysis.

## Overview

Urban analytics in GEO-INFER combines four modules:

- **GEO-INFER-SPACE** -- H3 spatial indexing for uniform analysis grids
- **GEO-INFER-RISK** -- hazard, exposure, and risk modeling
- **GEO-INFER-CIV** -- civic participation and equity analysis
- **GEO-INFER-DATA** -- data ingestion and processing

The output is a composite risk-equity map that identifies areas with high hazard exposure and low civic engagement -- the zones most in need of intervention.

## Prerequisites

```bash
# From the repository root: installs SPACE, RISK, CIV, DATA and the locked
# numpy, pandas, geopandas, matplotlib, shapely and h3 versions
uv sync --all-packages --all-extras --all-groups
```

## Section 1: Population Density H3 Maps

Population density is the foundation of urban risk analysis. This section creates multi-resolution H3 grids and distributes census population counts to hexagonal cells using dasymetric mapping.

### Creating the Analysis Grid

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, Point


def create_city_h3_grid(
    center_lat: float,
    center_lng: float,
    ring_size: int = 30,
    resolution: int = 8
) -> gpd.GeoDataFrame:
    """Create an H3 hex grid covering a city area.

    Resolution 8 gives cells approximately 460m edge-to-edge,
    suitable for neighborhood-level analysis.

    Args:
        center_lat: City center latitude.
        center_lng: City center longitude.
        ring_size: Number of hex rings from center (controls coverage).
        resolution: H3 resolution.

    Returns:
        GeoDataFrame with H3 cells and geometry.
    """
    center_cell = h3.latlng_to_cell(center_lat, center_lng, resolution)
    cells = list(h3.grid_disk(center_cell, ring_size))

    rows = []
    for cell in cells:
        boundary = h3.cell_to_boundary(cell)
        ring = [(lng, lat) for lat, lng in boundary]
        ring.append(ring[0])
        lat, lng = h3.cell_to_latlng(cell)
        rows.append({
            "h3_index": cell,
            "geometry": Polygon(ring),
            "center_lat": lat,
            "center_lng": lng,
        })

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# Portland, Oregon as the example city
city_grid = create_city_h3_grid(
    center_lat=45.5152, center_lng=-122.6784,
    ring_size=30, resolution=8
)
print(f"City grid: {len(city_grid)} H3 cells at resolution 8")
area_m2 = city_grid.to_crs(epsg=32610).area.mean()
print(f"Mean cell area: {area_m2:.0f} m^2 ({area_m2 / 10000:.2f} ha)")
```

### Dasymetric Population Mapping

Dasymetric mapping distributes coarse census population counts to fine H3 cells using land use as an ancillary variable. Built-up areas receive more population; parks and water receive less.

```python
def generate_census_tracts(
    city_grid: gpd.GeoDataFrame,
    n_tracts: int = 25,
    seed: int = 42
) -> gpd.GeoDataFrame:
    """Generate synthetic census tracts with population counts.

    Creates Voronoi-like tract boundaries and assigns population
    based on distance from city center (higher in center).

    Args:
        city_grid: H3 city grid GeoDataFrame.
        n_tracts: Number of census tracts.
        seed: Random seed.

    Returns:
        GeoDataFrame with tract polygons and 'population' column.
    """
    rng = np.random.default_rng(seed)

    # Sample tract centroids from the grid
    sample_idx = rng.choice(len(city_grid), size=n_tracts, replace=False)
    centroids = city_grid.iloc[sample_idx][["center_lat", "center_lng"]].values

    # Assign population: higher near center
    city_center = np.array([45.5152, -122.6784])
    distances = np.sqrt(np.sum((centroids - city_center) ** 2, axis=1))
    max_dist = distances.max()
    pop_factor = 1.0 - 0.7 * (distances / max_dist)
    population = (5000 * pop_factor + rng.uniform(500, 2000, n_tracts)).astype(int)

    # Create tract polygons (simplified as buffered points)
    geometries = [
        Point(lng, lat).buffer(0.01)
        for lat, lng in centroids
    ]

    return gpd.GeoDataFrame(
        {
            "tract_id": [f"tract_{i:03d}" for i in range(n_tracts)],
            "population": population,
        },
        geometry=geometries,
        crs="EPSG:4326",
    )


def dasymetric_population_mapping(
    city_grid: gpd.GeoDataFrame,
    tracts: gpd.GeoDataFrame,
    seed: int = 42
) -> gpd.GeoDataFrame:
    """Distribute tract-level population to H3 cells.

    Uses a land-use weight: each H3 cell gets a synthetic built-up
    fraction, and population is distributed proportionally within
    each tract.

    Args:
        city_grid: H3 grid GeoDataFrame.
        tracts: Census tract GeoDataFrame with 'population' column.
        seed: Random seed.

    Returns:
        Grid GeoDataFrame with added 'population' and 'pop_density_km2'.
    """
    rng = np.random.default_rng(seed)

    # Assign synthetic built-up fraction (proxy for development intensity)
    center = np.array([-122.6784, 45.5152])  # (lng, lat)
    centroids = city_grid.geometry.centroid
    dist_from_center = np.sqrt(
        (centroids.x.values - center[0]) ** 2 +
        (centroids.y.values - center[1]) ** 2
    )
    max_dist = dist_from_center.max()
    built_up = np.clip(
        0.8 - 0.6 * (dist_from_center / max_dist) + rng.normal(0, 0.1, len(city_grid)),
        0.05, 1.0
    )
    city_grid = city_grid.copy()
    city_grid["built_up_fraction"] = built_up

    # Spatial join: assign each H3 cell to its containing tract
    cell_centroids = gpd.GeoDataFrame(
        city_grid[["h3_index", "built_up_fraction"]],
        geometry=centroids,
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(cell_centroids, tracts, how="left", predicate="within")

    # Distribute population proportionally by built-up fraction
    population = np.zeros(len(city_grid))
    for tract_id in tracts["tract_id"].unique():
        mask = joined["tract_id"] == tract_id
        if mask.sum() == 0:
            continue
        tract_pop = tracts.loc[tracts["tract_id"] == tract_id, "population"].iloc[0]
        weights = joined.loc[mask, "built_up_fraction"].values
        total_weight = weights.sum()
        if total_weight > 0:
            cell_indices = joined.index[mask]
            population[cell_indices] = tract_pop * weights / total_weight

    city_grid["population"] = population
    # Convert to density per km^2
    area_km2 = city_grid.to_crs(epsg=32610).area.values / 1e6
    city_grid["pop_density_km2"] = city_grid["population"] / np.maximum(area_km2, 1e-6)

    return city_grid


tracts = generate_census_tracts(city_grid)
pop_grid = dasymetric_population_mapping(city_grid, tracts)
print(f"Total population mapped: {pop_grid['population'].sum():.0f}")
print(f"Mean density: {pop_grid['pop_density_km2'].mean():.0f} people/km^2")
print(f"Max density: {pop_grid['pop_density_km2'].max():.0f} people/km^2")
```

### Choropleth Visualization

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
pop_grid.plot(
    column="pop_density_km2",
    ax=ax,
    legend=True,
    cmap="YlOrRd",
    scheme="quantiles",
    k=7,
    legend_kwds={"title": "Pop density (per km^2)", "loc": "lower left"},
)
ax.set_title("Population Density: H3 Resolution 8")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig("population_density_h3.png", dpi=150)
```

## Section 2: Infrastructure Exposure Assessment

Exposure quantifies how much is at risk. This section loads infrastructure layers and computes H3-indexed exposure scores.

### Generating Infrastructure Data

```python
def generate_infrastructure_layers(
    city_grid: gpd.GeoDataFrame,
    seed: int = 42
) -> dict[str, gpd.GeoDataFrame]:
    """Generate synthetic infrastructure data for exposure modeling.

    Creates three layers: buildings, roads, utilities.
    Building density and road density correlate with population.

    Args:
        city_grid: H3 grid with population data.
        seed: Random seed.

    Returns:
        Dict with 'buildings', 'roads', 'utilities' GeoDataFrames.
    """
    rng = np.random.default_rng(seed)

    # Buildings: count per H3 cell proportional to population
    pop_vals = city_grid["population"].values
    max_pop = max(pop_vals.max(), 1.0)
    building_count = (50 * pop_vals / max_pop + rng.uniform(0, 5, len(city_grid))).astype(int)

    # Building value (millions USD): correlated with density
    avg_value = 0.3 + 0.7 * (pop_vals / max_pop)  # 0.3M to 1.0M per building
    total_value = building_count * avg_value

    buildings = city_grid.copy()
    buildings["building_count"] = building_count
    buildings["total_value_musd"] = total_value

    # Road segments: density proportional to built-up area
    road_km = 2.0 * city_grid["built_up_fraction"].values + rng.uniform(0, 0.5, len(city_grid))
    roads = city_grid.copy()
    roads["road_km"] = road_km

    # Utilities: binary presence of critical infrastructure
    utility_probability = 0.1 + 0.3 * (pop_vals / max_pop)
    has_utility = rng.random(len(city_grid)) < utility_probability
    utilities = city_grid.copy()
    utilities["has_critical_infra"] = has_utility
    utilities["utility_type"] = np.where(has_utility, "substation", "none")

    return {
        "buildings": buildings,
        "roads": roads,
        "utilities": utilities,
    }


infra = generate_infrastructure_layers(pop_grid)
print(f"Total buildings: {infra['buildings']['building_count'].sum()}")
print(f"Total road km: {infra['roads']['road_km'].sum():.0f}")
print(f"Critical infrastructure nodes: {infra['utilities']['has_critical_infra'].sum()}")
```

### Computing Exposure Scores

```python
from geo_infer_risk.core.exposure_model import ExposureModel


def compute_exposure_index(
    pop_grid: gpd.GeoDataFrame,
    infra: dict[str, gpd.GeoDataFrame],
    weights: dict[str, float] | None = None
) -> gpd.GeoDataFrame:
    """Compute a composite exposure index per H3 cell.

    Combines population, building value, road density, and critical
    infrastructure presence into a single 0-1 index.

    Args:
        pop_grid: Grid with population data.
        infra: Infrastructure layers dict.
        weights: Relative weights for each component.

    Returns:
        Grid GeoDataFrame with added 'exposure_index' column.
    """
    if weights is None:
        weights = {
            "population": 0.35,
            "building_value": 0.30,
            "road_density": 0.15,
            "critical_infra": 0.20,
        }

    def normalize_01(arr: np.ndarray) -> np.ndarray:
        rng = arr.max() - arr.min()
        if rng == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / rng

    pop_norm = normalize_01(pop_grid["population"].values)
    bldg_norm = normalize_01(infra["buildings"]["total_value_musd"].values)
    road_norm = normalize_01(infra["roads"]["road_km"].values)
    crit_norm = infra["utilities"]["has_critical_infra"].values.astype(float)

    exposure = (
        weights["population"] * pop_norm +
        weights["building_value"] * bldg_norm +
        weights["road_density"] * road_norm +
        weights["critical_infra"] * crit_norm
    )

    result = pop_grid.copy()
    result["exposure_index"] = exposure
    return result


exposure_grid = compute_exposure_index(pop_grid, infra)
print(f"Exposure index range: {exposure_grid['exposure_index'].min():.3f} - "
      f"{exposure_grid['exposure_index'].max():.3f}")
print(f"Mean exposure: {exposure_grid['exposure_index'].mean():.3f}")
```

## Section 3: Multi-Hazard Risk Scoring

Risk is the product of hazard, exposure, and vulnerability. This section combines flood, seismic, and heat island hazards into a composite risk index.

### Generating Hazard Layers

```python
def generate_hazard_layers(
    city_grid: gpd.GeoDataFrame,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate synthetic hazard probability maps.

    Creates three hazard layers:
    - Flood: higher near rivers (simulated as low elevation)
    - Seismic: uniform with localized fault zones
    - Heat island: higher in dense built-up areas

    Args:
        city_grid: H3 grid GeoDataFrame.
        seed: Random seed.

    Returns:
        Dict mapping hazard name to probability arrays (0-1).
    """
    rng = np.random.default_rng(seed)
    n = len(city_grid)
    centroids = city_grid.geometry.centroid

    lons = centroids.x.values
    lats = centroids.y.values

    # Flood hazard: gradient from east (river) to west
    lon_norm = (lons - lons.min()) / max(lons.max() - lons.min(), 1e-6)
    flood_base = 0.6 * lon_norm + rng.normal(0, 0.05, n)
    flood = np.clip(flood_base, 0.0, 1.0)

    # Seismic hazard: baseline + fault zone
    fault_lat = 45.52
    fault_dist = np.abs(lats - fault_lat)
    seismic_base = 0.15 + 0.5 * np.exp(-fault_dist / 0.005)
    seismic = np.clip(seismic_base + rng.normal(0, 0.03, n), 0.0, 1.0)

    # Heat island: correlated with built-up fraction
    if "built_up_fraction" in city_grid.columns:
        heat = 0.3 + 0.5 * city_grid["built_up_fraction"].values
    else:
        heat = 0.3 + rng.uniform(0, 0.5, n)
    heat = np.clip(heat + rng.normal(0, 0.05, n), 0.0, 1.0)

    return {"flood": flood, "seismic": seismic, "heat_island": heat}


hazards = generate_hazard_layers(exposure_grid)
for name, arr in hazards.items():
    print(f"{name}: mean={arr.mean():.3f}, max={arr.max():.3f}")
```

### Composite Risk Computation

```python
from geo_infer_risk.core.risk_engine import RiskEngine


def compute_composite_risk(
    exposure_grid: gpd.GeoDataFrame,
    hazards: dict[str, np.ndarray],
    hazard_weights: dict[str, float] | None = None,
    vulnerability_factor: float = 0.5
) -> gpd.GeoDataFrame:
    """Compute composite risk index from multiple hazards and exposure.

    Risk = Hazard * Exposure * Vulnerability

    The composite hazard is a weighted combination of individual hazards.
    Vulnerability is simplified as a constant factor (in practice,
    this would be derived from building type, age, income data).

    Args:
        exposure_grid: Grid with exposure_index column.
        hazards: Dict of hazard arrays.
        hazard_weights: Relative hazard weights.
        vulnerability_factor: Baseline vulnerability (0-1).

    Returns:
        Grid with hazard, exposure, vulnerability, and risk columns.
    """
    if hazard_weights is None:
        hazard_weights = {"flood": 0.45, "seismic": 0.30, "heat_island": 0.25}

    # Composite hazard
    composite_hazard = np.zeros(len(exposure_grid))
    for name, weight in hazard_weights.items():
        composite_hazard += weight * hazards[name]

    exposure = exposure_grid["exposure_index"].values

    # Risk index
    risk = composite_hazard * exposure * vulnerability_factor

    result = exposure_grid.copy()
    result["hazard_flood"] = hazards["flood"]
    result["hazard_seismic"] = hazards["seismic"]
    result["hazard_heat"] = hazards["heat_island"]
    result["composite_hazard"] = composite_hazard
    result["vulnerability"] = vulnerability_factor
    result["risk_index"] = risk

    return result


risk_grid = compute_composite_risk(exposure_grid, hazards)
print(f"Risk index range: {risk_grid['risk_index'].min():.4f} - "
      f"{risk_grid['risk_index'].max():.4f}")

# Identify high-risk zones (top 10%)
threshold = risk_grid["risk_index"].quantile(0.90)
high_risk = risk_grid[risk_grid["risk_index"] >= threshold]
print(f"High-risk cells (top 10%): {len(high_risk)} "
      f"(threshold: {threshold:.4f})")
```

### Risk Map Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

for ax, (col, title, cmap) in zip(
    axes.ravel(),
    [
        ("hazard_flood", "Flood Hazard", "Blues"),
        ("hazard_seismic", "Seismic Hazard", "Reds"),
        ("hazard_heat", "Heat Island Hazard", "YlOrRd"),
        ("risk_index", "Composite Risk Index", "RdYlGn_r"),
    ]
):
    risk_grid.plot(column=col, ax=ax, legend=True, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("multi_hazard_risk.png", dpi=150)
```

## Section 4: Civic Engagement Analysis

Civic participation data reveals which communities are organized and can advocate for resources. Overlaying this with risk zones identifies underserved areas.

### Generating Civic Participation Data

```python
def generate_civic_data(
    city_grid: gpd.GeoDataFrame,
    seed: int = 42
) -> gpd.GeoDataFrame:
    """Generate synthetic civic participation data.

    Models three civic indicators:
    - voter_turnout: fraction of eligible voters participating
    - meeting_attendance: community meeting participation rate
    - complaint_rate: 311/service request rate per capita

    Lower-income areas (approximated by lower building values)
    tend to have lower turnout but higher complaint rates.

    Args:
        city_grid: H3 grid GeoDataFrame.
        seed: Random seed.

    Returns:
        Grid with civic indicator columns.
    """
    rng = np.random.default_rng(seed)
    n = len(city_grid)

    # Socioeconomic proxy: inverse of population density (suburban = higher income)
    if "pop_density_km2" in city_grid.columns:
        density = city_grid["pop_density_km2"].values
        max_density = max(density.max(), 1.0)
        ses_proxy = 1.0 - 0.5 * (density / max_density)
    else:
        ses_proxy = rng.uniform(0.3, 0.9, n)

    voter_turnout = np.clip(
        0.35 + 0.35 * ses_proxy + rng.normal(0, 0.05, n),
        0.1, 0.95
    )

    meeting_attendance = np.clip(
        0.02 + 0.08 * ses_proxy + rng.normal(0, 0.01, n),
        0.0, 0.30
    )

    # Complaint rate inversely correlated with engagement
    complaint_rate = np.clip(
        15.0 - 8.0 * voter_turnout + rng.normal(0, 2.0, n),
        0.0, 30.0
    )

    result = city_grid.copy()
    result["voter_turnout"] = voter_turnout
    result["meeting_attendance"] = meeting_attendance
    result["complaint_rate_per_1k"] = complaint_rate

    # Composite civic engagement index
    turnout_norm = (voter_turnout - voter_turnout.min()) / max(
        voter_turnout.max() - voter_turnout.min(), 1e-6
    )
    meeting_norm = (meeting_attendance - meeting_attendance.min()) / max(
        meeting_attendance.max() - meeting_attendance.min(), 1e-6
    )
    result["civic_engagement_index"] = 0.6 * turnout_norm + 0.4 * meeting_norm

    return result


civic_grid = generate_civic_data(risk_grid)
print(f"Mean voter turnout: {civic_grid['voter_turnout'].mean():.1%}")
print(f"Mean civic engagement index: {civic_grid['civic_engagement_index'].mean():.3f}")
```

### Identifying Underserved High-Risk Areas

```python
def identify_underserved_zones(
    risk_civic_grid: gpd.GeoDataFrame,
    risk_threshold_quantile: float = 0.75,
    engagement_threshold_quantile: float = 0.25
) -> gpd.GeoDataFrame:
    """Identify areas with high risk but low civic engagement.

    These are the neighborhoods most in need of proactive outreach
    and resource allocation.

    Args:
        risk_civic_grid: Grid with risk_index and civic_engagement_index.
        risk_threshold_quantile: Risk quantile threshold (above = high risk).
        engagement_threshold_quantile: Engagement quantile (below = low engagement).

    Returns:
        Filtered GeoDataFrame of underserved high-risk cells.
    """
    risk_threshold = risk_civic_grid["risk_index"].quantile(risk_threshold_quantile)
    engagement_threshold = risk_civic_grid["civic_engagement_index"].quantile(
        engagement_threshold_quantile
    )

    mask = (
        (risk_civic_grid["risk_index"] >= risk_threshold) &
        (risk_civic_grid["civic_engagement_index"] <= engagement_threshold)
    )

    underserved = risk_civic_grid[mask].copy()
    underserved["priority_score"] = (
        risk_civic_grid.loc[mask, "risk_index"] *
        (1.0 - risk_civic_grid.loc[mask, "civic_engagement_index"])
    )

    return underserved.sort_values("priority_score", ascending=False)


underserved = identify_underserved_zones(civic_grid)
print(f"Underserved high-risk zones: {len(underserved)} cells")
print(f"Population in these zones: {underserved['population'].sum():.0f}")
```

### Civic-Risk Overlay Visualization

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Civic engagement
civic_grid.plot(
    column="civic_engagement_index",
    ax=axes[0], legend=True, cmap="Greens",
    legend_kwds={"label": "Civic engagement"},
)
axes[0].set_title("Civic Engagement Index")

# Risk
civic_grid.plot(
    column="risk_index",
    ax=axes[1], legend=True, cmap="Reds",
    legend_kwds={"label": "Risk index"},
)
axes[1].set_title("Composite Risk Index")

# Underserved zones highlighted
civic_grid.plot(ax=axes[2], facecolor="lightgray", edgecolor="none")
if len(underserved) > 0:
    underserved.plot(
        column="priority_score",
        ax=axes[2], legend=True, cmap="YlOrRd",
        legend_kwds={"label": "Priority score"},
    )
axes[2].set_title("Underserved High-Risk Zones")

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("civic_risk_overlay.png", dpi=150)
```

## Full Pipeline Integration

```python
from typing import Any


def run_urban_analytics_pipeline(
    center_lat: float,
    center_lng: float,
    h3_resolution: int = 8,
    ring_size: int = 30
) -> dict[str, Any]:
    """Execute the complete urban analytics pipeline.

    Steps:
        1. Create H3 grid
        2. Map population density
        3. Assess infrastructure exposure
        4. Compute multi-hazard risk
        5. Analyze civic engagement
        6. Identify underserved zones

    Args:
        center_lat: City center latitude.
        center_lng: City center longitude.
        h3_resolution: H3 grid resolution.
        ring_size: Grid extent in hex rings.

    Returns:
        Dict with intermediate and final GeoDataFrames.
    """
    grid = create_city_h3_grid(center_lat, center_lng, ring_size, h3_resolution)
    tracts = generate_census_tracts(grid)
    pop = dasymetric_population_mapping(grid, tracts)
    infra = generate_infrastructure_layers(pop)
    exposure = compute_exposure_index(pop, infra)
    hazards = generate_hazard_layers(exposure)
    risk = compute_composite_risk(exposure, hazards)
    civic = generate_civic_data(risk)
    underserved = identify_underserved_zones(civic)

    return {
        "grid": grid,
        "population": pop,
        "exposure": exposure,
        "risk": risk,
        "civic": civic,
        "underserved": underserved,
        "summary": {
            "total_cells": len(grid),
            "total_population": float(pop["population"].sum()),
            "mean_risk": float(risk["risk_index"].mean()),
            "underserved_cells": len(underserved),
            "underserved_population": float(underserved["population"].sum()),
        },
    }


results = run_urban_analytics_pipeline(
    center_lat=45.5152, center_lng=-122.6784
)

print("Urban Analytics Summary:")
for key, value in results["summary"].items():
    if isinstance(value, float):
        print(f"  {key}: {value:,.1f}")
    else:
        print(f"  {key}: {value}")
```

## Outputs Summary

| Output | Description | Module(s) |
|--------|-------------|-----------|
| `population_density_h3.png` | Choropleth of population density per H3 cell | SPACE, DATA |
| `multi_hazard_risk.png` | Four-panel hazard and composite risk maps | RISK |
| `civic_risk_overlay.png` | Side-by-side civic engagement, risk, and underserved zones | CIV, RISK |
| `results["summary"]` | Pipeline summary statistics | All modules |

## Next Steps

- **Real data integration**: Replace synthetic generators with OpenStreetMap, census API, and FEMA flood map loaders via GEO-INFER-DATA
- **Climate hazards**: Connect climate projections to heat island hazard modeling (see [Climate Modeling](climate_modeling.md))
- **Planning recommendations**: Use the underserved zones as input to urban planning workflows (see [Urban Planning](urban_planning.md))
- **Active Inference**: Model community response dynamics using GEO-INFER-ACT perception-action loops
- **Temporal analysis**: Track risk and engagement changes over time with GEO-INFER-TIME
