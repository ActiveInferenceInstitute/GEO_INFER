# Urban Planning: Data-Driven Planning with Active Inference

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


This guide demonstrates data-driven urban planning workflows using GEO-INFER modules for zoning analysis, transport network optimization, green space scoring, participatory planning, and multi-criteria site selection.

## Overview

Urban planning decisions involve competing objectives -- density vs. livability, economic development vs. environmental preservation, transit access vs. cost. Active Inference provides a principled framework for combining technical analysis with community preferences, treating the planning process as iterative belief updating under uncertainty.

This guide covers five interconnected analyses that feed into a final site selection decision.

## Prerequisites

```bash
# From the repository root: installs CIV, TRANSPORT, SPACE and the locked
# numpy, pandas, geopandas, matplotlib, shapely, h3 and networkx versions
uv sync --all-packages --all-extras
```

## Section 1: Zoning Analysis

Zoning determines permitted land uses and development intensity. This section loads zoning layers and computes mixed-use indices and density metrics at H3 resolution 9.

### Setting Up the Planning Grid

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, Point, LineString
import networkx as nx


def create_planning_grid(
    center_lat: float,
    center_lng: float,
    ring_size: int = 20,
    resolution: int = 9
) -> gpd.GeoDataFrame:
    """Create an H3 planning grid for a study area.

    Resolution 9 gives cells approximately 175m edge-to-edge,
    suitable for block-level planning analysis.

    Args:
        center_lat: Study area center latitude.
        center_lng: Study area center longitude.
        ring_size: Number of hex rings from center.
        resolution: H3 resolution.

    Returns:
        GeoDataFrame with H3 cells.
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


# Study area: Portland, Oregon central city
planning_grid = create_planning_grid(
    center_lat=45.5152, center_lng=-122.6784,
    ring_size=20, resolution=9
)
print(f"Planning grid: {len(planning_grid)} cells at resolution 9")
```

### Generating Zoning Data

```python
def generate_zoning_data(
    grid: gpd.GeoDataFrame,
    seed: int = 42
) -> gpd.GeoDataFrame:
    """Generate synthetic zoning designations for each H3 cell.

    Assigns zone types based on distance from center:
    - Core: commercial, mixed-use
    - Inner ring: residential high-density
    - Middle ring: residential medium-density
    - Outer ring: residential low-density, industrial

    Args:
        grid: H3 grid GeoDataFrame.
        seed: Random seed.

    Returns:
        Grid with zoning columns added.
    """
    rng = np.random.default_rng(seed)

    centroids = grid.geometry.centroid
    center = np.array([-122.6784, 45.5152])
    distances = np.sqrt(
        (centroids.x.values - center[0]) ** 2 +
        (centroids.y.values - center[1]) ** 2
    )
    max_dist = distances.max()
    normalized_dist = distances / max_dist

    zones = []
    for d in normalized_dist:
        if d < 0.15:
            zone = rng.choice(["commercial", "mixed_use"], p=[0.4, 0.6])
        elif d < 0.35:
            zone = rng.choice(["mixed_use", "residential_high"], p=[0.3, 0.7])
        elif d < 0.60:
            zone = rng.choice(["residential_high", "residential_medium"], p=[0.4, 0.6])
        elif d < 0.85:
            zone = rng.choice(["residential_medium", "residential_low"], p=[0.5, 0.5])
        else:
            zone = rng.choice(["residential_low", "industrial", "open_space"], p=[0.5, 0.3, 0.2])
        zones.append(zone)

    result = grid.copy()
    result["zone_type"] = zones

    # Floor area ratio (FAR)
    far_map = {
        "commercial": 4.0, "mixed_use": 3.0,
        "residential_high": 2.5, "residential_medium": 1.5,
        "residential_low": 0.5, "industrial": 1.0, "open_space": 0.1,
    }
    result["max_far"] = result["zone_type"].map(far_map)

    # Current built FAR (fraction of permitted)
    built_fraction = rng.uniform(0.3, 0.9, len(result))
    result["current_far"] = result["max_far"] * built_fraction
    result["development_capacity"] = result["max_far"] - result["current_far"]

    return result


zoned_grid = generate_zoning_data(planning_grid)
print(f"Zone distribution:")
print(zoned_grid["zone_type"].value_counts())
print(f"\nMean development capacity (FAR): {zoned_grid['development_capacity'].mean():.2f}")
```

### Mixed-Use Index

```python
def compute_mixed_use_index(
    grid: gpd.GeoDataFrame,
    radius_rings: int = 2
) -> gpd.GeoDataFrame:
    """Compute a mixed-use index for each cell based on surrounding zone diversity.

    The index uses Shannon entropy of zone types within a neighborhood.
    Higher values indicate more diverse (mixed) land use.

    Args:
        grid: Zoned grid GeoDataFrame.
        radius_rings: Number of H3 rings to include in neighborhood.

    Returns:
        Grid with 'mixed_use_index' column (0 = homogeneous, 1 = maximum mix).
    """
    zone_types = sorted(grid["zone_type"].unique())
    n_types = len(zone_types)
    max_entropy = np.log(n_types) if n_types > 1 else 1.0

    # Build a lookup dict for cell -> zone_type
    cell_to_zone = dict(zip(grid["h3_index"], grid["zone_type"]))

    mixed_use = np.zeros(len(grid))

    for idx, row in grid.iterrows():
        cell = row["h3_index"]
        neighbors = list(h3.grid_disk(cell, radius_rings))

        # Count zone types in neighborhood
        zone_counts = {}
        total = 0
        for neighbor in neighbors:
            if neighbor in cell_to_zone:
                zt = cell_to_zone[neighbor]
                zone_counts[zt] = zone_counts.get(zt, 0) + 1
                total += 1

        if total == 0:
            mixed_use[idx] = 0.0
            continue

        # Shannon entropy
        entropy = 0.0
        for count in zone_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)

        mixed_use[idx] = entropy / max_entropy

    result = grid.copy()
    result["mixed_use_index"] = mixed_use
    return result


mixed_grid = compute_mixed_use_index(zoned_grid)
print(f"Mixed-use index range: {mixed_grid['mixed_use_index'].min():.3f} - "
      f"{mixed_grid['mixed_use_index'].max():.3f}")
print(f"Mean mixed-use index: {mixed_grid['mixed_use_index'].mean():.3f}")
```

## Section 2: Transport Network Optimization

Transport accessibility is a key driver of land value and livability. This section builds a road network graph and computes centrality metrics and accessibility isochrones.

### Building the Network Graph

```python
def build_road_network(
    grid: gpd.GeoDataFrame,
    connectivity: float = 0.4,
    seed: int = 42
) -> nx.Graph:
    """Build a synthetic road network graph on the H3 grid.

    Each H3 cell is a node. Edges connect neighboring cells with
    probability proportional to the connectivity parameter and
    built-up intensity.

    Args:
        grid: H3 grid GeoDataFrame.
        connectivity: Base probability of edge existence between neighbors.
        seed: Random seed.

    Returns:
        NetworkX Graph with travel time edge weights.
    """
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    cell_set = set(grid["h3_index"])

    # Add nodes
    for _, row in grid.iterrows():
        G.add_node(
            row["h3_index"],
            lat=row["center_lat"],
            lng=row["center_lng"],
            zone=row.get("zone_type", "unknown"),
        )

    # Add edges between neighbors
    for _, row in grid.iterrows():
        cell = row["h3_index"]
        neighbors = h3.grid_ring(cell, 1)
        for neighbor in neighbors:
            if neighbor in cell_set and not G.has_edge(cell, neighbor):
                if rng.random() < connectivity:
                    # Travel time in minutes (shorter in urban core)
                    base_time = 2.0 + rng.exponential(1.5)
                    G.add_edge(cell, neighbor, travel_time=base_time)

    # Ensure connectivity by adding edges to isolated nodes
    components = list(nx.connected_components(G))
    if len(components) > 1:
        main_component = max(components, key=len)
        for component in components:
            if component == main_component:
                continue
            # Connect to nearest node in main component
            node = list(component)[0]
            nearest = min(
                main_component,
                key=lambda n: abs(G.nodes[n]["lat"] - G.nodes[node]["lat"]) +
                              abs(G.nodes[n]["lng"] - G.nodes[node]["lng"])
            )
            G.add_edge(node, nearest, travel_time=5.0)

    return G


road_network = build_road_network(mixed_grid)
print(f"Network: {road_network.number_of_nodes()} nodes, "
      f"{road_network.number_of_edges()} edges")
print(f"Connected: {nx.is_connected(road_network)}")
```

### Network Centrality Analysis

```python
from geo_infer_transport.core.network import NetworkAnalyzer


def compute_network_centrality(
    G: nx.Graph,
    grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Compute betweenness and closeness centrality for each node.

    Betweenness identifies nodes that serve as bridges (critical for
    network flow). Closeness identifies nodes with shortest average
    paths (most accessible locations).

    Args:
        G: Road network graph.
        grid: H3 grid GeoDataFrame.

    Returns:
        Grid with centrality columns added.
    """
    # Betweenness centrality (weighted by travel time)
    betweenness = nx.betweenness_centrality(G, weight="travel_time")

    # Closeness centrality
    closeness = nx.closeness_centrality(G, distance="travel_time")

    result = grid.copy()
    result["betweenness_centrality"] = result["h3_index"].map(betweenness).fillna(0)
    result["closeness_centrality"] = result["h3_index"].map(closeness).fillna(0)

    return result


transport_grid = compute_network_centrality(road_network, mixed_grid)
print(f"Betweenness range: {transport_grid['betweenness_centrality'].min():.4f} - "
      f"{transport_grid['betweenness_centrality'].max():.4f}")
print(f"Closeness range: {transport_grid['closeness_centrality'].min():.4f} - "
      f"{transport_grid['closeness_centrality'].max():.4f}")
```

### Accessibility Isochrones

```python
def compute_isochrone(
    G: nx.Graph,
    origin: str,
    max_time_minutes: float = 15.0
) -> list[str]:
    """Compute an isochrone: all cells reachable within a time limit.

    Uses Dijkstra's algorithm on the travel time-weighted network.

    Args:
        G: Road network graph.
        origin: Origin H3 cell index.
        max_time_minutes: Maximum travel time.

    Returns:
        List of H3 cell indexes within the isochrone.
    """
    if origin not in G:
        return [origin]

    distances = nx.single_source_dijkstra_path_length(
        G, origin, cutoff=max_time_minutes, weight="travel_time"
    )
    return list(distances.keys())


def compute_accessibility_score(
    G: nx.Graph,
    grid: gpd.GeoDataFrame,
    time_thresholds: list[float] | None = None
) -> gpd.GeoDataFrame:
    """Compute accessibility score: how many cells are reachable.

    The score is the fraction of total grid cells reachable within
    the given time threshold.

    Args:
        G: Road network graph.
        grid: H3 grid GeoDataFrame.
        time_thresholds: List of time thresholds in minutes.

    Returns:
        Grid with accessibility columns per threshold.
    """
    if time_thresholds is None:
        time_thresholds = [5.0, 10.0, 15.0]

    total_cells = len(grid)
    result = grid.copy()

    for threshold in time_thresholds:
        col_name = f"access_{int(threshold)}min"
        scores = []
        for cell in grid["h3_index"]:
            reachable = compute_isochrone(G, cell, threshold)
            scores.append(len(reachable) / total_cells)
        result[col_name] = scores

    return result


access_grid = compute_accessibility_score(road_network, transport_grid)
for col in ["access_5min", "access_10min", "access_15min"]:
    mean_val = access_grid[col].mean()
    print(f"{col}: mean reachable fraction = {mean_val:.3f}")
```

## Section 3: Green Space Scoring

Access to parks and vegetation affects property values, health outcomes, and equity. This section computes distance-to-green-space metrics and equity analysis across the planning area.

### Generating Green Space Data

```python
def generate_green_spaces(
    grid: gpd.GeoDataFrame,
    n_parks: int = 8,
    seed: int = 42
) -> gpd.GeoDataFrame:
    """Generate synthetic park and green space polygons.

    Distributes parks across the study area with varying sizes.
    Park placement is biased toward lower-density areas.

    Args:
        grid: Planning grid GeoDataFrame.
        n_parks: Number of parks to generate.
        seed: Random seed.

    Returns:
        GeoDataFrame with park polygons and attributes.
    """
    rng = np.random.default_rng(seed)

    # Sample park locations from the grid
    # Prefer cells with lower density (more likely to be parks)
    if "current_far" in grid.columns:
        weights = 1.0 / (grid["current_far"].values + 0.1)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(grid)) / len(grid)

    park_indices = rng.choice(len(grid), size=n_parks, replace=False, p=weights)

    parks = []
    for i, idx in enumerate(park_indices):
        center = grid.geometry.centroid.iloc[idx]
        # Park size: 0.001 to 0.005 degrees (~100m to ~500m)
        park_radius = rng.uniform(0.001, 0.005)
        park_poly = center.buffer(park_radius)

        parks.append({
            "park_id": f"park_{i:03d}",
            "park_name": f"Park {chr(65 + i)}",
            "geometry": park_poly,
            "area_ha": park_poly.area * 111000 * 111000 / 10000,  # rough conversion
            "park_type": rng.choice(["neighborhood", "community", "regional"]),
        })

    return gpd.GeoDataFrame(parks, crs="EPSG:4326")


def compute_green_space_access(
    grid: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Compute distance-to-nearest-green-space for each H3 cell.

    Metrics:
    - distance_to_park_m: Euclidean distance to nearest park centroid
    - within_400m: Boolean flag for the 5-minute walk standard
    - green_space_score: Normalized accessibility (1 = best)

    Args:
        grid: Planning grid GeoDataFrame.
        parks: Green space GeoDataFrame.

    Returns:
        Grid with green space access columns.
    """
    grid_projected = grid.to_crs(epsg=32610)
    parks_projected = parks.to_crs(epsg=32610)

    cell_centroids = grid_projected.geometry.centroid
    park_centroids = parks_projected.geometry.centroid

    # Compute distance to nearest park for each cell
    distances = np.zeros(len(grid))
    for i, cell_pt in enumerate(cell_centroids):
        min_dist = float("inf")
        for park_pt in park_centroids:
            d = cell_pt.distance(park_pt)
            if d < min_dist:
                min_dist = d
        distances[i] = min_dist

    result = grid.copy()
    result["distance_to_park_m"] = distances
    result["within_400m"] = distances <= 400
    max_dist = distances.max()
    result["green_space_score"] = 1.0 - (distances / max(max_dist, 1.0))

    return result


parks = generate_green_spaces(access_grid)
green_grid = compute_green_space_access(access_grid, parks)
pct_within_400m = green_grid["within_400m"].mean() * 100
print(f"Parks generated: {len(parks)}")
print(f"Cells within 400m of park: {pct_within_400m:.1f}%")
print(f"Mean green space score: {green_grid['green_space_score'].mean():.3f}")
```

### Equity Analysis

```python
def green_space_equity(
    grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Analyze green space access equity across zone types.

    Compares mean distance to parks and the fraction of cells
    within 400m by zone type. Equitable access means all zones
    have similar scores.

    Args:
        grid: Grid with zone_type and green space access columns.

    Returns:
        DataFrame summarizing equity metrics by zone.
    """
    equity = grid.groupby("zone_type").agg(
        mean_distance_m=("distance_to_park_m", "mean"),
        pct_within_400m=("within_400m", lambda x: x.mean() * 100),
        mean_green_score=("green_space_score", "mean"),
        cell_count=("h3_index", "count"),
    ).round(1)

    return equity.sort_values("mean_green_score", ascending=True)


equity_report = green_space_equity(green_grid)
print("Green Space Equity Report:")
print(equity_report.to_string())
```

## Section 4: Participatory Planning Integration

Citizen input adds a preference layer that complements technical analysis. This section aggregates community feedback to the H3 grid and uses Active Inference to model policy preferences.

### Aggregating Citizen Input

```python
def generate_citizen_input(
    grid: gpd.GeoDataFrame,
    n_responses: int = 500,
    seed: int = 42
) -> gpd.GeoDataFrame:
    """Generate synthetic citizen survey responses geolocated to H3 cells.

    Each response indicates priority preferences: housing, transit,
    parks, jobs, safety (rated 1-5). Preferences correlate spatially
    with local conditions.

    Args:
        grid: Planning grid.
        n_responses: Number of survey responses.
        seed: Random seed.

    Returns:
        GeoDataFrame of survey responses with H3 cell assignments.
    """
    rng = np.random.default_rng(seed)

    # Sample response locations
    response_cells = rng.choice(grid["h3_index"].values, size=n_responses, replace=True)

    # Build lookup for grid attributes
    cell_data = grid.set_index("h3_index")

    responses = []
    for cell in response_cells:
        row = cell_data.loc[cell]

        # Preferences influenced by local conditions
        zone = row.get("zone_type", "residential_medium")
        green_score = row.get("green_space_score", 0.5)

        # High-density zones prioritize transit; low-density prioritize housing
        if zone in ("commercial", "mixed_use"):
            transit_pref = rng.integers(3, 6)
            housing_pref = rng.integers(2, 5)
        elif zone in ("residential_high", "residential_medium"):
            transit_pref = rng.integers(2, 5)
            housing_pref = rng.integers(3, 6)
        else:
            transit_pref = rng.integers(1, 4)
            housing_pref = rng.integers(2, 5)

        # Low green access -> higher park preference
        park_pref = max(1, min(5, int(5 * (1 - green_score) + rng.normal(0, 0.5))))

        responses.append({
            "h3_index": cell,
            "housing_priority": housing_pref,
            "transit_priority": transit_pref,
            "parks_priority": park_pref,
            "jobs_priority": rng.integers(2, 6),
            "safety_priority": rng.integers(2, 6),
        })

    return pd.DataFrame(responses)


citizen_df = generate_citizen_input(green_grid, n_responses=500)
print(f"Citizen responses: {len(citizen_df)}")
print(f"Priority means:")
for col in ["housing_priority", "transit_priority", "parks_priority",
            "jobs_priority", "safety_priority"]:
    print(f"  {col}: {citizen_df[col].mean():.2f}")
```

### Aggregating to Grid

```python
def aggregate_citizen_preferences(
    citizen_df: pd.DataFrame,
    grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Aggregate citizen preferences to H3 cells.

    Computes mean priority scores per cell and a composite
    community priority index.

    Args:
        citizen_df: Survey response DataFrame.
        grid: Planning grid.

    Returns:
        Grid with community priority columns.
    """
    priority_cols = [c for c in citizen_df.columns if c.endswith("_priority")]

    agg = citizen_df.groupby("h3_index")[priority_cols].agg(["mean", "count"]).reset_index()
    agg.columns = ["h3_index"] + [
        f"{col}_{stat}" for col in priority_cols for stat in ["mean", "count"]
    ]

    # Simplify: take just the mean columns
    mean_cols = {f"{col}_mean": f"community_{col}" for col in priority_cols}
    for old, new in mean_cols.items():
        if old in agg.columns:
            agg = agg.rename(columns={old: new})

    # Response count (use first priority's count)
    count_col = f"{priority_cols[0]}_count"
    if count_col in agg.columns:
        agg = agg.rename(columns={count_col: "response_count"})

    # Keep only relevant columns
    keep_cols = ["h3_index", "response_count"] + list(mean_cols.values())
    keep_cols = [c for c in keep_cols if c in agg.columns]
    agg = agg[keep_cols]

    result = grid.merge(agg, on="h3_index", how="left")
    for col in mean_cols.values():
        if col in result.columns:
            result[col] = result[col].fillna(3.0)  # neutral default
    result["response_count"] = result.get("response_count", pd.Series(0)).fillna(0)

    return result


community_grid = aggregate_citizen_preferences(citizen_df, green_grid)
print(f"Cells with community input: {(community_grid['response_count'] > 0).sum()}")
```

### Active Inference for Policy Preferences

```python
def compute_policy_preferences(
    grid: gpd.GeoDataFrame,
    priority_weights: dict[str, float] | None = None
) -> gpd.GeoDataFrame:
    """Model planning policy preferences using a simplified Active Inference approach.

    The policy preference score combines community priorities with
    technical analysis (accessibility, green space, development capacity)
    using a free-energy-like objective: minimize surprise between
    community expectations and current conditions.

    Args:
        grid: Grid with community priorities and technical scores.
        priority_weights: Weights for each priority dimension.

    Returns:
        Grid with policy preference scores.
    """
    if priority_weights is None:
        priority_weights = {
            "housing": 0.25,
            "transit": 0.25,
            "parks": 0.20,
            "jobs": 0.15,
            "safety": 0.15,
        }

    result = grid.copy()

    # Technical supply scores (what the area currently provides)
    supply = {
        "housing": grid.get("current_far", pd.Series(1.0)).values / 4.0,
        "transit": grid.get("closeness_centrality", pd.Series(0.5)).values,
        "parks": grid.get("green_space_score", pd.Series(0.5)).values,
        "jobs": np.where(
            grid["zone_type"].isin(["commercial", "mixed_use", "industrial"]),
            0.7, 0.3
        ),
        "safety": np.full(len(grid), 0.6),  # baseline
    }

    # Community demand (from survey, scaled to 0-1)
    demand = {}
    for dim in priority_weights:
        col = f"community_{dim}_priority"
        if col in grid.columns:
            vals = grid[col].values
            demand[dim] = (vals - 1.0) / 4.0  # scale 1-5 to 0-1
        else:
            demand[dim] = np.full(len(grid), 0.5)

    # Gap analysis: where demand exceeds supply
    # Positive gap = unmet need
    total_gap = np.zeros(len(grid))
    for dim, weight in priority_weights.items():
        gap = np.maximum(demand[dim] - supply[dim], 0.0)
        result[f"gap_{dim}"] = gap
        total_gap += weight * gap

    result["total_unmet_need"] = total_gap

    # Policy intervention priority: high gap + high community engagement
    engagement = result.get("response_count", pd.Series(0)).values
    max_engagement = max(engagement.max(), 1)
    engagement_norm = engagement / max_engagement
    result["intervention_priority"] = total_gap * (0.5 + 0.5 * engagement_norm)

    return result


policy_grid = compute_policy_preferences(community_grid)
print(f"Mean unmet need: {policy_grid['total_unmet_need'].mean():.3f}")
print(f"Top unmet gaps:")
for dim in ["housing", "transit", "parks", "jobs", "safety"]:
    col = f"gap_{dim}"
    if col in policy_grid.columns:
        print(f"  {dim}: {policy_grid[col].mean():.3f}")
```

## Section 5: Full Planning Workflow -- Site Selection

The final workflow combines all layers into a multi-criteria analysis for selecting the best location for a new community facility (e.g., a community center with park space).

### Multi-Criteria Site Selection

```python
def site_selection_analysis(
    grid: gpd.GeoDataFrame,
    criteria_weights: dict[str, float] | None = None,
    n_candidates: int = 10
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Perform multi-criteria site selection for a community facility.

    Criteria:
    - High unmet community need (demand-supply gap)
    - Good transit accessibility
    - Development capacity available
    - Low current green space access (new park would help)
    - Community engagement (responses indicate interest)

    Args:
        grid: Fully-attributed planning grid.
        criteria_weights: Relative weights for each criterion.
        n_candidates: Number of top candidate sites to return.

    Returns:
        Tuple of (scored_grid, top_candidates) GeoDataFrames.
    """
    if criteria_weights is None:
        criteria_weights = {
            "unmet_need": 0.30,
            "transit_access": 0.20,
            "dev_capacity": 0.15,
            "green_gap": 0.20,
            "engagement": 0.15,
        }

    def norm_01(arr: np.ndarray) -> np.ndarray:
        r = arr.max() - arr.min()
        if r == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / r

    scored = grid.copy()

    # Criterion scores (all normalized 0-1, higher = better candidate)
    scores = {
        "unmet_need": norm_01(
            grid.get("total_unmet_need", pd.Series(0.5)).values
        ),
        "transit_access": norm_01(
            grid.get("closeness_centrality", pd.Series(0.5)).values
        ),
        "dev_capacity": norm_01(
            grid.get("development_capacity", pd.Series(0.5)).values
        ),
        "green_gap": 1.0 - norm_01(
            grid.get("green_space_score", pd.Series(0.5)).values
        ),
        "engagement": norm_01(
            grid.get("response_count", pd.Series(0)).values.astype(float)
        ),
    }

    # Composite suitability score
    composite = np.zeros(len(grid))
    for name, weight in criteria_weights.items():
        col_name = f"criterion_{name}"
        scored[col_name] = scores[name]
        composite += weight * scores[name]

    scored["suitability_score"] = composite

    # Select top candidates
    top_idx = np.argsort(composite)[-n_candidates:][::-1]
    candidates = scored.iloc[top_idx].copy()
    candidates["rank"] = range(1, n_candidates + 1)

    return scored, candidates


scored_grid, candidates = site_selection_analysis(policy_grid, n_candidates=10)

print("Top 10 candidate sites for community facility:")
display_cols = ["rank", "h3_index", "zone_type", "suitability_score",
                "total_unmet_need", "green_space_score"]
display_cols = [c for c in display_cols if c in candidates.columns]
print(candidates[display_cols].to_string(index=False))
```

### Decision Support Output

```python
def generate_planning_report(
    candidates: gpd.GeoDataFrame,
    scored_grid: gpd.GeoDataFrame
) -> str:
    """Generate a planning decision support report.

    Args:
        candidates: Top candidate sites.
        scored_grid: Full scored grid.

    Returns:
        Formatted report string.
    """
    best = candidates.iloc[0]
    lines = [
        "=" * 70,
        "SITE SELECTION REPORT: New Community Facility",
        "=" * 70,
        "",
        "RECOMMENDED SITE",
        f"  H3 Cell: {best['h3_index']}",
        f"  Location: ({best['center_lat']:.4f}, {best['center_lng']:.4f})",
        f"  Current Zoning: {best.get('zone_type', 'N/A')}",
        f"  Suitability Score: {best['suitability_score']:.3f}",
        "",
        "CRITERION BREAKDOWN",
    ]

    for col in candidates.columns:
        if col.startswith("criterion_"):
            name = col.replace("criterion_", "")
            lines.append(f"  {name}: {best[col]:.3f}")

    lines.extend([
        "",
        "CONTEXT",
        f"  Development Capacity (FAR): {best.get('development_capacity', 'N/A')}",
        f"  Green Space Score: {best.get('green_space_score', 'N/A'):.3f}" if 'green_space_score' in best.index else "",
        f"  Community Responses: {best.get('response_count', 0):.0f}",
        "",
        f"ALTERNATIVES: {len(candidates) - 1} additional candidates evaluated",
        "See candidates GeoDataFrame for full comparison.",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


report_text = generate_planning_report(candidates, scored_grid)
print(report_text)
```

### Site Selection Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Suitability map
scored_grid.plot(
    column="suitability_score",
    ax=axes[0], legend=True, cmap="YlOrRd",
    legend_kwds={"label": "Suitability score"},
)
if len(candidates) > 0:
    candidates.plot(ax=axes[0], facecolor="none", edgecolor="black", linewidth=2)
axes[0].set_title("Site Suitability")

# Unmet need
if "total_unmet_need" in scored_grid.columns:
    scored_grid.plot(
        column="total_unmet_need",
        ax=axes[1], legend=True, cmap="Purples",
        legend_kwds={"label": "Unmet need"},
    )
axes[1].set_title("Community Unmet Need")

# Green space access
if "green_space_score" in scored_grid.columns:
    scored_grid.plot(
        column="green_space_score",
        ax=axes[2], legend=True, cmap="Greens",
        legend_kwds={"label": "Green space access"},
    )
    parks.plot(ax=axes[2], facecolor="green", alpha=0.5, edgecolor="darkgreen")
axes[2].set_title("Green Space Access")

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("site_selection.png", dpi=150)
```

## Outputs Summary

| Output | Description | Module(s) |
|--------|-------------|-----------|
| Zoning distribution table | Zone type counts and development capacity | CIV |
| Mixed-use index map | Shannon entropy of surrounding zone diversity | SPACE, CIV |
| Network centrality maps | Betweenness and closeness per H3 cell | TRANSPORT |
| Accessibility isochrones | Fraction of city reachable at 5/10/15 min | TRANSPORT |
| Green space equity report | Distance-to-park by zone type | SPACE |
| Site selection report | Ranked candidate locations with criteria scores | All modules |
| `site_selection.png` | Three-panel suitability, need, and green space maps | All modules |

## Next Steps

- **Real data**: Integrate OSM road networks via GEO-INFER-DATA for realistic transport analysis
- **Risk overlay**: Combine with [Urban Analytics](urban_analytics.md) hazard layers for risk-aware site selection
- **Climate projections**: Account for future heat island changes using [Climate Modeling](climate_modeling.md)
- **Temporal tracking**: Monitor how community needs evolve using GEO-INFER-TIME repeat surveys
- **Scaling**: For metropolitan-scale analysis, see [Scaling Guide](../advanced/scaling_guide.md) and [Memory Management](../advanced/memory_management.md)
