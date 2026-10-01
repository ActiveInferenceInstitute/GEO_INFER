# GEO-INFER Module Integration

This document describes how the 45 GEO-INFER modules communicate, share data, and compose into analysis pipelines.

## Data Flow Architecture

Data moves through GEO-INFER in a layered pipeline. Foundation modules have zero internal dependencies, and each successive layer builds on the one below it.

```
Data Sources --> GEO-INFER-DATA --> SPACE/TIME --> MATH/BAYES/ACT --> AI/AGENT --> Domain Modules --> API/APP
```

### Layer Breakdown

| Layer | Modules | Role |
|-------|---------|------|
| **Foundation** | MATH | Linear algebra, statistics, transforms. No dependencies. |
| **Core Analytics** | BAYES, ACT | Bayesian inference, Active Inference. Depend on MATH. |
| **Spatial-Temporal** | SPACE, TIME, IOT | Coordinate systems, H3 indexing, time series, sensors. |
| **Intelligence** | AI, COG, AGENT | Machine learning, cognitive models, multi-agent coordination. |
| **Domain** | AG, HEALTH, ECON, RISK, LOG, BIO, CLIMATE, ENERGY, FOREST, MARINE, EMERGENCY, EDU, TRANSPORT, WATER | Specialized geospatial analysis per vertical. |
| **Infrastructure** | DATA, API, SEC, OPS, METAGOV, NORMS, REQ | Data management, API gateway, security, operations. |
| **Operations** | INTRA, GIT, TEST, EXAMPLES, PLACE | Documentation, version control, testing, demos, location intelligence. |

## Layer Roles

The layers above describe dependency direction. The thematic grouping of all
45 modules is the root README "Module Themes" table.

### Analytical Core

MATH, ACT, BAYES, AI, COG, AGENT, and SPM form the analytical backbone. MATH provides the primitives (spatial statistics, coordinate transforms, topology). ACT and BAYES build on MATH for Active Inference and Bayesian reasoning. AI and AGENT sit on top for machine learning and multi-agent coordination.

### Spatial-Temporal

SPACE handles H3 v4 hexagonal indexing, spatial joins, and coordinate transformations. TIME handles temporal decomposition, forecasting, and change detection. IOT bridges real-time sensor networks into the spatial-temporal layer.

### Domain-Specific

Each domain module (AG, HEALTH, ECON, RISK, etc.) consumes outputs from the analytical core and spatial-temporal layers, then applies domain logic (crop models, epidemiological models, risk scoring).

### Infrastructure

DATA manages ETL and data formats. API exposes module functionality over REST/GraphQL. SEC handles authentication and encryption. OPS manages deployment, monitoring, and logging.

## Installing Multiple Modules

GEO-INFER is a uv workspace; `uv sync` installs members in editable mode from
the root `uv.lock`:

```bash
# Every module with every optional extra (what CI uses)
uv sync --all-packages --all-extras --all-groups

# One module and its workspace dependencies
uv sync --package geo-infer-ag

# One module with optional extras (dev tools, documentation)
uv sync --package geo-infer-ai --extra dev --extra docs
```

Module load order does not matter at install time; Python resolves imports at runtime. However, if a module's optional dependency is missing, its `__init__.py` uses `try/except` to degrade gracefully:

```python
# Pattern used in every GEO-INFER module's __init__.py
try:
    from geo_infer_bayes import GaussianProcess
except ImportError:
    GaussianProcess = None  # GEO-INFER-BAYES not installed
```

## Cross-Module Data Formats

Modules exchange data using standard geospatial formats. No custom serialization protocols are needed.

### GeoJSON

The default interchange format for vector features between modules.

```python
import json

feature_collection = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-122.4, 47.6]},
            "properties": {"temperature": 18.5, "station_id": "SEA-001"},
        }
    ],
}
```

### GeoParquet

For large datasets, GeoParquet provides columnar storage with spatial metadata.

```python
import geopandas as gpd

# Write analysis results as GeoParquet
gdf = gpd.GeoDataFrame(...)
gdf.to_parquet("analysis_output.parquet")

# Read in another module
gdf = gpd.read_parquet("analysis_output.parquet")
```

### H3 Cell Indices

GEO-INFER-SPACE uses H3 v4 for hexagonal spatial indexing. H3 cell IDs (integers) are the standard spatial key for joining data across modules.

```python
import h3

# H3 v4 API
cell = h3.latlng_to_cell(47.6, -122.4, res=7)
lat, lng = h3.cell_to_latlng(cell)
neighbors = h3.grid_disk(cell, k=1)
```

### NumPy Arrays

Raster data and matrix computations pass as NumPy arrays. MATH, BAYES, and AI modules all operate on `numpy.ndarray` directly.

## Multi-Module Workflow Examples

### Spatial Analysis with Bayesian Uncertainty

This pipeline reads spatial data, computes statistics with GEO-INFER-MATH, then quantifies uncertainty with GEO-INFER-BAYES.

```python
from geo_infer_math import MoranI
from geo_infer_math.core.transforms import CoordinateTransformer
from geo_infer_bayes import GaussianProcess
import geopandas as gpd
import numpy as np

# Load spatial data
gdf = gpd.read_file("soil_samples.geojson")
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
values = gdf["nitrogen_ppm"].values

# Compute spatial autocorrelation (GEO-INFER-MATH)
morans_i = MoranI().compute(values, coords=coords)
print(f"Moran's I: {morans_i['I']:.3f} (p={morans_i['p_value']:.4f})")

# Fit a Gaussian Process for interpolation (GEO-INFER-BAYES)
gp = GaussianProcess(kernel_type="matern32", length_scale=0.1)
gp.fit(coords, values)

# Predict on a grid with uncertainty estimates
grid_x = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 50)
grid_y = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 50)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])

mean, std = gp.predict(grid_points, return_std=True)
```

### Active Inference with Spatial Context

Combine GEO-INFER-ACT with GEO-INFER-SPACE for spatially-aware Active Inference.

```python
from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_space.backends.h3.h3_backend import H3Backend
import numpy as np

# Initialize spatial backend (GEO-INFER-SPACE)
h3_backend = H3Backend()
region_cells = h3_backend.polygon_to_cells(region_boundary, resolution=7)

# Build a generative model over the spatial region (GEO-INFER-ACT)
agent = ActiveInferenceModel(
    model_type="categorical",
    num_states=len(region_cells),
    num_observations=4,  # e.g., land cover categories
    num_actions=3,
)

# Perception-action loop
for timestep in range(100):
    observation = environment.observe(agent.current_state)
    agent.update_beliefs(observation)
    action = agent.select_action()
    agent.execute_action(action)
```

### Multi-Module Domain Pipeline

A complete agricultural analysis combining four modules:

```python
from geo_infer_space.backends.h3.h3_backend import H3Backend
from geo_infer_math import MoranI
from geo_infer_bayes.core.model_comparison import ModelComparison

# Step 1: Spatial indexing (SPACE)
h3 = H3Backend()
field_cells = h3.polygon_to_cells(field_boundary, resolution=9)

# Step 2: Aggregate sensor data per cell (SPACE + DATA)
cell_data = {}
for cell in field_cells:
    cell_boundary = h3.cell_to_boundary(cell)
    cell_data[cell] = aggregate_sensors_in_polygon(cell_boundary, sensor_readings)

# Step 3: Spatial statistics (MATH)
stats = SpatialStatistics()
values = np.array([cell_data[c]["soil_moisture"] for c in field_cells])
coords = np.array([h3.cell_to_latlng(c) for c in field_cells])
autocorrelation = stats.morans_i(coords, values)

# Step 4: Model comparison (BAYES)
comparison = ModelComparison()
results = comparison.compare_models(
    models=["linear", "gp", "bayesian_ridge"],
    X=coords,
    y=values,
    criteria=["loo", "waic"],
)
best_model = results.best_model
```

## Dependency Management with uv

The `uv` package manager handles all Python dependencies. Each module's
`pyproject.toml` is its only dependency declaration (no `setup.py` or
`requirements.txt`), and the root `uv.lock` resolves all of them together.

```bash
# Install all modules for integration testing
uv sync --all-packages --all-extras --all-groups

# Add a dependency to one module and refresh uv.lock
uv add --package geo-infer-space "pyproj>=3.0"

# Check what is installed
uv pip list --python .venv | grep geo-infer
```

### Resolving Dependency Conflicts

If two modules pin conflicting versions of a shared dependency, `uv` will report the conflict. Resolution options:

1. Relax the version constraint in one module's `pyproject.toml`, then run `uv lock`.
2. Align shared floors across the root and module `pyproject.toml` files (see
   the Dependency Floor Policy in `.agents/standards.md`).
3. Check the lower bounds with `uv lock --resolution lowest-direct` in a
   scratch checkout.

## Using GEO-INFER-API as the Integration Layer

For applications that need to consume multiple modules without importing them directly, GEO-INFER-API provides a unified REST interface.

```python
import httpx

API_BASE = "http://localhost:8000/api/v1"

# Spatial analysis via the API layer
response = httpx.post(
    f"{API_BASE}/space/h3/polyfill",
    json={
        "polygon": region_boundary,
        "resolution": 7,
    },
)
cells = response.json()["cells"]

# Bayesian model fitting via the API layer
response = httpx.post(
    f"{API_BASE}/bayes/gaussian-process/fit",
    json={
        "coordinates": coords.tolist(),
        "values": values.tolist(),
        "kernel": "matern",
    },
)
model_id = response.json()["model_id"]

# Prediction via the API layer
response = httpx.post(
    f"{API_BASE}/bayes/gaussian-process/predict",
    json={
        "model_id": model_id,
        "coordinates": grid_points.tolist(),
    },
)
predictions = response.json()["predictions"]
```

## Cross-Module Communication Patterns

### Direct Import (In-Process)

The simplest pattern. One module imports classes from another.

```python
from geo_infer_math.core.transforms import CoordinateTransformer
from geo_infer_space.backends.h3.h3_backend import H3Backend
```

Use when: both modules are installed in the same environment and latency must be minimal.

### API-Mediated (HTTP)

Modules communicate through GEO-INFER-API REST endpoints. Suitable for distributed deployments where modules run as separate services.

### Event-Driven (Kafka/Redis)

For asynchronous workflows, modules publish events to message queues. GEO-INFER-IOT and GEO-INFER-COMMS use this pattern for real-time data.

### File-Based (GeoParquet/GeoJSON)

For batch workflows, one module writes results to disk (or cloud storage) and another reads them. GEO-INFER-DATA manages the catalog of available datasets.

## Testing Cross-Module Integration

Run the unified test suite to validate that modules work together:

```bash
# All integration tests
uv run python GEO-INFER-TEST/run_unified_tests.py --category integration

# Integration tests for a specific module
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE --category integration

# Direct pytest for a specific integration test
uv run python -m pytest GEO-INFER-SPACE/tests/integration/ -v
```

Pytest markers relevant to cross-module tests: `integration`, `geospatial`, `slow`.
