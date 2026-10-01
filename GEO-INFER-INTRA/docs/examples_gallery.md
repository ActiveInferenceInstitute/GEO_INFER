# GEO-INFER Examples Gallery

This gallery indexes the example code and notebooks available in the GEO-INFER
framework. Examples are organized by topic and difficulty level. Each entry lists
the modules involved, estimated runtime, and a link to the source directory.

## Quick Navigation

| Category | Difficulty | Modules | Section |
|----------|-----------|---------|---------|
| Basic Usage | Beginner | Individual modules | [Basic Usage](#basic-usage) |
| Spatial Analysis | Beginner-Intermediate | SPACE, MATH, PLACE | [Spatial Analysis](#spatial-analysis) |
| Time Series | Intermediate | TIME, DATA | [Time Series](#time-series) |
| Bayesian Inference | Intermediate | BAYES, MATH | [Bayesian Inference](#bayesian-inference) |
| Active Inference | Advanced | ACT, AGENT, COG | [Active Inference](#active-inference) |
| Agricultural Intelligence | Intermediate-Advanced | AG, SPACE, TIME, DATA | [Agriculture](#agricultural-intelligence) |
| Climate Analysis | Intermediate-Advanced | CLIMATE, TIME, SPACE, DATA | [Climate](#climate-analysis) |
| Risk Assessment | Advanced | RISK, BAYES, SPACE, ECON | [Risk](#risk-assessment) |
| Multi-Module Integration | Advanced | 3+ modules | [Integration](#multi-module-integration) |
| Real World Applications | Advanced | Various | [Applications](#real-world-applications) |

## How to Run Examples

### Prerequisites

Install the required modules for the example you want to run:

```bash
# Install from the GEO-INFER root directory
cd /path/to/GEO_INFER

# Install every module (core and domain) with all optional extras
uv sync --all-packages --all-extras
```

### Running a Python Example

```bash
# Run directly
uv run python GEO-INFER-EXAMPLES/examples/spatial/h3_grid_analysis.py

# Or from within the examples directory
cd GEO-INFER-EXAMPLES/examples
uv run python spatial/h3_grid_analysis.py
```

### Running a Jupyter Notebook

```bash
uv run --with jupyter jupyter notebook GEO-INFER-EXAMPLES/examples/spatial/h3_grid_analysis.ipynb
```

### Environment Variables

Some examples require data paths or API keys:

```bash
export GEO_INFER_DATA_DIR="/path/to/sample/data"
export GEO_INFER_CACHE_DIR="/tmp/geo_infer_cache"
```

---

## Basic Usage

Entry-level examples demonstrating individual module functionality. Each example
focuses on a single module and requires no external data.

### MATH Module: Spatial Statistics

Compute Moran's I spatial autocorrelation on a synthetic dataset.

- **Difficulty**: Beginner
- **Modules**: GEO-INFER-MATH
- **Estimated Runtime**: < 30 seconds
- **Directory**: `GEO-INFER-EXAMPLES/examples/math/`

```python
from geo_infer_math import MoranI
import numpy as np

values = np.random.randn(100)
coordinates = np.random.rand(100, 2) * 10  # 100 random points in 10x10 space
result = MoranI().compute(values, coords=coordinates)
print(f"Moran's I: {result['I']:.4f}, p-value: {result['p_value']:.4f}")
```

### SPACE Module: H3 Grid Creation

Create an H3 hexagonal grid covering a bounding box.

- **Difficulty**: Beginner
- **Modules**: GEO-INFER-SPACE
- **Estimated Runtime**: < 10 seconds
- **Directory**: `GEO-INFER-EXAMPLES/examples/spatial/`

```python
import h3
import geopandas as gpd
from shapely.geometry import Polygon

# Portland metro bounding box
cells = set()
for lat in [45.45, 45.50, 45.55, 45.60]:
    for lng in [-122.75, -122.70, -122.65, -122.60]:
        cells.add(h3.latlng_to_cell(lat, lng, 8))

print(f"Generated {len(cells)} H3 cells at resolution 8")
```

### ACT Module: Free Energy Computation

Compute variational free energy for a simple categorical model.

- **Difficulty**: Beginner
- **Modules**: GEO-INFER-ACT
- **Estimated Runtime**: < 5 seconds
- **Directory**: `GEO-INFER-EXAMPLES/examples/active_inference/`

```python
from geo_infer_act.core.free_energy import FreeEnergyCalculator
import numpy as np

calc = FreeEnergyCalculator()
beliefs = np.array([0.25, 0.25, 0.25, 0.25])
observations = np.array([0.8, 0.1, 0.05, 0.05])
fe = calc.compute_categorical_free_energy(beliefs, observations)
print(f"Free energy: {fe:.4f}")
```

---

## Spatial Analysis

Examples covering H3 grid operations, spatial statistics, geometric analysis,
and map visualization.

### H3 Multi-Resolution Analysis

Compare spatial patterns at different H3 resolutions to identify scale-dependent
phenomena.

- **Difficulty**: Beginner-Intermediate
- **Modules**: GEO-INFER-SPACE, GEO-INFER-MATH
- **Estimated Runtime**: 30-60 seconds
- **Directory**: `GEO-INFER-EXAMPLES/examples/spatial/multi_resolution/`

### Spatial Clustering with DBSCAN

Identify spatial clusters of points using density-based clustering with
geographic distance metrics.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-SPACE, GEO-INFER-MATH
- **Estimated Runtime**: 1-2 minutes (depends on data size)
- **Directory**: `GEO-INFER-EXAMPLES/examples/spatial/clustering/`

### Voronoi Tessellation and Service Areas

Generate Voronoi polygons from facility locations and analyze service area
coverage.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-SPACE, GEO-INFER-CIV
- **Estimated Runtime**: 30-60 seconds
- **Directory**: `GEO-INFER-EXAMPLES/examples/spatial/voronoi/`

### Choropleth Map with GeoPandas

Create thematic maps from census or survey data using standard GeoDataFrame
styling.

- **Difficulty**: Beginner
- **Modules**: GEO-INFER-SPACE
- **Estimated Runtime**: < 30 seconds
- **Directory**: `GEO-INFER-EXAMPLES/examples/spatial/visualization/`

---

## Time Series

Temporal analysis examples covering forecasting, anomaly detection, and
seasonality decomposition.

### Temporal Decomposition

Decompose a geospatial time series into trend, seasonal, and residual
components using STL decomposition.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-TIME
- **Estimated Runtime**: 30-60 seconds
- **Directory**: `GEO-INFER-EXAMPLES/examples/temporal/decomposition/`

### Change Point Detection

Detect structural breaks in spatial time series data using Bayesian change
point analysis.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-TIME, GEO-INFER-BAYES
- **Estimated Runtime**: 1-3 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/temporal/changepoint/`

### Spatiotemporal Forecasting

Forecast values at spatial locations using time series models that incorporate
spatial correlation structure.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-TIME, GEO-INFER-SPACE, GEO-INFER-MATH
- **Estimated Runtime**: 2-5 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/temporal/forecasting/`

---

## Bayesian Inference

Probabilistic modeling examples using the GEO-INFER-BAYES module.

### Gaussian Process Regression

Fit a Gaussian Process to spatial data and generate interpolated predictions
with uncertainty bounds.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-BAYES, GEO-INFER-MATH
- **Estimated Runtime**: 1-3 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/bayesian/gp_regression/`

```python
from geo_infer_bayes import GaussianProcess
import numpy as np

# Training data: spatial coordinates and observed values
X_train = np.random.rand(50, 2) * 10  # 50 points in 2D space
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + np.random.randn(50) * 0.1

gp = GaussianProcess(kernel_type="rbf", length_scale=2.0)
gp.fit(X_train, y_train)

# Predict at new locations
X_test = np.random.rand(20, 2) * 10
mean, std = gp.predict(X_test, return_std=True)
print(f"Predictions: mean range [{mean.min():.2f}, {mean.max():.2f}]")
print(f"Uncertainty: std range [{std.min():.3f}, {std.max():.3f}]")
```

### Hierarchical Bayesian Model

Build a hierarchical model for spatial data where parameters vary across
regions but share common hyperpriors.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-BAYES, GEO-INFER-SPACE
- **Estimated Runtime**: 5-15 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/bayesian/hierarchical/`

### Model Comparison

Compare competing models using LOO-CV, WAIC, DIC, BIC, and AIC information
criteria.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-BAYES
- **Estimated Runtime**: 3-10 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/bayesian/model_comparison/`

---

## Active Inference

Examples demonstrating Active Inference principles in geospatial contexts.

### Free Energy Minimization

Step-by-step demonstration of belief updating through free energy minimization
over a sequence of spatial observations.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-ACT
- **Estimated Runtime**: 30-60 seconds
- **Directory**: `GEO-INFER-EXAMPLES/examples/active_inference/free_energy/`

### Spatial Belief Updating on H3 Grid

Update beliefs about land cover classification across an H3 grid as new
satellite observations arrive.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-ACT, GEO-INFER-SPACE
- **Estimated Runtime**: 1-3 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/active_inference/spatial_beliefs/`

### Policy Selection for Environmental Monitoring

Use expected free energy to select optimal sensor placement locations that
balance information gain and goal achievement.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-ACT, GEO-INFER-AGENT, GEO-INFER-SPACE
- **Estimated Runtime**: 2-5 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/active_inference/policy_selection/`

### Multi-Agent Active Inference

Coordinate multiple Active Inference agents performing collaborative spatial
exploration.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-ACT, GEO-INFER-AGENT, GEO-INFER-COG
- **Estimated Runtime**: 5-10 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/active_inference/multi_agent/`

---

## Agricultural Intelligence

Domain examples for precision agriculture and crop monitoring.

### Crop Yield Estimation

Estimate crop yield using satellite-derived vegetation indices, soil data,
and weather observations.

- **Difficulty**: Intermediate-Advanced
- **Modules**: GEO-INFER-AG, GEO-INFER-SPACE, GEO-INFER-TIME, GEO-INFER-DATA
- **Estimated Runtime**: 3-10 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/agriculture/yield_estimation/`

### Soil Moisture Prediction

Predict soil moisture at unsampled locations using spatial interpolation and
temporal modeling.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-AG, GEO-INFER-BAYES, GEO-INFER-SPACE
- **Estimated Runtime**: 2-5 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/agriculture/soil_moisture/`

### Field Boundary Detection

Detect agricultural field boundaries from satellite imagery using edge detection
and polygon extraction.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-AG, GEO-INFER-SPACE, GEO-INFER-AI
- **Estimated Runtime**: 5-15 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/agriculture/field_detection/`

---

## Climate Analysis

Climate science examples for anomaly detection, trend analysis, and
downscaling.

### Temperature Anomaly Detection

Detect statistically significant temperature anomalies relative to a
climatological baseline using spatial statistics.

- **Difficulty**: Intermediate-Advanced
- **Modules**: GEO-INFER-CLIMATE, GEO-INFER-TIME, GEO-INFER-MATH
- **Estimated Runtime**: 2-5 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/climate/anomaly_detection/`

### Trend Extraction

Extract long-term trends from multi-decadal climate records, separating
secular change from natural variability.

- **Difficulty**: Intermediate
- **Modules**: GEO-INFER-CLIMATE, GEO-INFER-TIME
- **Estimated Runtime**: 1-3 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/climate/trend_extraction/`

### Statistical Downscaling

Downscale coarse-resolution climate model output to fine spatial resolution
using transfer functions trained on observational data.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-CLIMATE, GEO-INFER-SPACE, GEO-INFER-BAYES, GEO-INFER-AI
- **Estimated Runtime**: 10-30 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/climate/downscaling/`

---

## Risk Assessment

Examples for hazard modeling, vulnerability analysis, and catastrophe modeling.

### Flood Hazard Mapping

Generate flood hazard maps using elevation data, hydrological modeling, and
return period analysis.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-RISK, GEO-INFER-SPACE, GEO-INFER-WATER
- **Estimated Runtime**: 5-15 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/risk/flood_hazard/`

### Vulnerability Assessment

Assess building vulnerability to natural hazards using fragility curves and
exposure databases.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-RISK, GEO-INFER-ECON, GEO-INFER-SPACE
- **Estimated Runtime**: 3-10 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/risk/vulnerability/`

### Catastrophe Model Pipeline

Run a complete catastrophe model pipeline: hazard generation, vulnerability
assessment, loss calculation, and financial impact analysis.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-RISK, GEO-INFER-BAYES, GEO-INFER-ECON, GEO-INFER-SPACE
- **Estimated Runtime**: 15-30 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/risk/catastrophe_model/`

---

## Multi-Module Integration

End-to-end pipelines combining three or more modules into complete workflows.

### Module Orchestrator

Use the GEO-INFER-EXAMPLES module orchestrator to chain operations across
multiple modules in a declarative pipeline.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-EXAMPLES (orchestrator), 3+ domain modules
- **Estimated Runtime**: 5-20 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/integration/orchestrator/`

```python
from geo_infer_examples.core.module_orchestrator import ModuleOrchestrator

orchestrator = ModuleOrchestrator()

# Define pipeline: load data -> spatial index -> temporal analysis -> prediction
pipeline = orchestrator.create_pipeline([
    {"module": "DATA", "operation": "load", "params": {"source": "sensor_network"}},
    {"module": "SPACE", "operation": "h3_index", "params": {"resolution": 9}},
    {"module": "TIME", "operation": "resample", "params": {"freq": "1H"}},
    {"module": "BAYES", "operation": "gp_predict", "params": {"kernel": "rbf"}},
])

result = orchestrator.execute(pipeline)
```

### Spatial-Temporal-Bayesian Pipeline

Combine spatial indexing, temporal aggregation, and Bayesian prediction for
environmental monitoring.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-SPACE, GEO-INFER-TIME, GEO-INFER-BAYES, GEO-INFER-DATA
- **Estimated Runtime**: 10-20 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/integration/stb_pipeline/`

### Active Inference Urban Planning

Use Active Inference agents to model urban development decisions, combining
spatial analysis, economic modeling, and risk assessment.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-ACT, GEO-INFER-CIV, GEO-INFER-ECON, GEO-INFER-RISK, GEO-INFER-SPACE
- **Estimated Runtime**: 15-30 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/integration/urban_planning/`

---

## Real World Applications

Production-ready patterns demonstrating how GEO-INFER modules combine to solve
real problems.

### Conservation Area Prioritization

Prioritize conservation areas using biodiversity data, connectivity analysis,
and multi-criteria optimization.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-BIO, GEO-INFER-SPACE, GEO-INFER-FOREST, GEO-INFER-ACT
- **Estimated Runtime**: 10-30 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/applications/conservation/`

### Supply Chain Logistics

Optimize supply chain routing using transport network analysis, risk modeling,
and demand forecasting.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-LOG, GEO-INFER-TRANSPORT, GEO-INFER-RISK, GEO-INFER-ECON
- **Estimated Runtime**: 10-20 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/applications/supply_chain/`

### Emergency Response Planning

Model emergency response scenarios including evacuation routing, resource
allocation, and population exposure.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-EMERGENCY, GEO-INFER-TRANSPORT, GEO-INFER-RISK, GEO-INFER-SPACE
- **Estimated Runtime**: 10-20 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/applications/emergency/`

### Marine Ecosystem Monitoring

Monitor marine ecosystem health using satellite ocean color data, species
distribution models, and temporal trend analysis.

- **Difficulty**: Advanced
- **Modules**: GEO-INFER-MARINE, GEO-INFER-BIO, GEO-INFER-TIME, GEO-INFER-SPACE
- **Estimated Runtime**: 10-30 minutes
- **Directory**: `GEO-INFER-EXAMPLES/examples/applications/marine/`

---

## Contributing Examples

To add an example to this gallery:

1. Create your example in `GEO-INFER-EXAMPLES/examples/<category>/`.
2. Include a README in the example directory explaining prerequisites,
   data requirements, and expected outputs.
3. Verify the example runs with a clean install of the required modules.
4. Add an entry to this gallery with the required metadata (difficulty,
   modules, runtime, directory path).
5. Submit a pull request.

### Example Directory Structure

```text
GEO-INFER-EXAMPLES/examples/<category>/<example_name>/
    README.md           # Prerequisites and description
    example.py          # Main script
    example.ipynb       # Jupyter notebook (optional)
    data/               # Sample data if needed (keep small)
    expected_output/    # Reference output for validation
```

## Related Documentation

- [Installation Guide](installation.md) -- setting up modules
- [Active Inference Guide](active_inference_guide.md) -- mathematical foundations
- [Data Dictionary](data_dictionary.md) -- data format conventions
- [Overview](overview.md) -- module descriptions and architecture
