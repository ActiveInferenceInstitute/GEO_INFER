---
name: geo-infer-math
description: Spatial statistics, topology, and graph theory for geospatial analysis. Use when computing Moran's I, spatial autocorrelation, geodesic distances, graph connectivity, kernel density estimation, or any mathematical operation on geographic data.
prerequisites:
  required: []
  recommended: []
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-MATH

## Instructions

Foundation module with zero internal dependencies. Provides mathematical primitives consumed by all other modules.

### Core Capabilities

- **Spatial statistics**: Moran's I, Geary's C, Getis-Ord G*, LISA, semivariograms
- **Interpolation**: IDW, kriging, RBF spatial interpolation
- **Graph theory**: Network analysis, shortest paths, centrality measures
- **Kernel density**: Gaussian KDE (scipy) for distribution comparison and divergence measures
- **Distance metrics**: Haversine, Vincenty, geodesic on WGS84 ellipsoid

### Key Imports

```python
# Spatial statistics
from geo_infer_math.core.spatial_statistics import (
    MoranI, SpatialDescriptiveStats,
    getis_ord_g, ripley_k, semivariogram,
    spatial_descriptive_statistics, spatial_entropy,
    local_indicators_spatial_association,
)
# Interpolation
from geo_infer_math.core.interpolation import (
    InterpolationConfig, SpatialInterpolator,
    IDWInterpolator, KrigingInterpolator, RBFInterpolator,
    InterpolationManager, create_interpolation_manager,
    interpolate_spatial_data, create_interpolation_grid,
)
# Optimization
from geo_infer_math.core.optimization import (
    OptimizationConfig, Optimizer,
    GradientDescentOptimizer, GeneticAlgorithmOptimizer,
    ScipyOptimizer, MultiObjectiveOptimizer,
    OptimizationManager, create_optimization_manager,
    optimize_function, compare_optimization_methods,
)
# Geometry / distance
from geo_infer_math.core.geometry import (
    Point, LineString, Polygon,
    haversine_distance, vincenty_distance,
    bearing, destination_point,
    point_in_polygon, buffer_point,
    line_intersection, polygon_area_spherical,
)
```

## Examples

```python
import numpy as np
from geo_infer_math.core.spatial_statistics import (
    MoranI, getis_ord_g, ripley_k, semivariogram,
    spatial_descriptive_statistics,
)

# Moran's I spatial autocorrelation
values = np.random.randn(100)
weights = np.random.rand(100, 100)
moran = MoranI(weights)
result = moran.compute(values)
print(f"Moran's I: {result['I']}, p-value: {result['p_value']}")

# Getis-Ord G* hot-spot statistic
g_stat = getis_ord_g(values, weights)
print(f"Getis-Ord G: {g_stat}")

# Ripley's K function
coords = np.random.rand(50, 2)
k_values = ripley_k(coords, distances=np.linspace(0.01, 1.0, 20))
print(f"Ripley K at first distance: {k_values[0]:.4f}")

# Semivariogram
semivar = semivariogram(values, coords)
print(f"Semivariogram computed: {len(semivar)} lag bins")

# Descriptive spatial statistics
stats = spatial_descriptive_statistics(values, coords)
print(stats)
```

## Guidelines

- All distance calculations default to WGS84 ellipsoid
- Weight matrices should be row-standardized for spatial statistics
- Core deps are numpy/scipy/pandas/psutil/scikit-learn/sympy (see pyproject.toml); no external geo-dependencies (no GDAL/h3/shapely required). Optional: z3-solver for theorem proving, matplotlib for examples (`examples` extra).
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH`

### Integrations

- **BAYES** → Spatial statistics feeding Bayesian priors
- **SPACE** → H3 spatial weights for autocorrelation
- **SPM** → Statistical parametric map computation
- **AI** → Spatial feature engineering for ML
- **EDU** → Spatial statistics teaching exercises

### Memory and accelerator boundaries

Import `MemoryConstraintError` from `geo_infer_math.utils` for module memory-budget
failures. Python's builtin `MemoryError` is distinct. MATH imports defer GPU runtime
initialization; explicit accelerator operations perform capability probing.
