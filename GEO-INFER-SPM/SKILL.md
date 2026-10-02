---
name: geo-infer-spm
description: Statistical Parametric Mapping for geospatial data. Use when performing GLM-based spatial analysis, random field theory corrections, cluster-level inference, or neuroimaging-style statistical mapping on geographic datasets.
difficulty: advanced
estimated_time: 60min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-SPM

## Instructions

### Core Capabilities

- **GLM fitting**: General linear models with spatial design matrices
- **Random field theory**: Multiple comparison correction for spatial data
- **Cluster inference**: Cluster-level and peak-level statistics
- **Contrast testing**: T-contrasts and F-contrasts on spatial maps
- **Visualization**: Interactive time series explorer (mean±SD + residuals)

### Key Imports

```python
from geo_infer_spm.core.glm import GeneralLinearModel
from geo_infer_spm.core.rft import RandomFieldTheory
from geo_infer_spm.models.data_models import SPMData, SPMResult
from geo_infer_spm.visualization.interactive import create_time_series_explorer
```

### Random Field Theory Inference

```python
import numpy as np
from geo_infer_spm.core.rft import RandomFieldTheory

rft = RandomFieldTheory(
    field_shape=(64, 64),
    smoothness=np.array([4.5, 4.5]),
)
rft.compute_resel_counts()

peak_height = rft.peak_threshold(0.05, stat_type="Z", two_sided=True)
cluster_p = rft.cluster_extent_p_value(
    extent=1.25,
    cluster_forming_threshold=3.09,
    stat_type="Z",
    two_sided=True,
)
```

## Examples

```python
from geo_infer_spm.models.data_models import SPMData
import numpy as np

data = SPMData(
    data=np.random.randn(100, 50),
    coordinates=np.column_stack([
        np.random.uniform(-90, 90, 100),   # latitudes
        np.random.uniform(-180, 180, 100)   # longitudes
    ])
)
```

## Guidelines

- Coordinates must be valid: latitude ∈ [-90, 90], longitude ∈ [-180, 180]
- GLM implementation is Alpha status — spatial design matrices in progress
- Time series explorer uses Plotly for interactive mean±SD visualization
- Test from the workspace root:
  `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPM`
- Native Bayesian fitting uses two chains and one sampling process by default;
  set `chains` and `cores` explicitly when allocating parallel CPU work.
- On macOS 27 with Apple Clang, use the explicit native compiler profile in
  [native sampling profiles](docs/native_sampling_profiles.md). Linux uses the
  standard compiler profile; both profiles run actual PyMC and CLinker tests.

### Integrations

- None required: SPM is fully self-contained (no `geo_infer_*` imports in src/)
  and consumes plain numpy arrays / GeoDataFrames.
- Optional extras: `bayesian` (pymc + arviz enable the full MCMC path in
  `BayesianSPM`; without them it falls back to a logged empirical-Bayes
  approximation).
