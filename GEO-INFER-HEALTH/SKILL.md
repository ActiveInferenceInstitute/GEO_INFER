---
name: geo-infer-health
description: Spatial epidemiology and public health analysis. Use when modeling disease spread, analyzing health disparities, performing spatial health risk assessment, building epidemiological surveillance systems, or assessing healthcare accessibility.
prerequisites:
  recommended:
    - geo-infer-space
    - geo-infer-data
difficulty: intermediate
estimated_time: 45min
examples_dir: ../examples/
---

# GEO-INFER-HEALTH

## Instructions

### Core Capabilities

- **Disease surveillance**: hotspot detection (DBSCAN-style local density), local incidence rates, temporal trend analysis
- **Active Inference surveillance**: belief-state updating, free-energy driven anomaly detection, and risk prediction (`ActiveInferenceDiseaseAnalyzer`)
- **Healthcare accessibility**: facility search by radius, nearest-facility lookup, facility-to-population ratios
- **Environmental health**: exposure averaging over space and time windows anchored to the latest reading
- **Geospatial utilities**: UTM projection, spatial clustering, Moran's I autocorrelation, Voronoi regions, local hotspot statistics

### Key Imports

```python
from geo_infer_health.core import (
    DiseaseHotspotAnalyzer,
    ActiveInferenceDiseaseAnalyzer,
    HealthcareAccessibilityAnalyzer,
    EnvironmentalHealthAnalyzer,
)
from geo_infer_health.models import (
    Location,
    DiseaseReport,
    PopulationData,
    EnvironmentalData,
)
from geo_infer_health.utils.advanced_geospatial import (
    spatial_clustering,
    calculate_spatial_autocorrelation,
    calculate_hotspot_statistics,
)
```

## Examples

```python
from datetime import datetime, timezone
from geo_infer_health.core import DiseaseHotspotAnalyzer
from geo_infer_health.models import DiseaseReport, Location

reports = [
    DiseaseReport(
        report_id="case001",
        disease_code="FLU",
        location=Location(latitude=34.05, longitude=-118.24),
        report_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
        case_count=5,
        source="Hospital A",
    ),
    DiseaseReport(
        report_id="case002",
        disease_code="FLU",
        location=Location(latitude=34.06, longitude=-118.25),
        report_date=datetime(2024, 1, 11, tzinfo=timezone.utc),
        case_count=3,
        source="Clinic B",
    ),
]

analyzer = DiseaseHotspotAnalyzer(reports=reports)
hotspots = analyzer.identify_simple_hotspots(threshold_case_count=2, scan_radius_km=2.0)
rate, cases, population, population_estimated = analyzer.calculate_local_incidence_rate(
    center_loc=Location(latitude=34.05, longitude=-118.24),
    radius_km=5.0,
)
```

```python
from geo_infer_health.core import ActiveInferenceDiseaseAnalyzer

# Enhanced surveillance with belief updating and risk prediction
enhanced = ActiveInferenceDiseaseAnalyzer(reports=reports)
result = enhanced.analyze_with_active_inference(time_window_days=30)
print(result["overall_risk"])
```

### API server

```bash
geo-infer-health serve --host 0.0.0.0 --port 8000
# Routes are mounted under /api/v1: /surveillance, /environment, /accessibility
```

## Guidelines

- All analytics run on plain Pydantic models — no external geospatial services required.
- Supply facilities for the intended population region before calculating ratios;
  the analyzer applies type filters but does not clip to area geometry. A zero
  population retains an infinite ratio and reports the filtered facility count.
- Radius queries use kilometers, filter by type and services, and preserve input
  order for equal distances. Haversine utilities use degree-valued coordinates
  on a 6,371 km sphere and do not transform CRS metadata.
- `calculate_local_incidence_rate` returns raw case counts in the rate slot (flagged via
  `population_estimated=False`) when no population data is available.

### Integrations

None: this module has no runtime imports of other GEO-INFER modules.

Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module HEALTH`
