--

`GEO-INFER-HEALTH` owns the `geo_infer_health` package under `GEO-INFER-HEALTH/src/`.
GEO-INFER-HEALTH: Geospatial Applications for Public Health, Epidemiology, and Healthcare Accessibility

## Public import surface

These names are exported by the current owning package:

- `geo_infer_health.DiseaseHotspotAnalyzer`
- `geo_infer_health.HealthcareAccessibilityAnalyzer`
- `geo_infer_health.EnvironmentalHealthAnalyzer`
- `geo_infer_health.Location`
- `geo_infer_health.HealthFacility`
- `geo_infer_health.DiseaseReport`
- `geo_infer_health.PopulationData`
- `geo_infer_health.EnvironmentalData`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_health
from geo_infer_health import DiseaseHotspotAnalyzer, HealthcareAccessibilityAnalyzer, EnvironmentalHealthAnalyzer, Location
assert all(value is not None for value in (DiseaseHotspotAnalyzer, HealthcareAccessibilityAnalyzer, EnvironmentalHealthAnalyzer, Location,))
assert geo_infer_health.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Healthcare query contracts

`find_facilities_in_radius` takes a radius in kilometers and returns facilities
in ascending distance order. Equal distances preserve input order. Type and
service filters select eligible facilities before distance calculation; each
eligible facility is measured once, including candidates outside the radius.

`calculate_facility_to_population_ratio` counts the caller-supplied facilities
for the selected population area. Supply facilities for the intended region;
the method does not clip them to an area geometry. Type filtering applies even
when the population is zero: the result retains an infinite ratio and reports
the filtered facility count. An unknown population area returns `None`.

`haversine_distance` interprets latitude and longitude as degrees on a sphere
with a 6,371 km radius; it does not transform CRS metadata or compute an
ellipsoidal geodesic. Valid antipodal coordinates remain in the real numerical
domain despite floating-point roundoff.

These contracts are exercised in
[healthcare accessibility tests](../../../GEO-INFER-HEALTH/tests/unit/test_healthcare_accessibility.py)
and [geospatial utility tests](../../../GEO-INFER-HEALTH/tests/unit/test_geospatial_utils.py).

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module HEALTH --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-HEALTH/src/geo_infer_health/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-HEALTH/README.md)
- [Module operating contract](../../../GEO-INFER-HEALTH/AGENTS.md)
- [Regression: test_full_workflow.py](../../../GEO-INFER-HEALTH/tests/integration/test_full_workflow.py)
- [Regression: test_disease_surveillance_integration.py](../../../GEO-INFER-HEALTH/tests/test_disease_surveillance_integration.py)
- [Regression: test_active_inference_failure_flags.py](../../../GEO-INFER-HEALTH/tests/unit/test_active_inference_failure_flags.py)
- [Example source: advanced_health_analysis.py](../../../GEO-INFER-HEALTH/examples/advanced_health_analysis.py)
- [Example source: example_disease_surveillance.py](../../../GEO-INFER-HEALTH/examples/example_disease_surveillance.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
