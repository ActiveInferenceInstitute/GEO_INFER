# Agricultural Applications

Spatial field analysis combines measurement support, crop/soil units, sampling
time, and management boundaries. H3 is an index; a cell can cross a field boundary
and its center membership does not establish polygon coverage or crop area.

## Align synthetic soil-moisture readings

This bounded fixture represents two adjacent cells and two instants. Moisture is
expressed as a declared fraction. A zero and a missing reading remain distinct.

```python
import h3
import numpy as np
import pandas as pd
from geo_infer_space import H3StateSpace, align_h3_observations

center = h3.latlng_to_cell(41.75, -124.2, 8)
neighbor = sorted(set(h3.grid_disk(center, 1)) - {center})[0]
space = H3StateSpace([neighbor, center])
times = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]
records = pd.DataFrame([
    {"cell": center, "timestamp": "2023-12-31T16:00:00-08:00", "value": 0.2},
    {"cell": neighbor, "timestamp": times[0], "value": 0.0},
    {"cell": center, "timestamp": times[1], "value": 0.4},
])
series = align_h3_observations(records, state_space=space, timestamps=times)
assert list(series.data.columns) == [neighbor, center]
assert series.data.iloc[0, 0] == 0.0
assert np.isnan(series.data.iloc[1, 0])
assert series.duration.total_seconds() == 3600
```

An actual field pipeline must specify how multiple sensors, satellite pixels, or
partial cell overlaps aggregate to a cell observation. NDVI is a reflectance ratio,
not a direct yield or soil-moisture measurement; its denominator, cloud mask,
calibration, and temporal coverage need separate validation. Keep harvest/yield
support and field area consistent before comparing seasons.

Model interpolation with [BAYES](../modules/geo-infer-bayes.md) using declared
spatial and temporal scales, and evaluate on withheld sites or seasons. SPACE's
`interpolate_spatiotemporal` is softened inverse-distance interpolation; it does
not estimate a kriging posterior or uncertainty. Decisions about irrigation and
fertilization require calibrated domain evidence beyond this alignment example.
See [agricultural intelligence](agricultural_intelligence.md).
