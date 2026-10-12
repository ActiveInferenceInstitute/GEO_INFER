# Urban Analytics

Population density, infrastructure exposure, and access measures use different
observation supports. Keep population counts, area denominators, timestamp,
geometry, and aggregation rules explicit before joining layers. H3 area varies
by cell; use the actual cell area when defining a density.

## A density calculation on real H3 cells

This synthetic one-instant population example uses SPACE's explicit state order
and TIME's UTC axis. Counts are assumed to cover each whole cell for this fixture.

```python
import h3
import numpy as np
import pandas as pd
from geo_infer_space import H3StateSpace, align_h3_observations

center = h3.latlng_to_cell(45.52, -122.67, 8)
neighbor = sorted(set(h3.grid_disk(center, 1)) - {center})[0]
space = H3StateSpace([neighbor, center])
time = "2024-01-01T00:00:00Z"
records = pd.DataFrame({"cell": list(space.cells), "timestamp": [time, time], "value": [100.0, 200.0]})
series = align_h3_observations(records, state_space=space, timestamps=[time])
areas = np.array([h3.cell_area(cell, unit="km^2") for cell in space.cells])
density = series.data.iloc[0].to_numpy() / areas
np.testing.assert_allclose(density * areas, [100.0, 200.0])
assert (density > 0).all()
```

Partial polygons need an explicitly justified allocation rule, such as area overlap
or independently validated dasymetric weights. Facility exposure is an intersection
or distance calculation with declared support and CRS; it is not automatically a
risk probability. For access, distinguish Euclidean distance, network travel time,
and service availability. Report disaggregated uncertainty and data coverage when
aggregating across neighborhoods.

See [urban planning](urban_planning.md), [spatial concepts](../ontology/spatial_concepts.md),
and the [H3 format guide](../geospatial/data_formats/h3/index.md).
