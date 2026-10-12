# GEO-INFER Cookbook

These small recipes use maintained package interfaces with explicit inputs.
Run each Python block independently from the locked workspace environment.
The fixtures are synthetic and contain assertions for their expected behavior.

## H3/UTC observations with explicit missingness

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
    {"cell": center, "timestamp": "2023-12-31T16:00:00-08:00", "value": 2.0},
    {"cell": neighbor, "timestamp": times[0], "value": 0.0},
    {"cell": center, "timestamp": times[1], "value": 4.0},
])
series = align_h3_observations(records, state_space=space, timestamps=times)
assert list(series.data.columns) == [neighbor, center]
assert series.data.iloc[0, 0] == 0.0
assert np.isnan(series.data.iloc[1, 0])
assert series.duration.total_seconds() == 3600
```

## A UTC CSV round trip

TIME IO uses the same strict timestamp rules as construction. Use a named time
index and preserve timezone and nanosecond precision through the write/read path.
Temporary files keep the example self-contained.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from geo_infer_time import TimeSeries
from geo_infer_time.io import TimeSeriesReader, TimeSeriesWriter

source = TimeSeries(pd.DataFrame({"value": [2.0, 4.0]}, index=pd.DatetimeIndex([
    "2024-01-01T00:00:00.000000001Z", "2024-01-01T00:00:00.000000009Z"
], name="timestamp")))
with TemporaryDirectory() as directory:
    path = Path(directory) / "observations.csv"
    TimeSeriesWriter().write(source, path)
    restored = TimeSeriesReader().read(path)
    pd.testing.assert_frame_equal(restored.data, source.data)
assert restored.duration == pd.Timedelta(nanoseconds=8)
```
## Choose an analytical contract

Use [TIME](../modules/geo-infer-time.md) for a declared temporal model,
[SPACE](../modules/geo-infer-space.md) for real H3 interfaces,
[BAYES](../modules/geo-infer-bayes.md) for probabilistic models, and
[ACT](../modules/geo-infer-act.md) for a specified perception/action loop.
Domain recipes cover [agriculture](../examples/agricultural_applications.md),
[climate](../examples/climate_modeling.md), and [urban analysis](../examples/urban_analytics.md).
A transport, scheduler, visualization, or API deployment has a separate acceptance
surface; follow its owning package guide instead of assuming a common facade.
