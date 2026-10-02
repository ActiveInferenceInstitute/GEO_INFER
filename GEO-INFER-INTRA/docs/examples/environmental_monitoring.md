# Environmental Monitoring

This local example summarizes a synthetic hourly temperature series through TIME's
real model. Measurements need explicit units, locations, and source/calibration
provenance in a field application; the numbers below are test inputs, not observations.

## Verify a trend before designing alerts

A trend slope and its fit quality are different quantities. The example has a
known slope, making the expected result independently checkable.

```python
import numpy as np
import pandas as pd
from geo_infer_time import TimeSeries, TemporalAnalyzer

values = -5.0 + 0.25 * np.arange(8)
series = TimeSeries(values, timestamps=pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"))
result = TemporalAnalyzer().detect_trend(series)
assert result["trend_direction"] == "increasing"
assert np.isclose(result["slope_per_sample"], 0.25)
assert np.isclose(result["r_squared"], 1.0)
```

The slope here is 0.25 temperature units per sample, equivalent to per hour only
because the fixture has a regular hourly axis. The result does not provide a
forecast, confidence interval, or environmental risk threshold. A field workflow
must choose a baseline, account for measurement uncertainty and seasonality,
and validate alert false-positive/false-negative behavior on held-out observations.

For multiple sites, use [H3/UTC alignment](../../../GEO-INFER-SPACE/docs/CROSS_MODULE_COMPOSITION.md).
For asynchronous input, see [environmental integration](../guides/ENVIRONMENTAL_MONITORING_INTEGRATION.md).
