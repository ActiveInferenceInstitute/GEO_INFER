# Climate Modeling

Climate analysis must distinguish spatial support, measurement uncertainty,
baseline climatology, sampling changes, and model scenarios. A climatology should
be fit to a declared reference period; testing on that same period cannot establish
future predictive performance.

## A synthetic temporal trend with a numerical oracle

TIME can represent and analyze a UTC series. This daily toy signal has a known
linear slope and serves as a contract check, not a climate projection.

```python
import numpy as np
import pandas as pd
from geo_infer_time import TimeSeries, TemporalAnalyzer

values = -5.0 + 0.25 * np.arange(8)
series = TimeSeries(values, timestamps=pd.date_range("2024-01-01", periods=8, tz="UTC"))
result = TemporalAnalyzer().detect_trend(series)
assert result["trend_direction"] == "increasing"
assert np.isclose(result["slope_per_sample"], 0.25)
assert np.isclose(result["r_squared"], 1.0)
```

`trend_strength` is the absolute slope per observation and `r_squared` measures
linear fit. Neither supplies a confidence interval or causal attribution. For
calendar-month data, one observation step is a month index, not a fixed number
of seconds. For physical rates, regress against declared elapsed time or regularize
onto an explicit physical grid before fitting.

For gridded climate variables, retain CRS, calendar conventions, units, missingness,
and cell support through storage and alignment. Mask unobserved pairs when building
BAYES inputs. A separable spatial/temporal kernel requires separate scale choices.
Validate extrapolation, station changes, and seasonal residuals with withheld time
blocks or sites. See [CLIMATE](../modules/geo-infer-climate.md),
[TIME](../modules/geo-infer-time.md), and [custom models](../advanced/custom_models.md).
