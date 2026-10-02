# GEO-INFER-CLIMATE: climate analysis

CLIMATE owns climate and temperature trend analysis under `geo_infer_climate`.
Data acquisition, coordinate alignment, units and observational provenance are
separate inputs to a climate analysis; synthetic examples cannot establish
results about a real region.

## A small numerical trend

`TemperatureTrendAnalyzer` accepts observations and a matching numeric year axis.
This analytical sequence rises by two temperature units every year. The expected
slope follows directly from the input, without a timing assumption.

```python
import math
import numpy as np
from geo_infer_climate.core.temperature_trends import TemperatureTrendAnalyzer

result = TemperatureTrendAnalyzer().linear_trend(
    np.array([10., 12., 14., 16.]),
    years=np.array([2000, 2001, 2002, 2003]),
)
assert math.isclose(result["slope"], 2., abs_tol=1e-12)
assert math.isclose(result["r_squared"], 1., abs_tol=1e-12)
```

The same class provides Mann–Kendall trend testing and Sen slope estimation. These
methods have different statistical assumptions and should be selected for the
measurement process, sampling cadence and missing-data policy. Physical-time
composition uses explicit UTC instants from TIME, while the example deliberately
uses a numeric year covariate.

## Verification and scaling

Run `uv run python GEO-INFER-TEST/run_unified_tests.py --module CLIMATE`. Small
numerical references remain in unit tests. The 10,000-observation trend workload
runs in the performance lane, records algorithm durations and checks finite
results under a process deadline. Its quadratic pairwise work is not asserted to
finish within one second on every CPU.

The module's configured data/model integrations have their own dependency and
source requirements. This tiny reference establishes arithmetic behavior, not
licensed-data acquisition, regional predictions or accelerated hardware results.
