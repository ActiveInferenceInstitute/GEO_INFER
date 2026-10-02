"""Record large-series trend performance separately from numerical unit checks."""

import time

import numpy as np
import pytest

from geo_infer_climate.core.temperature_trends import TemperatureTrendAnalyzer


@pytest.mark.performance
@pytest.mark.timeout(60)
def test_ten_thousand_observation_trends(record_testsuite_property):
    """Run real trend algorithms and retain their measured durations."""
    analyzer = TemperatureTrendAnalyzer()
    data = np.random.default_rng(3).normal(0, 1, 10_000)
    started = time.perf_counter()
    mk = analyzer.mann_kendall_test(data)
    midpoint = time.perf_counter()
    slope = analyzer.sens_slope(data)
    record_testsuite_property("mann_kendall_seconds", midpoint - started)
    record_testsuite_property("sen_slope_seconds", time.perf_counter() - midpoint)
    assert np.isfinite(mk["s_statistic"])
    assert np.isfinite(slope["median_slope"])
    assert slope["n_slopes"] == len(data) * (len(data) - 1) // 2
