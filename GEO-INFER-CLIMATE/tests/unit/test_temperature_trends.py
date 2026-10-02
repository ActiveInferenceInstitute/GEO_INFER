"""Tests for temperature trend analysis module."""

import numpy as np
import pytest


from geo_infer_climate.core.temperature_trends import TemperatureTrendAnalyzer


@pytest.fixture
def analyzer():
    return TemperatureTrendAnalyzer()


class TestLinearTrend:
    def test_positive_trend(self, analyzer):
        data = np.linspace(10, 15, 50)
        result = analyzer.linear_trend(data)
        assert result["slope"] > 0
        assert result["r_squared"] > 0.9

    def test_negative_trend(self, analyzer):
        data = np.linspace(20, 15, 50)
        result = analyzer.linear_trend(data)
        assert result["slope"] < 0

    def test_no_trend(self, analyzer):
        data = np.full(50, 15.0)
        result = analyzer.linear_trend(data)
        assert abs(result["slope"]) < 1e-10

    def test_with_years(self, analyzer):
        years = np.arange(1970, 2020)
        data = np.linspace(14.0, 15.5, 50)
        result = analyzer.linear_trend(data, years=years)
        assert result["slope_per_decade"] > 0

    def test_insufficient_data(self, analyzer):
        result = analyzer.linear_trend(np.array([15.0]))
        assert result["p_value"] == 1.0


class TestMannKendall:
    def test_increasing_trend(self, analyzer):
        data = np.linspace(10, 20, 100) + np.random.normal(0, 0.5, 100)
        result = analyzer.mann_kendall_test(data)
        assert result["trend"] == "increasing"
        assert result["s_statistic"] > 0
        assert result["significant"]

    def test_decreasing_trend(self, analyzer):
        data = np.linspace(20, 10, 100) + np.random.normal(0, 0.5, 100)
        result = analyzer.mann_kendall_test(data)
        assert result["trend"] == "decreasing"
        assert result["s_statistic"] < 0

    def test_no_trend_random(self, analyzer):
        np.random.seed(42)
        data = np.random.normal(15.0, 0.1, 20)
        result = analyzer.mann_kendall_test(data)
        assert result["n_observations"] == 20

    def test_short_series(self, analyzer):
        result = analyzer.mann_kendall_test(np.array([1.0, 2.0, 3.0]))
        assert result["n_observations"] == 3
        assert result["trend"] == "no trend"


class TestSensSlope:
    def test_positive_slope(self, analyzer):
        data = np.linspace(10, 15, 30)
        result = analyzer.sens_slope(data)
        assert result["median_slope"] > 0
        assert result["slope_per_decade"] > 0

    def test_slope_robust_to_outliers(self, analyzer):
        data = np.linspace(10, 15, 30)
        data_outlier = data.copy()
        data_outlier[15] = 100.0  # Add outlier
        result_clean = analyzer.sens_slope(data)
        result_outlier = analyzer.sens_slope(data_outlier)
        assert abs(result_clean["median_slope"] - result_outlier["median_slope"]) < 1.0


class TestChangepoint:
    def test_detects_shift(self, analyzer):
        data = np.concatenate([np.full(25, 14.0), np.full(25, 16.0)])
        result = analyzer.detect_changepoint(data)
        assert result["magnitude"] > 1.0
        assert result["changepoint_index"] > 0

    def test_no_shift(self, analyzer):
        data = np.full(50, 15.0)
        result = analyzer.detect_changepoint(data)
        assert abs(result["magnitude"]) < 1e-10


class TestHeatIsland:
    def test_positive_uhi(self, analyzer):
        urban = np.full(30, 22.0)
        rural = np.full(30, 19.0)
        result = analyzer.calculate_heat_island_effect(urban, rural)
        assert result["mean_uhi_c"] == pytest.approx(3.0, abs=0.01)

    def test_zero_uhi(self, analyzer):
        temps = np.full(30, 20.0)
        result = analyzer.calculate_heat_island_effect(temps, temps)
        assert result["mean_uhi_c"] == pytest.approx(0.0, abs=1e-10)

    def test_unequal_lengths(self, analyzer):
        urban = np.full(20, 22.0)
        rural = np.full(30, 19.0)
        result = analyzer.calculate_heat_island_effect(urban, rural)
        assert result["n_observations"] == 20


class TestVectorizedStatistics:
    """Pin the vectorized Mann-Kendall / Sen's slope behavior (GS-151)."""

    def test_mann_kendall_s_matches_brute_force_with_ties_and_nan(self, analyzer):
        rng = np.random.default_rng(7)
        data = np.round(rng.normal(0, 1, 120))  # heavy ties
        data[17] = np.nan
        clean = data[~np.isnan(data)]
        expected = sum(
            1 if clean[j] > clean[i] else -1 if clean[j] < clean[i] else 0
            for i in range(len(clean) - 1)
            for j in range(i + 1, len(clean))
        )
        result = analyzer.mann_kendall_test(data)
        assert result["s_statistic"] == expected

    def test_mann_kendall_exact_s_values(self, analyzer):
        assert analyzer.mann_kendall_test(np.arange(10.0))["s_statistic"] == 45
        assert (
            analyzer.mann_kendall_test(np.arange(10.0)[::-1].copy())["s_statistic"]
            == -45
        )

    def test_sen_slope_matches_brute_force(self, analyzer):
        rng = np.random.default_rng(11)
        data = rng.normal(0, 1, 90)
        data[40] = np.nan
        clean = data[~np.isnan(data)]
        n = len(clean)
        expected = np.sort(
            [(clean[j] - clean[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
        )
        result = analyzer.sens_slope(data)
        ns = len(expected)
        assert result["n_slopes"] == ns
        assert result["median_slope"] == pytest.approx(float(np.median(expected)))
        c_alpha = 1.96 * np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
        m1 = max(0, int((ns - c_alpha) / 2))
        m2 = min(ns - 1, int((ns + c_alpha) / 2))
        assert result["lower_ci"] == pytest.approx(float(expected[m1]))
        assert result["upper_ci"] == pytest.approx(float(expected[m2]))

    def test_sen_slope_median_even_count_semantics(self, analyzer):
        # 4 slopes: (2-1)/1, (3-1)/2, (3-2)/1, (4-1)/3, (4-2)/2, (4-3)/1 for 1..4
        result = analyzer.sens_slope(np.array([1.0, 2.0, 3.0, 4.0]))
        slopes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert result["median_slope"] == pytest.approx(float(np.median(slopes)))
