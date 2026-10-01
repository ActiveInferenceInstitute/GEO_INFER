"""Tests for precipitation analysis module."""

import numpy as np
import pytest


from geo_infer_climate.core.precipitation_analysis import PrecipitationAnalyzer


@pytest.fixture
def analyzer():
    return PrecipitationAnalyzer()


class TestIDFCurve:
    def test_fit_idf_single_duration(self, analyzer):
        np.random.seed(42)
        maxima_1h = np.random.gumbel(loc=30, scale=8, size=30)
        result = analyzer.fit_idf_curve({1.0: maxima_1h})
        assert 1.0 in result
        entry = result[1.0]
        assert entry["duration_hours"] == 1.0
        assert entry["n_years"] == 30
        assert entry["gumbel_beta"] > 0

    def test_fit_idf_multiple_durations(self, analyzer):
        np.random.seed(42)
        annual_maxima = {
            0.5: np.random.gumbel(loc=50, scale=12, size=20),
            1.0: np.random.gumbel(loc=35, scale=10, size=20),
            6.0: np.random.gumbel(loc=60, scale=15, size=20),
            24.0: np.random.gumbel(loc=80, scale=20, size=20),
        }
        result = analyzer.fit_idf_curve(annual_maxima)
        assert len(result) == 4
        for entry in result.values():
            rp_intensities = entry["return_period_intensities"]
            assert 2 in rp_intensities
            assert 100 in rp_intensities
            assert rp_intensities[100]["depth_mm"] >= rp_intensities[2]["depth_mm"]

    def test_idf_intensity_decreases_with_duration(self, analyzer):
        np.random.seed(42)
        annual_maxima = {
            1.0: np.random.gumbel(loc=30, scale=8, size=25),
            6.0: np.random.gumbel(loc=55, scale=12, size=25),
        }
        result = analyzer.fit_idf_curve(annual_maxima)
        i_1h = result[1.0]["return_period_intensities"][10]["intensity_mm_h"]
        i_6h = result[6.0]["return_period_intensities"][10]["intensity_mm_h"]
        assert i_1h > i_6h

    def test_idf_skips_insufficient_data(self, analyzer):
        result = analyzer.fit_idf_curve({1.0: np.array([10.0, 15.0])})
        assert len(result) == 0

    def test_idf_handles_nan_values(self, analyzer):
        data = np.array(
            [10.0, np.nan, 20.0, 15.0, np.nan, 25.0, 18.0, 22.0, 30.0, 12.0]
        )
        result = analyzer.fit_idf_curve({1.0: data})
        assert 1.0 in result
        assert result[1.0]["n_years"] == 8


class TestGumbelReturnPeriod:
    def test_high_value_long_return(self, analyzer):
        np.random.seed(42)
        data = np.random.gumbel(loc=50, scale=10, size=50)
        extreme = float(np.max(data)) + 30
        result = analyzer.gumbel_return_period(data, extreme)
        assert (
            result["return_period_years"] is None or result["return_period_years"] > 50
        )
        assert result["exceedance_probability"] < 0.05

    def test_median_value_short_return(self, analyzer):
        np.random.seed(42)
        data = np.random.gumbel(loc=50, scale=10, size=50)
        median_val = float(np.median(data))
        result = analyzer.gumbel_return_period(data, median_val)
        assert result["exceedance_probability"] > 0.2

    def test_insufficient_data(self, analyzer):
        result = analyzer.gumbel_return_period(np.array([10.0, 20.0]), 15.0)
        assert result["exceedance_probability"] == 1.0
        assert result["return_period_years"] == 0.0

    def test_result_fields(self, analyzer):
        np.random.seed(42)
        data = np.random.gumbel(loc=50, scale=10, size=30)
        result = analyzer.gumbel_return_period(data, 60.0)
        assert "return_period_years" in result
        assert "exceedance_probability" in result
        assert "design_value" in result
        assert "gumbel_mu" in result
        assert "gumbel_beta" in result


class TestRainfallDepthForReturnPeriod:
    def test_depth_increases_with_return_period(self, analyzer):
        np.random.seed(42)
        data = np.random.gumbel(loc=50, scale=10, size=30)
        d10 = analyzer.rainfall_depth_for_return_period(data, 10)
        d100 = analyzer.rainfall_depth_for_return_period(data, 100)
        assert d100["design_depth_mm"] > d10["design_depth_mm"]

    def test_depth_positive(self, analyzer):
        np.random.seed(42)
        data = np.random.gumbel(loc=50, scale=10, size=30)
        result = analyzer.rainfall_depth_for_return_period(data, 50)
        assert result["design_depth_mm"] > 0

    def test_exceedance_probability(self, analyzer):
        np.random.seed(42)
        data = np.random.gumbel(loc=50, scale=10, size=30)
        result = analyzer.rainfall_depth_for_return_period(data, 25)
        assert result["exceedance_probability"] == pytest.approx(1.0 / 25, rel=1e-10)

    def test_insufficient_data(self, analyzer):
        result = analyzer.rainfall_depth_for_return_period(np.array([10.0]), 50)
        assert result["design_depth_mm"] == 0.0


class TestPrecipitationStatistics:
    def test_basic_statistics(self, analyzer):
        np.random.seed(42)
        data = np.random.exponential(5, 365)
        result = analyzer.calculate_precipitation_statistics(data)
        assert result["n_days"] == 365
        assert result["total_mm"] > 0
        assert result["mean_daily_mm"] > 0
        assert result["max_daily_mm"] >= result["mean_daily_mm"]

    def test_wet_dry_day_counts(self, analyzer):
        data = np.array([0.0, 0.5, 2.0, 5.0, 0.0, 0.0, 10.0, 0.2, 3.0, 0.0])
        result = analyzer.calculate_precipitation_statistics(data)
        assert result["wet_day_count"] == 4
        assert result["dry_day_count"] == 6
        assert result["wet_day_count"] + result["dry_day_count"] == 10

    def test_consecutive_dry_days(self, analyzer):
        data = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0])
        result = analyzer.calculate_precipitation_statistics(data)
        assert result["max_consecutive_dry_days"] == 5

    def test_consecutive_wet_days(self, analyzer):
        data = np.array([0.0, 2.0, 3.0, 5.0, 1.5, 0.0, 0.0, 4.0, 2.0, 0.0])
        result = analyzer.calculate_precipitation_statistics(data)
        assert result["max_consecutive_wet_days"] == 4

    def test_percentiles(self, analyzer):
        np.random.seed(42)
        data = np.random.exponential(5, 1000)
        result = analyzer.calculate_precipitation_statistics(data)
        assert result["percentile_99_mm"] > result["percentile_95_mm"]


class TestGammaDistribution:
    def test_fit_gamma(self, analyzer):
        np.random.seed(42)
        data = np.random.gamma(shape=2.0, scale=5.0, size=200)
        result = analyzer.fit_gamma_distribution(data)
        assert result["alpha"] > 0
        assert result["beta"] > 0
        assert result["n_observations"] == 200

    def test_gamma_mean_variance(self, analyzer):
        np.random.seed(42)
        data = np.random.gamma(shape=3.0, scale=4.0, size=500)
        result = analyzer.fit_gamma_distribution(data)
        assert result["mean"] > 0
        assert result["variance"] > 0

    def test_insufficient_data(self, analyzer):
        result = analyzer.fit_gamma_distribution(np.array([5.0]))
        assert result["alpha"] == 0.0
        assert result["beta"] == 0.0

    def test_filters_zero_values(self, analyzer):
        np.random.seed(42)
        data = np.concatenate([np.zeros(50), np.random.gamma(2, 5, 100)])
        result = analyzer.fit_gamma_distribution(data)
        assert result["n_observations"] == 100
