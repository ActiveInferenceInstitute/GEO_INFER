"""Tests for wind energy analysis module."""

import numpy as np
import pytest

from geo_infer_energy.core.wind_analysis import WindAnalyzer


@pytest.fixture
def analyzer():
    return WindAnalyzer()


class TestWeibullFit:
    def test_fit_parameters(self, analyzer):
        np.random.seed(42)
        speeds = np.random.weibull(2.0, 1000) * 7.0
        result = analyzer.fit_weibull(speeds)
        assert result["shape_k"] > 1.0
        assert result["scale_c"] > 0.0
        assert result["mean_speed"] > 0.0

    def test_zero_speeds(self, analyzer):
        speeds = np.zeros(10)
        result = analyzer.fit_weibull(speeds)
        assert result["scale_c"] == 0.0

    def test_uniform_speed(self, analyzer):
        speeds = np.full(100, 8.0)
        result = analyzer.fit_weibull(speeds)
        assert abs(result["mean_speed"] - 8.0) < 0.1

    def test_degenerate_fits_expose_std_speed(self, analyzer):
        """Early-return paths include std_speed so callers never KeyError (GS-181)."""
        zero = analyzer.fit_weibull(np.zeros(5))
        assert zero["std_speed"] == 0.0
        uniform = analyzer.fit_weibull(np.full(100, 8.0))
        assert uniform["std_speed"] == pytest.approx(0.0)


class TestWeibullPDF:
    def test_pdf_positive(self, analyzer):
        speeds = np.linspace(0.1, 25, 100)
        pdf = analyzer.weibull_pdf(speeds, k=2.0, c=7.0)
        assert np.all(pdf >= 0)

    def test_pdf_integrates_approximately_to_one(self, analyzer):
        speeds = np.linspace(0.01, 30, 1000)
        pdf = analyzer.weibull_pdf(speeds, k=2.0, c=7.0)
        integral = float(np.sum(np.diff(speeds) * (pdf[1:] + pdf[:-1]) / 2.0))
        assert abs(integral - 1.0) < 0.01

    def test_zero_speed(self, analyzer):
        pdf = analyzer.weibull_pdf(np.array([0.0]), k=2.0, c=7.0)
        assert pdf[0] == 0.0


class TestWindPowerDensity:
    def test_cubic_relationship(self, analyzer):
        speed_1 = np.array([5.0])
        speed_2 = np.array([10.0])
        wpd_1 = analyzer.wind_power_density(speed_1)
        wpd_2 = analyzer.wind_power_density(speed_2)
        ratio = float(np.asarray(wpd_2 / wpd_1).reshape(-1)[0])
        assert abs(ratio - 8.0) < 0.01  # (10/5)^3 = 8

    def test_positive_values(self, analyzer):
        speeds = np.array([3.0, 7.0, 12.0])
        wpd = analyzer.wind_power_density(speeds)
        assert np.all(wpd > 0)


class TestWindExtrapolation:
    def test_higher_speed_at_greater_height(self, analyzer):
        speed_50m = analyzer.extrapolate_wind_speed(7.0, 10.0, 50.0)
        assert speed_50m > 7.0

    def test_same_height_same_speed(self, analyzer):
        speed = analyzer.extrapolate_wind_speed(7.0, 50.0, 50.0)
        assert abs(speed - 7.0) < 0.01

    def test_lower_height_lower_speed(self, analyzer):
        speed_5m = analyzer.extrapolate_wind_speed(7.0, 10.0, 5.0)
        assert speed_5m < 7.0

    def test_roughness_effect(self, analyzer):
        # With log wind profile, higher roughness gives steeper shear,
        # so extrapolating UP from same ref-height speed yields higher target speed.
        speed_smooth = analyzer.extrapolate_wind_speed(
            7.0, 10.0, 80.0, roughness_length=0.001
        )
        speed_rough = analyzer.extrapolate_wind_speed(
            7.0, 10.0, 80.0, roughness_length=0.5
        )
        assert speed_rough > speed_smooth


class TestTurbinePowerCurve:
    def test_zero_below_cutin(self, analyzer):
        speeds = np.array([0.0, 1.0, 2.0])
        power = analyzer.turbine_power_curve(speeds)
        np.testing.assert_array_equal(power, 0.0)

    def test_rated_power_in_range(self, analyzer):
        speeds = np.array([14.0, 20.0])
        power = analyzer.turbine_power_curve(speeds, rated_power_kw=2000.0)
        np.testing.assert_array_equal(power, 2000.0)

    def test_zero_above_cutout(self, analyzer):
        speeds = np.array([26.0, 30.0])
        power = analyzer.turbine_power_curve(speeds)
        np.testing.assert_array_equal(power, 0.0)

    def test_ramp_region(self, analyzer):
        speeds = np.array([5.0, 8.0])
        power = analyzer.turbine_power_curve(speeds, rated_power_kw=1000.0)
        assert power[0] < power[1]
        assert power[0] > 0
        assert power[1] < 1000.0


class TestAEP:
    def test_positive_aep(self, analyzer):
        result = analyzer.annual_energy_production(k=2.0, c=8.0)
        assert result["aep_kwh"] > 0
        assert result["capacity_factor"] > 0
        assert result["capacity_factor"] < 1.0

    def test_higher_wind_higher_aep(self, analyzer):
        low = analyzer.annual_energy_production(k=2.0, c=5.0)
        high = analyzer.annual_energy_production(k=2.0, c=9.0)
        assert high["aep_kwh"] > low["aep_kwh"]

    def test_availability_effect(self, analyzer):
        full = analyzer.annual_energy_production(k=2.0, c=8.0, availability=1.0)
        half = analyzer.annual_energy_production(k=2.0, c=8.0, availability=0.5)
        assert abs(full["aep_kwh"] / half["aep_kwh"] - 2.0) < 0.05
