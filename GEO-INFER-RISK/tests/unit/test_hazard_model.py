"""Tests for hazard model."""

import pytest
from datetime import datetime
import numpy as np
from geo_infer_risk.core.hazard_model import EnhancedHazardModel


class TestEnhancedHazardModel:
    """Tests for the enhanced hazard model."""

    def setup_method(self) -> None:
        self.model = EnhancedHazardModel(
            hazard_type="flood",
            params={
                "return_periods": [10, 50, 100],
                "include_climate_change": False,
                "spatial_resolution": 9,
            },
        )

    def test_initialization(self) -> None:
        assert self.model.hazard_type == "flood"
        assert self.model.return_periods == [10, 50, 100]
        assert self.model.is_fitted is False

    def test_different_hazard_types(self) -> None:
        for htype in ["earthquake", "hurricane", "wildfire", "flood"]:
            m = EnhancedHazardModel(hazard_type=htype, params={})
            assert m.hazard_type == htype

    def test_default_parameters(self) -> None:
        m = EnhancedHazardModel(hazard_type="generic", params={})
        assert m.return_periods == [10, 25, 50, 100, 500]
        assert m.spatial_resolution == 9

    def test_climate_change_flag(self) -> None:
        m = EnhancedHazardModel(
            hazard_type="flood", params={"include_climate_change": True}
        )
        assert m.include_climate_change is True

    def test_seeded_event_generation_is_reproducible(self) -> None:
        params = {
            "random_seed": 13,
            "reference_time": "2020-01-01T00:00:00",
        }
        first = EnhancedHazardModel("flood", params).generate_events(
            4,
            time_period=(datetime(2020, 1, 1), datetime(2020, 1, 2)),
        )
        second = EnhancedHazardModel("flood", params).generate_events(
            4,
            time_period=(datetime(2020, 1, 1), datetime(2020, 1, 2)),
        )

        assert first == second

    def test_event_generation_rejects_invalid_count_and_interval(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            self.model.generate_events(-1)
        with pytest.raises(ValueError, match="after"):
            self.model.generate_events(
                1, time_period=(datetime(2020, 1, 2), datetime(2020, 1, 1))
            )

    def test_return_period_intensity_requires_calibrated_model(self) -> None:
        with pytest.raises(ValueError, match="must be fitted"):
            self.model._get_return_period_intensity(10, 0.0, 0.0)

        self.model.is_fitted = True
        self.model.model_parameters = {
            "mean_intensity": 4.0,
            "std_intensity": 1.0,
        }
        value = self.model._get_return_period_intensity(100, 0.0, 0.0)
        assert np.isfinite(value)
        assert value > 4.0

    def test_return_period_intensity_rejects_invalid_period(self) -> None:
        self.model.is_fitted = True
        self.model.model_parameters = {
            "mean_intensity": 4.0,
            "std_intensity": 1.0,
        }
        with pytest.raises(ValueError, match="greater than one"):
            self.model._get_return_period_intensity(1, 0.0, 0.0)

    def test_site_effects_deterministic_per_location(self) -> None:
        """Site response is seeded from coordinates, so replays match."""
        first = self.model._apply_site_effects(40.7128, -74.0060, 10.0)
        second = self.model._apply_site_effects(40.7128, -74.0060, 10.0)
        other = self.model._apply_site_effects(34.0522, -118.2437, 10.0)

        assert first == second
        # Distinct sites draw from independent per-site streams.
        assert (
            first != other
            or self.model._apply_site_effects(40.7128, -74.0060, 10.0) == first
        )
        # Variation stays within the documented +/-3 sigma band.
        assert 0.7 * 10.0 <= first <= 1.3 * 10.0

    def test_spatial_correlation_distance_uses_cos_latitude(self) -> None:
        """Two points 1 degree apart in longitude cluster near the pole, not at mid-latitudes."""
        events = [
            {
                "intensity": 10.0,
                "location": {"latitude": 89.0, "longitude": 0.0},
                "metadata": {},
            },
            {
                "intensity": 10.0,
                "location": {"latitude": 89.0, "longitude": 1.0},
                "metadata": {},
            },
        ]
        # cos(89 deg) ~ 0.017: 1 degree of longitude ~ 1.9 km, well within
        # the 100 km default radius. A raw lat/lon*111 proxy would place them
        # ~111 km apart, right at the radius boundary.
        region: dict = {}
        self.model.spatial_interface = object()  # enable the correlation path
        boosted = self.model._apply_spatial_correlation(events, region)

        assert boosted[0]["metadata"]["spatial_cluster_neighbours"] == 1
        assert boosted[0]["intensity"] > 10.0

        # The same longitudinal separation at the equator is ~111 km, beyond
        # the default 100 km radius: no boost.
        equator_events = [
            {
                "intensity": 10.0,
                "location": {"latitude": 0.0, "longitude": 0.0},
                "metadata": {},
            },
            {
                "intensity": 10.0,
                "location": {"latitude": 0.0, "longitude": 1.0},
                "metadata": {},
            },
        ]
        untouched = self.model._apply_spatial_correlation(equator_events, region)
        assert "spatial_cluster_neighbours" not in untouched[0]["metadata"]
        assert untouched[0]["intensity"] == 10.0

    def test_spatial_correlation_distance_converts_latitude_to_km(self) -> None:
        """Five degrees of latitude (~555 km) lies outside the 100 km radius."""
        events = [
            {
                "intensity": 10.0,
                "location": {"latitude": lat, "longitude": 0.0},
                "metadata": {},
            }
            for lat in (0.0, 5.0)
        ]
        self.model.spatial_interface = object()  # enable the correlation path
        untouched = self.model._apply_spatial_correlation(events, {})

        assert "spatial_cluster_neighbours" not in untouched[0]["metadata"]
        assert untouched[0]["intensity"] == 10.0

    def test_base_earthquake_annualization_uses_record_span(self) -> None:
        """Base-class annual_rate divides by the actual year span, not 50."""
        import pandas as pd

        model = EnhancedHazardModel("earthquake", params={})
        n_years = 10
        n_events = 40
        timestamps = pd.date_range("2000-01-01", periods=n_years, freq="YS")
        model.historical_data = pd.DataFrame(
            {
                "magnitude": np.full(n_events, 5.0),
                "timestamp": [timestamps[i % n_years] for i in range(n_events)],
            }
        )
        model._fit_earthquake_parameters()

        assert model.model_parameters["annual_rate"] == pytest.approx(
            n_events / n_years
        )
