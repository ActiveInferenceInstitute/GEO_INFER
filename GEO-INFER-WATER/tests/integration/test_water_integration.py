"""
Integration tests for GEO-INFER-WATER: water quality assessment and watershed analysis pipeline.

Tests WaterQualityAssessor and WatershedDelineator working together in a
comprehensive water resource analysis pipeline.
"""

import pytest
import numpy as np

import xarray as xr

pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def ph_field():
    """Create synthetic pH data across a watershed."""
    np.random.seed(42)
    return xr.DataArray(
        np.random.uniform(6.0, 9.0, (10, 10)),
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(38.0, 39.0, 10),
            "lon": np.linspace(-122.0, -121.0, 10),
        },
    )


@pytest.fixture
def dissolved_oxygen_field():
    """Create synthetic dissolved oxygen data (mg/L)."""
    np.random.seed(43)
    return xr.DataArray(
        np.random.uniform(3.0, 10.0, (10, 10)),
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(38.0, 39.0, 10),
            "lon": np.linspace(-122.0, -121.0, 10),
        },
    )


@pytest.fixture
def elevation_data():
    """Create synthetic elevation data for watershed delineation."""
    np.random.seed(44)
    lat = np.linspace(38.0, 39.0, 20)
    lon = np.linspace(-122.0, -121.0, 20)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    # Create a simple bowl-shaped elevation (lower in center)
    elevation = 500 + 200 * ((lat_grid - 38.5) ** 2 + (lon_grid + 121.5) ** 2)
    return xr.DataArray(
        elevation,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )


@pytest.fixture
def water_samples():
    """Create synthetic water quality samples."""
    from geo_infer_water.core.water_quality import WaterSample

    samples = [
        WaterSample(
            sample_id="ws_001",
            location=(-121.5, 38.5),
            timestamp="2025-06-01",
            ph=7.2,
            dissolved_oxygen=8.5,
            turbidity=0.5,
            temperature=18.0,
            nitrate=3.0,
            phosphate=0.03,
            e_coli=10.0,
        ),
        WaterSample(
            sample_id="ws_002",
            location=(-121.4, 38.6),
            timestamp="2025-06-15",
            ph=6.8,
            dissolved_oxygen=6.0,
            turbidity=2.5,
            temperature=22.0,
            nitrate=8.0,
            phosphate=0.08,
            e_coli=50.0,
        ),
        WaterSample(
            sample_id="ws_003",
            location=(-121.3, 38.7),
            timestamp="2025-07-01",
            ph=7.5,
            dissolved_oxygen=7.2,
            turbidity=1.0,
            temperature=20.0,
            nitrate=5.0,
            phosphate=0.05,
            e_coli=25.0,
        ),
        WaterSample(
            sample_id="ws_004",
            location=(-121.6, 38.4),
            timestamp="2025-07-15",
            ph=8.2,
            dissolved_oxygen=9.0,
            turbidity=0.3,
            temperature=16.0,
            nitrate=1.5,
            phosphate=0.01,
            e_coli=2.0,
        ),
        WaterSample(
            sample_id="ws_005",
            location=(-121.2, 38.8),
            timestamp="2025-08-01",
            ph=5.8,  # Below EPA standard
            dissolved_oxygen=3.5,  # Below standard
            turbidity=8.0,  # Above standard
            temperature=28.0,
            nitrate=15.0,  # Above standard
            phosphate=0.15,  # Above standard
            e_coli=300.0,  # Above standard
        ),
    ]
    return samples


class TestWaterQualityAssessment:
    """Test water quality assessment from spatial data."""

    def test_basic_quality_assessment(self, ph_field, dissolved_oxygen_field):
        """Test water quality assessment from pH and DO fields."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()
        result = assessor.assess_water_quality(
            ph_field,
            dissolved_oxygen=dissolved_oxygen_field,
        )

        assert "ph_compliant" in result
        assert "do_compliant" in result
        assert "quality_index" in result

        # Quality index should be a ratio of compliance
        qi = result["quality_index"]
        assert float(qi.min()) >= 0
        assert float(qi.max()) <= 1

    def test_quality_with_all_parameters(self, ph_field, dissolved_oxygen_field):
        """Test quality assessment with all available parameters."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        np.random.seed(50)
        turbidity = xr.DataArray(
            np.random.uniform(0.1, 5.0, ph_field.shape),
            dims=ph_field.dims,
            coords=ph_field.coords,
        )
        nitrate = xr.DataArray(
            np.random.uniform(0.5, 15.0, ph_field.shape),
            dims=ph_field.dims,
            coords=ph_field.coords,
        )

        assessor = WaterQualityAssessor()
        result = assessor.assess_water_quality(
            ph_field,
            dissolved_oxygen=dissolved_oxygen_field,
            turbidity=turbidity,
            nitrate=nitrate,
        )

        assert "ph_compliant" in result
        assert "do_compliant" in result
        assert "turb_compliant" in result
        assert "nit_compliant" in result
        assert "quality_index" in result


class TestWaterQualityIndex:
    """Test WQI calculation from sample data."""

    def test_wqi_calculation(self, water_samples):
        """Test WQI calculation for individual samples."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()

        # Clean sample should get higher WQI than polluted sample
        clean_wqi = assessor.calculate_wqi(water_samples[3])  # ws_004 is cleanest
        polluted_wqi = assessor.calculate_wqi(
            water_samples[4]
        )  # ws_005 is most polluted

        assert clean_wqi["wqi"] > polluted_wqi["wqi"], (
            "Clean sample should have higher WQI than polluted sample"
        )
        assert clean_wqi["classification"] in ["Excellent", "Good", "Medium"]
        assert polluted_wqi["classification"] in ["Bad", "Very Bad", "Medium"]

    def test_wqi_sub_indices(self, water_samples):
        """Test that WQI includes component sub-indices."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()
        result = assessor.calculate_wqi(water_samples[0])

        assert "sub_indices" in result
        assert "dissolved_oxygen" in result["sub_indices"]
        assert "ph" in result["sub_indices"]
        assert "turbidity" in result["sub_indices"]
        assert "sample_id" in result
        assert result["sample_id"] == "ws_001"


class TestTrendAnalysis:
    """Test water quality trend analysis."""

    def test_ph_trend_analysis(self, water_samples):
        """Test trend analysis for pH across samples."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()
        result = assessor.analyze_trends(water_samples, "ph")

        assert result["parameter"] == "ph"
        assert result["sample_count"] == 5
        assert "mean" in result
        assert "trend_direction" in result
        assert result["trend_direction"] in ["increasing", "decreasing", "stable"]
        assert "trend_slope" in result

    def test_dissolved_oxygen_trend(self, water_samples):
        """Test trend analysis for dissolved oxygen."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()
        result = assessor.analyze_trends(water_samples, "dissolved_oxygen")

        assert result["sample_count"] == 5
        assert result["min"] >= 0
        assert "exceedance_count" in result


class TestRiskAssessment:
    """Test water quality risk assessment for different use types."""

    def test_drinking_water_risk(self, water_samples):
        """Test risk assessment for drinking water use."""
        from geo_infer_water.core.water_quality import (
            WaterQualityAssessor,
            WaterBodyType,
        )

        assessor = WaterQualityAssessor()
        result = assessor.assess_risk(
            water_samples,
            WaterBodyType.RIVER,
            usage_type="drinking",
        )

        assert result["usage_type"] == "drinking"
        assert result["water_body_type"] == "river"
        assert result["risk_level"] in ["Low", "Moderate", "High", "Critical"]
        assert 0 <= result["risk_score"] <= 1
        # Should find violations given ws_005 exceeds many standards
        assert result["violation_count"] > 0

    def test_recreation_risk_is_lower(self, water_samples):
        """Test that recreation risk is lower than drinking water risk."""
        from geo_infer_water.core.water_quality import (
            WaterQualityAssessor,
            WaterBodyType,
        )

        assessor = WaterQualityAssessor()
        drinking_risk = assessor.assess_risk(
            water_samples,
            WaterBodyType.RIVER,
            usage_type="drinking",
        )
        recreation_risk = assessor.assess_risk(
            water_samples,
            WaterBodyType.RIVER,
            usage_type="recreation",
        )

        # Recreation has looser standards, so fewer violations expected
        assert recreation_risk["violation_count"] <= drinking_risk["violation_count"]


class TestRegulatoryCompliance:
    """Test regulatory compliance checking."""

    def test_epa_compliance(self, water_samples):
        """Test compliance checking against EPA standards."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()
        result = assessor.check_regulatory_compliance(water_samples, regulations="EPA")

        assert result["regulations"] == "EPA"
        assert result["sample_count"] == 5
        assert isinstance(result["overall_compliant"], bool)
        # ws_005 has nitrate=15 which exceeds EPA limit of 10
        assert result["overall_compliant"] is False
        assert "results" in result

    def test_who_vs_epa_compliance(self, water_samples):
        """Test that WHO and EPA have different compliance thresholds."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()
        epa_result = assessor.check_regulatory_compliance(
            water_samples, regulations="EPA"
        )
        who_result = assessor.check_regulatory_compliance(
            water_samples, regulations="WHO"
        )

        # WHO has higher nitrate limit (50 vs 10), so may have different results
        epa_nitrate = epa_result["results"].get("nitrate", {})
        who_nitrate = who_result["results"].get("nitrate", {})

        if epa_nitrate.get("violations", 0) > 0:
            assert who_nitrate.get("violations", 0) <= epa_nitrate["violations"], (
                "WHO has looser nitrate limits so should have fewer violations"
            )


class TestPollutantLoadCalculation:
    """Test pollutant load calculations."""

    def test_pollutant_load(self):
        """Test pollutant load calculation from concentration and flow."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()
        result = assessor.calculate_pollutant_load(
            concentration_mg_l=5.0,
            flow_rate_m3_s=10.0,
            time_period_hours=24.0,
        )

        assert result["concentration_mg_l"] == 5.0
        assert result["flow_rate_m3_s"] == 10.0
        assert result["load_kg"] > 0
        assert result["load_tonnes"] == result["load_kg"] / 1000
        # 5 mg/L * 10 m3/s * 86400 s * 1000 L/m3 = 4,320,000,000 mg = 4,320 kg
        expected_kg = 5.0 * 10.0 * 86400 * 1000 / 1e6
        assert abs(result["load_kg"] - expected_kg) < 1.0


class TestWatershedAnalysis:
    """Test watershed delineation and analysis."""

    def test_watershed_delineation(self, elevation_data):
        """Test watershed delineation from an elevation model."""
        from geo_infer_water.core.watershed_delineation import WatershedDelineator

        delineator = WatershedDelineator()
        # Bowl-shaped DEM: lowest point is at the centre (row 9, col 9 of
        # the 20x20 grid), so the basin traced from it should cover the
        # whole grid.
        result = delineator.full_delineation(
            elevation_data,
            outlet=(9, 9),
            stream_threshold=1.0,
            cell_size=1000.0,
        )

        assert "basin_mask" in result
        assert "flow_direction" in result
        assert "flow_accumulation" in result
        assert "stream_network" in result
        assert "slope_degrees" in result
        assert result.attrs["basin_area_km2"] > 0

    def test_stream_network_identification(self):
        """Test stream network identification from flow accumulation."""
        from geo_infer_water.core.watershed_delineation import WatershedDelineator

        delineator = WatershedDelineator()

        # Synthetic D8 flow-direction field where every cell flows east.
        flow_dir = np.full((10, 10), 1, dtype=int)
        accumulation = delineator.calculate_flow_accumulation(flow_dir)
        assert accumulation.shape == flow_dir.shape

        streams = delineator.extract_stream_network(accumulation, threshold=0.5)
        assert streams.dtype == int


class TestPollutionSourceTracking:
    """Test pollution source identification."""

    def test_identify_hotspots(self):
        """Test pollution hotspot identification from concentration field."""
        from geo_infer_water.core.water_quality import WaterQualityAssessor

        assessor = WaterQualityAssessor()

        np.random.seed(50)
        concentration = xr.DataArray(
            np.random.exponential(2.0, (10, 10)),
            dims=["lat", "lon"],
        )

        result = assessor.identify_pollution_sources(concentration)
        assert "pollution_hotspots" in result
        assert "potential_sources" in result
        assert "concentration" in result

        # Hotspots should be a boolean mask
        assert result["pollution_hotspots"].dtype == bool

    def test_pollution_plume_tracking(self):
        """Test pollution plume dispersion modeling."""
        from geo_infer_water.core.water_quality import (
            WaterQualityAssessor,
            PollutantType,
        )

        assessor = WaterQualityAssessor()

        result = assessor.track_pollution_plume(
            initial_location=(-121.5, 38.5),
            pollutant_type=PollutantType.NUTRIENT,
            flow_velocity=(0.5, 0.1),
            diffusion_coefficient=10.0,
            time_hours=6.0,
        )

        assert result["pollutant_type"] == "nutrient"
        assert result["time_hours"] == 6.0
        assert result["plume_area_km2"] > 0
        assert result["dispersion_sigma_m"] > 0
        assert result["max_extent_km"] > 0
        assert len(result["concentration_field"]) > 0
