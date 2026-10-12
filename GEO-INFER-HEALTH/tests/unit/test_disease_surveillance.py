"""
Unit tests for disease surveillance functionality.
"""

from datetime import datetime, timedelta, UTC

import pytest
from geo_infer_health.core.disease_surveillance import DiseaseHotspotAnalyzer
from geo_infer_health.models import DiseaseReport, Location, PopulationData


class TestDiseaseHotspotAnalyzer:
    """Test cases for DiseaseHotspotAnalyzer class."""

    def test_analyzer_creation(self, sample_disease_reports, sample_population_data):
        """Test creating a DiseaseHotspotAnalyzer instance."""
        analyzer = DiseaseHotspotAnalyzer(
            reports=sample_disease_reports, population_data=sample_population_data
        )

        assert len(analyzer.reports) == len(sample_disease_reports)
        assert len(analyzer.population_data) == len(sample_population_data)

    def test_analyzer_creation_empty_reports(self):
        """Test creating analyzer with empty reports."""
        analyzer = DiseaseHotspotAnalyzer(reports=[], population_data=[])

        assert len(analyzer.reports) == 0
        assert len(analyzer.population_data) == 0

    def test_analyzer_creation_no_population_data(self, sample_disease_reports):
        """Test creating analyzer without population data."""
        analyzer = DiseaseHotspotAnalyzer(
            reports=sample_disease_reports, population_data=None
        )

        assert len(analyzer.reports) == len(sample_disease_reports)
        assert analyzer.population_data == []


class TestCasesInRadius:
    """Test cases for finding cases within radius."""

    def test_cases_in_radius_single_point(self, disease_analyzer, sample_locations):
        """Test finding cases within radius of a single point."""
        center = sample_locations[0]
        radius_km = 1.0

        cases = disease_analyzer.get_cases_in_radius(center, radius_km)

        # Should find at least the cases at the exact location
        exact_matches = [
            r
            for r in disease_analyzer.reports
            if r.location.latitude == center.latitude
            and r.location.longitude == center.longitude
        ]

        assert len(cases) >= len(exact_matches)

    def test_cases_in_radius_zero_radius(self, disease_analyzer, sample_locations):
        """Test finding cases with zero radius."""
        center = sample_locations[0]
        radius_km = 0.0

        cases = disease_analyzer.get_cases_in_radius(center, radius_km)

        # Should only find cases at exact location (within floating point precision)
        exact_cases = [
            r
            for r in disease_analyzer.reports
            if abs(r.location.latitude - center.latitude) < 1e-6
            and abs(r.location.longitude - center.longitude) < 1e-6
        ]

        assert len(cases) == len(exact_cases)

    def test_cases_in_radius_large_radius(self, disease_analyzer, sample_locations):
        """Test finding cases with large radius."""
        center = sample_locations[0]
        radius_km = 10000.0  # Very large radius

        cases = disease_analyzer.get_cases_in_radius(center, radius_km)

        # Should find all cases
        assert len(cases) == len(disease_analyzer.reports)

    def test_cases_in_radius_no_cases(self, sample_locations):
        """Test finding cases when no reports exist."""
        analyzer = DiseaseHotspotAnalyzer(reports=[], population_data=[])
        center = sample_locations[0]
        radius_km = 1.0

        cases = analyzer.get_cases_in_radius(center, radius_km)

        assert len(cases) == 0


class TestIncidenceRateCalculation:
    """Test cases for incidence rate calculations."""

    def test_incidence_rate_with_population_data(
        self, disease_analyzer, sample_locations
    ):
        """Test incidence rate calculation with population data."""
        center = sample_locations[0]
        radius_km = 5.0

        rate, cases, population, population_estimated = (
            disease_analyzer.calculate_local_incidence_rate(
                center_loc=center, radius_km=radius_km
            )
        )

        assert isinstance(rate, float)
        assert isinstance(cases, int)
        assert isinstance(population, (int, float))

        # Rate should be non-negative
        assert rate >= 0

        # Cases should be non-negative
        assert cases >= 0

    def test_incidence_rate_without_population_data(
        self, sample_disease_reports, sample_locations
    ):
        """Test incidence rate calculation without population data."""
        analyzer = DiseaseHotspotAnalyzer(
            reports=sample_disease_reports, population_data=[]
        )
        center = sample_locations[0]
        radius_km = 5.0

        rate, cases, population, population_estimated = (
            analyzer.calculate_local_incidence_rate(
                center_loc=center, radius_km=radius_km
            )
        )

        # Should return raw case count as rate when no population data
        assert rate == cases
        assert population == 0

    def test_incidence_rate_zero_population(
        self, sample_disease_reports, sample_locations
    ):
        """Test incidence rate calculation with zero population."""
        # Create population data with zero population
        zero_pop_data = [PopulationData(area_id="test_area", population_count=0)]

        analyzer = DiseaseHotspotAnalyzer(
            reports=sample_disease_reports, population_data=zero_pop_data
        )

        center = sample_locations[0]
        radius_km = 5.0

        rate, cases, population, population_estimated = (
            analyzer.calculate_local_incidence_rate(
                center_loc=center, radius_km=radius_km
            )
        )

        # Should handle zero population case
        assert rate == float("inf") or rate == cases
        assert population == 0

    def test_incidence_rate_with_time_window(self, disease_analyzer, sample_locations):
        """Test incidence rate calculation with time window."""
        center = sample_locations[0]
        radius_km = 5.0
        time_window_days = 7

        rate, cases, population, population_estimated = (
            disease_analyzer.calculate_local_incidence_rate(
                center_loc=center,
                radius_km=radius_km,
                time_window_days=time_window_days,
            )
        )

        assert isinstance(rate, float)
        assert isinstance(cases, int)
        assert isinstance(population, (int, float))

    def test_incidence_rate_different_time_windows(
        self, disease_analyzer, sample_locations
    ):
        """Test incidence rates with different time windows."""
        center = sample_locations[0]
        radius_km = 5.0

        # Short time window
        rate_short, cases_short, _, population_estimated = (
            disease_analyzer.calculate_local_incidence_rate(
                center_loc=center, radius_km=radius_km, time_window_days=1
            )
        )

        # Long time window
        rate_long, cases_long, _, population_estimated = (
            disease_analyzer.calculate_local_incidence_rate(
                center_loc=center, radius_km=radius_km, time_window_days=30
            )
        )

        # Longer time window should generally have more cases
        assert cases_long >= cases_short


class TestHotspotDetection:
    """Test cases for hotspot detection functionality."""

    def test_simple_hotspot_detection(self, disease_analyzer):
        """Test basic hotspot detection."""
        hotspots = disease_analyzer.identify_simple_hotspots(
            threshold_case_count=1, scan_radius_km=1.0
        )

        assert isinstance(hotspots, list)

        # Each hotspot should have required fields
        for hotspot in hotspots:
            assert "location" in hotspot
            assert "case_count" in hotspot
            assert "radius_km" in hotspot
            assert isinstance(hotspot["case_count"], int)
            assert hotspot["case_count"] >= 1

    def test_hotspot_detection_with_density_filter(self, disease_analyzer):
        """Test hotspot detection with density filtering."""
        hotspots = disease_analyzer.identify_simple_hotspots(
            threshold_case_count=1, scan_radius_km=1.0, min_density_cases_per_sq_km=0.1
        )

        assert isinstance(hotspots, list)

        # Check that density filtering is applied
        for hotspot in hotspots:
            area_sq_km = 3.14159 * (hotspot["radius_km"] ** 2)  # πr²
            density = hotspot["case_count"] / area_sq_km
            assert density >= 0.1

    def test_hotspot_detection_high_threshold(self, disease_analyzer):
        """Test hotspot detection with high threshold."""
        hotspots = disease_analyzer.identify_simple_hotspots(
            threshold_case_count=1000,  # Very high threshold
            scan_radius_km=1.0,
        )

        # Should find few or no hotspots
        assert len(hotspots) <= len(disease_analyzer.reports)

    def test_hotspot_detection_no_reports(self):
        """Test hotspot detection with no reports."""
        analyzer = DiseaseHotspotAnalyzer(reports=[], population_data=[])

        hotspots = analyzer.identify_simple_hotspots(
            threshold_case_count=1, scan_radius_km=1.0
        )

        assert hotspots == []

    def test_hotspot_detection_large_radius(self, disease_analyzer):
        """Test hotspot detection with large scan radius."""
        hotspots = disease_analyzer.identify_simple_hotspots(
            threshold_case_count=1,
            scan_radius_km=1000.0,  # Very large radius
        )

        # Should potentially group more cases together
        assert isinstance(hotspots, list)


class TestHotspotAnalysisIntegration:
    """Test integration of hotspot analysis components."""

    def test_hotspot_location_format(self, disease_analyzer):
        """Test that hotspot locations are properly formatted."""
        hotspots = disease_analyzer.identify_simple_hotspots(
            threshold_case_count=1, scan_radius_km=1.0
        )

        for hotspot in hotspots:
            location = hotspot["location"]
            assert isinstance(location, dict)
            assert "latitude" in location
            assert "longitude" in location
            assert "crs" in location

    def test_hotspot_case_count_accuracy(self, disease_analyzer):
        """Test that hotspot case counts are accurate."""
        threshold = 5
        radius = 2.0

        hotspots = disease_analyzer.identify_simple_hotspots(
            threshold_case_count=threshold, scan_radius_km=radius
        )

        for hotspot in hotspots:
            # Verify that the hotspot actually has at least the threshold number of cases
            center_lat = hotspot["location"]["latitude"]
            center_lon = hotspot["location"]["longitude"]

            center_loc = Location(latitude=center_lat, longitude=center_lon)
            cases_in_radius = disease_analyzer.get_cases_in_radius(center_loc, radius)

            total_cases = sum(report.case_count for report in cases_in_radius)
            assert total_cases >= threshold

    def test_hotspot_deduplication(self, disease_analyzer):
        """Test that nearby hotspots are properly deduplicated."""
        # Create analyzer with clustered reports
        clustered_reports = []

        # Create multiple reports at the same location
        base_location = Location(latitude=34.0522, longitude=-118.2437)
        for i in range(10):
            report = DiseaseReport(
                report_id=f"cluster_report_{i}",
                disease_code="TEST",
                location=base_location,
                report_date=datetime.now(UTC),
                case_count=2,
            )
            clustered_reports.append(report)

        analyzer = DiseaseHotspotAnalyzer(reports=clustered_reports, population_data=[])

        hotspots = analyzer.identify_simple_hotspots(
            threshold_case_count=5, scan_radius_km=1.0
        )

        # Should find at most one hotspot in this cluster
        # (depending on deduplication logic)
        assert len(hotspots) <= 2  # Allow some flexibility in deduplication


class TestScaleBehavior:
    """Test disease-surveillance behavior with a larger fixture."""

    @pytest.mark.slow
    def test_large_dataset_processing(self):
        """Process a larger dataset without changing hotspot semantics."""
        # Create larger dataset
        locations = []
        reports = []

        base_time = datetime.now(UTC)

        for i in range(100):  # 100 locations
            lat = 30 + (i % 10) * 0.5
            lon = -120 + (i // 10) * 0.5
            location = Location(latitude=lat, longitude=lon)
            locations.append(location)

            # 5 reports per location
            for j in range(5):
                report = DiseaseReport(
                    report_id=f"perf_report_{i}_{j}",
                    disease_code="PERF_TEST",
                    location=location,
                    report_date=base_time,
                    case_count=1,
                )
                reports.append(report)

        analyzer = DiseaseHotspotAnalyzer(reports=reports, population_data=[])

        hotspots = analyzer.identify_simple_hotspots(
            threshold_case_count=2, scan_radius_km=1.0
        )

        # Should find some hotspots
        assert len(hotspots) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_analyzer_operations(self):
        """Test operations on empty analyzer."""
        analyzer = DiseaseHotspotAnalyzer(reports=[], population_data=[])

        # Should handle empty data gracefully
        cases = analyzer.get_cases_in_radius(Location(latitude=0, longitude=0), 1.0)
        assert cases == []

        rate, cases_count, pop, population_estimated = (
            analyzer.calculate_local_incidence_rate(
                Location(latitude=0, longitude=0), 1.0
            )
        )
        assert cases_count == 0

    def test_single_report_analysis(self):
        """Test analysis with single report."""
        location = Location(latitude=34.0522, longitude=-118.2437)
        report = DiseaseReport(
            report_id="single_report",
            disease_code="TEST",
            location=location,
            report_date=datetime.now(UTC),
            case_count=1,
        )

        analyzer = DiseaseHotspotAnalyzer(reports=[report], population_data=[])

        # Should handle single report
        cases = analyzer.get_cases_in_radius(location, 1.0)
        assert len(cases) == 1

        hotspots = analyzer.identify_simple_hotspots(
            threshold_case_count=1, scan_radius_km=1.0
        )
        assert len(hotspots) >= 1

    def test_reports_with_different_timestamps(self):
        """Test handling reports with different timestamps."""
        location = Location(latitude=34.0522, longitude=-118.2437)
        base_time = datetime.now(UTC)

        reports = []
        for i in range(10):
            report_time = base_time - timedelta(days=i)
            report = DiseaseReport(
                report_id=f"time_report_{i}",
                disease_code="TEST",
                location=location,
                report_date=report_time,
                case_count=1,
            )
            reports.append(report)

        analyzer = DiseaseHotspotAnalyzer(reports=reports, population_data=[])

        # Test with different time windows
        rate_recent, _, _, population_estimated = (
            analyzer.calculate_local_incidence_rate(
                center_loc=location, radius_km=1.0, time_window_days=2
            )
        )

        rate_all, _, _, population_estimated = analyzer.calculate_local_incidence_rate(
            center_loc=location, radius_km=1.0, time_window_days=None
        )

        # Recent rate should be less than or equal to all-time rate
        assert rate_recent <= rate_all


class TestIncidencePopulationFlag:
    """population_estimated must distinguish real population from fallback."""

    def test_flag_true_with_single_population_area(
        self, sample_disease_reports, sample_locations
    ):
        # PopulationData carries no geometry; a single unlocated area acts
        # as the coarse regional population estimate.
        pop = PopulationData(area_id="region", population_count=10000)
        analyzer = DiseaseHotspotAnalyzer(
            reports=sample_disease_reports, population_data=[pop]
        )
        rate, cases, population, population_estimated = (
            analyzer.calculate_local_incidence_rate(
                center_loc=sample_locations[0], radius_km=5.0
            )
        )
        assert population_estimated is True
        assert population == 10000
        if cases > 0:
            assert rate == pytest.approx(cases / 10000 * 100000)

    def test_flag_false_without_population(
        self, sample_disease_reports, sample_locations
    ):
        analyzer = DiseaseHotspotAnalyzer(
            reports=sample_disease_reports, population_data=[]
        )
        rate, cases, population, population_estimated = (
            analyzer.calculate_local_incidence_rate(
                center_loc=sample_locations[0], radius_km=5.0
            )
        )
        assert population_estimated is False
        assert population == 0
        assert rate == cases  # raw fallback count


class TestSIRModelValidation:
    """simulate_sir_model must reject non-positive recovery rates."""

    def test_zero_gamma_raises(self, sample_disease_reports):
        analyzer = DiseaseHotspotAnalyzer(reports=sample_disease_reports)
        with pytest.raises(ValueError, match="gamma must be > 0"):
            analyzer.simulate_sir_model(
                initial_infected=1, population=100, beta=0.3, gamma=0.0
            )

    def test_negative_gamma_raises(self, sample_disease_reports):
        analyzer = DiseaseHotspotAnalyzer(reports=sample_disease_reports)
        with pytest.raises(ValueError, match="gamma must be > 0"):
            analyzer.simulate_sir_model(
                initial_infected=1, population=100, beta=0.3, gamma=-0.1
            )

    def test_positive_gamma_gives_r0(self, sample_disease_reports):
        analyzer = DiseaseHotspotAnalyzer(reports=sample_disease_reports)
        result = analyzer.simulate_sir_model(
            initial_infected=1, population=100, beta=0.3, gamma=0.1
        )
        assert result["basic_reproduction_number"] == pytest.approx(3.0)
