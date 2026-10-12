"""
Unit tests for environmental health functionality.
"""

import pytest
from datetime import datetime, timedelta, UTC

from geo_infer_health.core.environmental_health import EnvironmentalHealthAnalyzer
from geo_infer_health.models import EnvironmentalData, Location


class TestEnvironmentalHealthAnalyzer:
    """Test cases for EnvironmentalHealthAnalyzer class."""

    def test_analyzer_creation(self, sample_environmental_data):
        """Test creating an EnvironmentalHealthAnalyzer instance."""
        analyzer = EnvironmentalHealthAnalyzer(
            environmental_readings=sample_environmental_data
        )

        assert len(analyzer.readings) == len(sample_environmental_data)
        # Should be sorted by timestamp
        assert analyzer.readings == sorted(analyzer.readings, key=lambda r: r.timestamp)

    def test_analyzer_creation_empty_data(self):
        """Test creating analyzer with empty data."""
        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=[])

        assert len(analyzer.readings) == 0

    def test_analyzer_creation_single_reading(self):
        """Test creating analyzer with single reading."""
        location = Location(latitude=34.0522, longitude=-118.2437)
        timestamp = datetime.now(UTC)

        reading = EnvironmentalData(
            data_id="single_reading",
            parameter_name="PM2.5",
            value=15.5,
            unit="µg/m³",
            location=location,
            timestamp=timestamp,
        )

        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=[reading])

        assert len(analyzer.readings) == 1
        assert analyzer.readings[0].data_id == "single_reading"


class TestEnvironmentalReadingsQuery:
    """Test cases for querying environmental readings."""

    def test_get_readings_near_location(self, environmental_analyzer, sample_locations):
        """Test getting readings near a location."""
        center = sample_locations[0]
        radius_km = 10.0

        readings = environmental_analyzer.get_environmental_readings_near_location(
            center_loc=center, radius_km=radius_km
        )

        assert isinstance(readings, list)

        # All returned readings should be within radius
        for reading in readings:
            distance = environmental_analyzer._calculate_distance(
                center, reading.location
            )
            assert distance <= radius_km

    def test_get_readings_with_parameter_filter(
        self, environmental_analyzer, sample_locations
    ):
        """Test getting readings with parameter filter."""
        center = sample_locations[0]
        radius_km = 1000.0
        parameter_name = "PM2.5"

        readings = environmental_analyzer.get_environmental_readings_near_location(
            center_loc=center, radius_km=radius_km, parameter_name=parameter_name
        )

        # All returned readings should match the parameter
        for reading in readings:
            assert reading.parameter_name.lower() == parameter_name.lower()

    def test_get_readings_with_time_filter(
        self, environmental_analyzer, sample_locations
    ):
        """Test getting readings with time filter."""
        center = sample_locations[0]
        radius_km = 1000.0

        base_time = datetime.now(UTC)
        start_time = base_time - timedelta(hours=2)
        end_time = base_time + timedelta(hours=1)

        readings = environmental_analyzer.get_environmental_readings_near_location(
            center_loc=center,
            radius_km=radius_km,
            start_time=start_time,
            end_time=end_time,
        )

        # All returned readings should be within time range
        for reading in readings:
            assert start_time <= reading.timestamp <= end_time

    def test_get_readings_zero_radius(self, environmental_analyzer, sample_locations):
        """Test getting readings with zero radius."""
        center = sample_locations[0]
        radius_km = 0.0

        readings = environmental_analyzer.get_environmental_readings_near_location(
            center_loc=center, radius_km=radius_km
        )

        # Should find readings at exact location (within floating point precision)
        assert isinstance(readings, list)

    def test_get_readings_large_radius(self, environmental_analyzer, sample_locations):
        """Test getting readings with large radius."""
        center = sample_locations[0]
        radius_km = 10000.0

        readings = environmental_analyzer.get_environmental_readings_near_location(
            center_loc=center, radius_km=radius_km
        )

        # Should find all readings
        assert len(readings) == len(environmental_analyzer.readings)

    def test_get_readings_no_matches(self, sample_locations):
        """Test getting readings when no readings exist."""
        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=[])
        center = sample_locations[0]
        radius_km = 1.0

        readings = analyzer.get_environmental_readings_near_location(
            center_loc=center, radius_km=radius_km
        )

        assert readings == []

    def test_get_readings_case_insensitive_parameter(
        self, environmental_analyzer, sample_locations
    ):
        """Test parameter filtering is case insensitive."""
        center = sample_locations[0]
        radius_km = 1000.0

        # Test with different cases
        readings_lower = (
            environmental_analyzer.get_environmental_readings_near_location(
                center_loc=center, radius_km=radius_km, parameter_name="pm2.5"
            )
        )

        readings_upper = (
            environmental_analyzer.get_environmental_readings_near_location(
                center_loc=center, radius_km=radius_km, parameter_name="PM2.5"
            )
        )

        readings_mixed = (
            environmental_analyzer.get_environmental_readings_near_location(
                center_loc=center, radius_km=radius_km, parameter_name="Pm2.5"
            )
        )

        # Should return same results regardless of case
        assert len(readings_lower) == len(readings_upper) == len(readings_mixed)


class TestAverageExposureCalculation:
    """Test cases for average exposure calculations."""

    def test_calculate_average_exposure(self, environmental_analyzer, sample_locations):
        """Test calculating average exposure for locations."""
        target_locations = sample_locations[:3]
        radius_km = 5.0
        parameter_name = "PM2.5"
        time_window_days = 7

        exposure_results = environmental_analyzer.calculate_average_exposure(
            target_locations=target_locations,
            radius_km=radius_km,
            parameter_name=parameter_name,
            time_window_days=time_window_days,
        )

        assert isinstance(exposure_results, dict)
        assert len(exposure_results) == len(target_locations)

        # Check that all target locations have results
        for loc in target_locations:
            key = f"{loc.latitude},{loc.longitude}"
            assert key in exposure_results

    def test_calculate_average_exposure_no_data(self, sample_locations):
        """Test calculating average exposure when no data exists."""
        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=[])
        target_locations = sample_locations[:2]
        radius_km = 5.0
        parameter_name = "PM2.5"
        time_window_days = 7

        exposure_results = analyzer.calculate_average_exposure(
            target_locations=target_locations,
            radius_km=radius_km,
            parameter_name=parameter_name,
            time_window_days=time_window_days,
        )

        # Should return None for all locations
        for loc in target_locations:
            key = f"{loc.latitude},{loc.longitude}"
            assert exposure_results[key] is None

    def test_calculate_average_exposure_single_location(
        self, environmental_analyzer, sample_locations
    ):
        """Test calculating average exposure for single location."""
        target_locations = [sample_locations[0]]
        radius_km = 10.0
        parameter_name = "PM2.5"
        time_window_days = 1

        exposure_results = environmental_analyzer.calculate_average_exposure(
            target_locations=target_locations,
            radius_km=radius_km,
            parameter_name=parameter_name,
            time_window_days=time_window_days,
        )

        assert len(exposure_results) == 1
        key = f"{target_locations[0].latitude},{target_locations[0].longitude}"
        assert key in exposure_results

        # Result should be a float or None
        result = exposure_results[key]
        assert result is None or isinstance(result, float)

    def test_calculate_average_exposure_different_parameters(
        self, environmental_analyzer, sample_locations
    ):
        """Test calculating average exposure for different parameters."""
        target_locations = [sample_locations[0]]
        radius_km = 1000.0
        time_window_days = 30

        # Test different parameters
        parameters = ["PM2.5", "Temperature", "Humidity"]

        for param in parameters:
            exposure_results = environmental_analyzer.calculate_average_exposure(
                target_locations=target_locations,
                radius_km=radius_km,
                parameter_name=param,
                time_window_days=time_window_days,
            )

            key = f"{target_locations[0].latitude},{target_locations[0].longitude}"
            result = exposure_results[key]

            # Result should be valid
            assert result is None or isinstance(result, float)
            if result is not None:
                assert result >= 0

    def test_calculate_average_exposure_time_window_filtering(
        self, environmental_analyzer, sample_locations
    ):
        """Test that time window filtering works correctly."""
        target_locations = [sample_locations[0]]
        radius_km = 1000.0
        parameter_name = "PM2.5"

        # Short time window
        short_window_results = environmental_analyzer.calculate_average_exposure(
            target_locations=target_locations,
            radius_km=radius_km,
            parameter_name=parameter_name,
            time_window_days=1,
        )

        # Long time window
        long_window_results = environmental_analyzer.calculate_average_exposure(
            target_locations=target_locations,
            radius_km=radius_km,
            parameter_name=parameter_name,
            time_window_days=30,
        )

        key = f"{target_locations[0].latitude},{target_locations[0].longitude}"

        # Results might differ based on data availability
        # This is mainly a smoke test for the functionality
        assert key in short_window_results
        assert key in long_window_results


class TestEnvironmentalHealthIntegration:
    """Test integration of environmental health components."""

    def test_readings_query_and_exposure_consistency(
        self, environmental_analyzer, sample_locations
    ):
        """Test consistency between readings query and exposure calculation."""
        target_location = sample_locations[0]
        radius_km = 5.0
        parameter_name = "PM2.5"
        time_window_days = 7

        # Get readings directly
        readings = environmental_analyzer.get_environmental_readings_near_location(
            center_loc=target_location,
            radius_km=radius_km,
            parameter_name=parameter_name,
            # Anchor to the latest reading, matching calculate_average_exposure
            start_time=environmental_analyzer.readings[-1].timestamp
            - timedelta(days=time_window_days),
            end_time=environmental_analyzer.readings[-1].timestamp,
        )

        # Calculate average exposure
        exposure_results = environmental_analyzer.calculate_average_exposure(
            target_locations=[target_location],
            radius_km=radius_km,
            parameter_name=parameter_name,
            time_window_days=time_window_days,
        )

        key = f"{target_location.latitude},{target_location.longitude}"
        avg_exposure = exposure_results[key]

        if readings:
            # If there are readings, exposure should be calculated
            assert avg_exposure is not None

            # Manual calculation should match
            manual_avg = sum(r.value for r in readings) / len(readings)
            assert avg_exposure == pytest.approx(manual_avg)
        else:
            # If no readings, exposure should be None
            assert avg_exposure is None

    def test_parameter_filtering_consistency(
        self, environmental_analyzer, sample_locations
    ):
        """Test parameter filtering consistency across methods."""
        center = sample_locations[0]
        radius_km = 1000.0
        parameter_name = "PM2.5"

        # Get readings with parameter filter
        filtered_readings = (
            environmental_analyzer.get_environmental_readings_near_location(
                center_loc=center, radius_km=radius_km, parameter_name=parameter_name
            )
        )

        # Calculate exposure for same parameter
        exposure_results = environmental_analyzer.calculate_average_exposure(
            target_locations=[center],
            radius_km=radius_km,
            parameter_name=parameter_name,
            time_window_days=30,
        )

        # All readings used in exposure calculation should match parameter
        key = f"{center.latitude},{center.longitude}"
        if exposure_results[key] is not None:
            # If exposure was calculated, there should be readings
            assert len(filtered_readings) > 0
            for reading in filtered_readings:
                assert reading.parameter_name.lower() == parameter_name.lower()


class TestPerformance:
    """Test performance characteristics of environmental health functions."""

    def test_large_dataset_performance(self):
        """Test performance with larger environmental dataset."""
        import time

        # Create larger dataset
        readings = []
        locations = []

        base_time = datetime.now(UTC)

        for i in range(200):  # 200 readings
            lat = 30 + (i % 14) * 0.5
            lon = -120 + (i // 14) * 0.5
            location = Location(latitude=lat, longitude=lon)
            locations.append(location)

            reading = EnvironmentalData(
                data_id=f"perf_reading_{i}",
                parameter_name="PM2.5" if i % 2 == 0 else "Temperature",
                value=10 + i * 0.1,
                unit="µg/m³" if i % 2 == 0 else "°C",
                location=location,
                timestamp=base_time - timedelta(hours=i),
            )
            readings.append(reading)

        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=readings)

        # Time readings query
        test_location = Location(latitude=35.0, longitude=-115.0)

        start_time = time.time()
        result_readings = analyzer.get_environmental_readings_near_location(
            center_loc=test_location, radius_km=10.0, parameter_name="PM2.5"
        )
        end_time = time.time()

        # Should complete quickly (< 0.1 seconds for 200 readings)
        duration = end_time - start_time
        assert duration < 0.1

        # Should return some results
        assert isinstance(result_readings, list)

    def test_exposure_calculation_performance(self):
        """Test performance of exposure calculation with multiple locations."""
        import time

        # Create dataset
        readings = []
        base_time = datetime.now(UTC)

        for i in range(100):
            reading = EnvironmentalData(
                data_id=f"exposure_reading_{i}",
                parameter_name="PM2.5",
                value=15 + i * 0.1,
                unit="µg/m³",
                location=Location(latitude=34.0 + i * 0.1, longitude=-118.0 + i * 0.1),
                timestamp=base_time,
            )
            readings.append(reading)

        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=readings)

        # Create multiple target locations
        target_locations = [
            Location(latitude=34.0 + i, longitude=-118.0 + i) for i in range(10)
        ]

        # Time exposure calculation
        start_time = time.time()
        exposure_results = analyzer.calculate_average_exposure(
            target_locations=target_locations,
            radius_km=5.0,
            parameter_name="PM2.5",
            time_window_days=7,
        )
        end_time = time.time()

        # Should complete in reasonable time (< 0.2 seconds)
        duration = end_time - start_time
        assert duration < 0.2

        # Should return results for all target locations
        assert len(exposure_results) == len(target_locations)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_analyzer_operations(self):
        """Test operations on empty analyzer."""
        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=[])

        # Should handle empty data gracefully
        readings = analyzer.get_environmental_readings_near_location(
            Location(latitude=0, longitude=0), 1.0
        )
        assert readings == []

        exposure = analyzer.calculate_average_exposure(
            target_locations=[Location(latitude=0, longitude=0)],
            radius_km=1.0,
            parameter_name="PM2.5",
            time_window_days=7,
        )
        key = "0.0,0.0"
        assert exposure[key] is None

    def test_single_reading_analysis(self):
        """Test analysis with single reading."""
        location = Location(latitude=34.0522, longitude=-118.2437)
        timestamp = datetime.now(UTC)

        reading = EnvironmentalData(
            data_id="single_env_reading",
            parameter_name="PM2.5",
            value=25.0,
            unit="µg/m³",
            location=location,
            timestamp=timestamp,
        )

        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=[reading])

        # Test various operations
        readings = analyzer.get_environmental_readings_near_location(
            center_loc=location, radius_km=1.0
        )
        assert len(readings) == 1

        exposure = analyzer.calculate_average_exposure(
            target_locations=[location],
            radius_km=1.0,
            parameter_name="PM2.5",
            time_window_days=1,
        )
        key = f"{location.latitude},{location.longitude}"
        assert exposure[key] == 25.0

    def test_identical_reading_locations(self):
        """Test handling of readings at identical locations."""
        location = Location(latitude=34.0522, longitude=-118.2437)
        timestamp = datetime.now(UTC)

        readings = []
        for i in range(3):
            reading = EnvironmentalData(
                data_id=f"identical_reading_{i}",
                parameter_name="PM2.5",
                value=10 + i * 5,  # Different values
                unit="µg/m³",
                location=location,  # Same location
                timestamp=timestamp + timedelta(minutes=i),
            )
            readings.append(reading)

        analyzer = EnvironmentalHealthAnalyzer(environmental_readings=readings)

        # Should find all readings
        found_readings = analyzer.get_environmental_readings_near_location(
            center_loc=location, radius_km=0.1
        )
        assert len(found_readings) == 3

        # Average exposure should be calculated correctly
        exposure = analyzer.calculate_average_exposure(
            target_locations=[location],
            radius_km=0.1,
            parameter_name="PM2.5",
            time_window_days=1,
        )
        key = f"{location.latitude},{location.longitude}"

        expected_avg = sum(r.value for r in readings) / len(readings)
        assert exposure[key] == pytest.approx(expected_avg)

    def test_time_window_edge_cases(self, sample_environmental_data):
        """Test that the exposure window anchors to the latest reading."""
        analyzer = EnvironmentalHealthAnalyzer(
            environmental_readings=sample_environmental_data
        )

        # Age all readings to 10 years ago; exposure is anchored to the
        # latest reading, so a 1-day window still includes them.
        past_time = datetime.now(UTC) - timedelta(days=365 * 10)  # 10 years ago
        for reading in analyzer.readings:
            reading.timestamp = past_time

        exposure = analyzer.calculate_average_exposure(
            target_locations=[Location(latitude=34.0, longitude=-118.0)],
            radius_km=1000.0,
            parameter_name="PM2.5",
            time_window_days=1,
        )

        key = "34.0,-118.0"
        assert exposure[key] is not None

    def test_distance_calculation_helper(self, environmental_analyzer):
        """Test the internal distance calculation method."""
        loc1 = Location(latitude=34.0522, longitude=-118.2437)
        loc2 = Location(latitude=40.7128, longitude=-74.0060)

        # Test that distance is symmetric
        dist1 = environmental_analyzer._calculate_distance(loc1, loc2)
        dist2 = environmental_analyzer._calculate_distance(loc2, loc1)

        assert dist1 == dist2
        assert dist1 > 0

        # Test distance to self
        dist_self = environmental_analyzer._calculate_distance(loc1, loc1)
        assert dist_self == pytest.approx(0.0, abs=1e-6)
