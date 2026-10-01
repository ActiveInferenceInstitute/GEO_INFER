"""
Tests for Spatio-Temporal Analysis Module.
"""

import pytest
from datetime import datetime, timedelta


from geo_infer_space.analytics.spatiotemporal import SpatioTemporalAnalyzer
from geo_infer_space.core.spatial_methods import SpatialMethods


@pytest.fixture
def st_analyzer():
    """Create SpatioTemporalAnalyzer instance."""
    return SpatioTemporalAnalyzer()


@pytest.fixture
def spatial_methods():
    """Create SpatialMethods instance."""
    return SpatialMethods()


@pytest.fixture
def sample_spatiotemporal_data(st_analyzer):
    """Generate sample spatio-temporal data."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    base_cell = st_analyzer.h3.latlng_to_cell(37.7749, -122.4194, 8)
    neighbors = st_analyzer.h3.get_cell_neighbors(base_cell, k=2)
    all_cells = [base_cell] + list(neighbors)

    data = []
    for i, cell in enumerate(all_cells[:10]):
        for day in range(10):
            data.append(
                {
                    "cell": cell,
                    "timestamp": (base_time + timedelta(days=day)).isoformat(),
                    "value": 50 + i * 5 + day * 2 + (i % 3) * 10,
                }
            )
    return data


@pytest.fixture
def sample_trajectory_data(st_analyzer):
    """Generate sample trajectory data."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    base_cell = st_analyzer.h3.latlng_to_cell(37.7749, -122.4194, 8)
    path = st_analyzer.h3.get_cell_path(
        base_cell, st_analyzer.h3.latlng_to_cell(37.78, -122.40, 8)
    )

    data = []
    for entity_id in range(3):
        for i, cell in enumerate(path):
            data.append(
                {
                    "entity_id": f"entity_{entity_id}",
                    "cell": cell,
                    "timestamp": (
                        base_time + timedelta(hours=entity_id * 2 + i * 0.5)
                    ).isoformat(),
                }
            )
    return data


class TestSpatioTemporalAnalyzer:
    """Tests for SpatioTemporalAnalyzer class."""

    def test_initialization(self, st_analyzer):
        """Test analyzer initializes properly."""
        assert st_analyzer.h3 is not None

    def test_analyze_spatial_time_series(self, st_analyzer, sample_spatiotemporal_data):
        """Test spatial time series analysis."""
        result = st_analyzer.analyze_spatial_time_series(
            sample_spatiotemporal_data,
            cell_column="cell",
            timestamp_column="timestamp",
            value_column="value",
        )

        assert "num_cells" in result
        assert result["num_cells"] > 0
        assert "cell_analyses" in result
        assert "spatial_summary" in result

    def test_analyze_spatial_time_series_empty(self, st_analyzer):
        """Test with empty data."""
        result = st_analyzer.analyze_spatial_time_series([], "cell", "ts", "val")
        assert "error" in result

    def test_detect_spatiotemporal_clusters(
        self, st_analyzer, sample_spatiotemporal_data
    ):
        """Test ST-DBSCAN clustering."""
        result = st_analyzer.detect_spatiotemporal_clusters(
            sample_spatiotemporal_data,
            cell_column="cell",
            timestamp_column="timestamp",
            spatial_eps=2,
            temporal_eps_hours=48,
            min_points=3,
        )

        assert "num_clusters" in result
        assert "clusters" in result
        assert "noise_points" in result
        assert "total_points" in result

    def test_compute_space_time_cube(self, st_analyzer, sample_spatiotemporal_data):
        """Test space-time cube creation."""
        result = st_analyzer.compute_space_time_cube(
            sample_spatiotemporal_data,
            cell_column="cell",
            timestamp_column="timestamp",
            value_column="value",
            temporal_bin_size="day",
            aggregation="mean",
        )

        assert "num_cells" in result
        assert "num_time_bins" in result
        assert "time_slices" in result
        assert result["num_time_bins"] > 0

    def test_compute_space_time_cube_aggregations(
        self, st_analyzer, sample_spatiotemporal_data
    ):
        """Test different aggregation methods."""
        for agg in ["mean", "sum", "count", "max", "min"]:
            result = st_analyzer.compute_space_time_cube(
                sample_spatiotemporal_data,
                cell_column="cell",
                timestamp_column="timestamp",
                value_column="value",
                aggregation=agg,
            )
            assert result["aggregation"] == agg

    def test_detect_emerging_hotspots(self, st_analyzer, sample_spatiotemporal_data):
        """Test emerging hotspot detection."""
        result = st_analyzer.detect_emerging_hotspots(
            sample_spatiotemporal_data,
            cell_column="cell",
            timestamp_column="timestamp",
            value_column="value",
            time_steps=5,
        )

        assert "classifications" in result
        assert "summary" in result
        assert "emerging_hotspots" in result["summary"]

    def test_compute_spatiotemporal_autocorrelation(
        self, st_analyzer, sample_spatiotemporal_data
    ):
        """Test space-time autocorrelation."""
        result = st_analyzer.compute_spatiotemporal_autocorrelation(
            sample_spatiotemporal_data,
            cell_column="cell",
            timestamp_column="timestamp",
            value_column="value",
            spatial_lag=2,
            temporal_lag_hours=48,
        )

        assert "morans_i" in result
        assert "interpretation" in result
        assert isinstance(result["morans_i"], float)

    def test_analyze_movement_patterns(self, st_analyzer, sample_trajectory_data):
        """Test movement pattern analysis."""
        result = st_analyzer.analyze_movement_patterns(
            sample_trajectory_data,
            id_column="entity_id",
            cell_column="cell",
            timestamp_column="timestamp",
        )

        assert "num_entities" in result
        assert result["num_entities"] == 3
        assert "top_flows" in result
        assert "summary" in result

    def test_kriging_spatiotemporal(self, st_analyzer, sample_spatiotemporal_data):
        """Test space-time kriging interpolation."""
        # Get some target cells
        base_cell = st_analyzer.h3.latlng_to_cell(37.77, -122.42, 8)
        target_cells = [base_cell]

        result = st_analyzer.kriging_spatiotemporal(
            sample_spatiotemporal_data,
            target_cells=target_cells,
            target_timestamp=datetime(2024, 1, 5, 12, 0),
            cell_column="cell",
            timestamp_column="timestamp",
            value_column="value",
            spatial_range=5,
            temporal_range_hours=72,
        )

        assert "num_targets" in result
        assert "interpolated" in result


class TestSpatialMethods:
    """Tests for SpatialMethods class."""

    def test_initialization(self, spatial_methods):
        """Test methods initialize properly."""
        assert spatial_methods.h3 is not None

    def test_buffer_analysis(self, spatial_methods):
        """Test buffer creation."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)

        result = spatial_methods.buffer_analysis([cell], buffer_rings=2)

        assert "center_cells" in result
        assert "buffer_cells" in result
        assert "rings" in result
        assert len(result["rings"]) == 2
        assert result["buffer_count"] > 0

    def test_overlay_cells_intersection(self, spatial_methods):
        """Test cell intersection."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)
        neighbors = spatial_methods.h3.get_cell_neighbors(cell, k=1)

        cells_a = [cell] + list(neighbors)[:3]
        cells_b = list(neighbors)[2:] + [cell]

        result = spatial_methods.overlay_cells(cells_a, cells_b, "intersection")

        assert result["operation"] == "intersection"
        assert result["result_count"] > 0

    def test_overlay_cells_union(self, spatial_methods):
        """Test cell union."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)
        neighbors = spatial_methods.h3.get_cell_neighbors(cell, k=1)

        cells_a = [cell]
        cells_b = list(neighbors)

        result = spatial_methods.overlay_cells(cells_a, cells_b, "union")

        assert result["result_count"] == len(cells_a) + len(cells_b)

    def test_spatial_filter_threshold(self, spatial_methods):
        """Test threshold filtering."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)
        neighbors = list(spatial_methods.h3.get_cell_neighbors(cell, k=1))
        cells = [cell] + neighbors
        values = [10, 20, 30, 40, 50, 60, 70][: len(cells)]

        result = spatial_methods.spatial_filter(
            cells, values, filter_type="threshold", threshold=40
        )

        assert result["filtered_count"] < len(cells)
        assert all(v >= 40 for v in result["filtered_values"])

    def test_spatial_filter_top_n(self, spatial_methods):
        """Test top-N filtering."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)
        neighbors = list(spatial_methods.h3.get_cell_neighbors(cell, k=1))
        cells = [cell] + neighbors
        values = list(range(len(cells)))

        result = spatial_methods.spatial_filter(
            cells, values, filter_type="top_n", top_n=3
        )

        assert result["filtered_count"] == 3

    def test_aggregate_to_region(self, spatial_methods):
        """Test aggregation to coarser resolution."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 9)
        neighbors = list(spatial_methods.h3.get_cell_neighbors(cell, k=1))
        cells = [cell] + neighbors
        values = [100.0] * len(cells)

        result = spatial_methods.aggregate_to_region(
            cells, values, target_resolution=7, aggregation="mean"
        )

        assert result["output_cells"] < result["input_cells"]
        assert result["compression_ratio"] > 1

    def test_disaggregate_to_cells(self, spatial_methods):
        """Test disaggregation to finer resolution."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 6)

        result = spatial_methods.disaggregate_to_cells(
            [cell], [100.0], target_resolution=8, method="equal"
        )

        assert result["output_cells"] > result["input_cells"]

    def test_calculate_coverage(self, spatial_methods):
        """Test coverage calculation."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)
        neighbors = list(spatial_methods.h3.get_cell_neighbors(cell, k=2))
        cells = [cell] + neighbors[:5]

        result = spatial_methods.calculate_coverage(cells)

        assert result["num_cells"] == len(cells)
        assert result["total_area_km2"] > 0

    def test_find_spatial_outliers(self, spatial_methods):
        """Test spatial outlier detection."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)
        neighbors = list(spatial_methods.h3.get_cell_neighbors(cell, k=1))
        cells = [cell] + neighbors
        # Make center cell an outlier
        values = [100.0] + [10.0] * len(neighbors)

        result = spatial_methods.find_spatial_outliers(cells, values, k=1)

        assert "outliers" in result
        assert result["spatial_outlier_count"] >= 0

    def test_compute_accessibility(self, spatial_methods):
        """Test accessibility computation."""
        origin = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)
        destinations = list(spatial_methods.h3.get_cell_neighbors(origin, k=2))

        result = spatial_methods.compute_accessibility(
            [origin], destinations, max_distance=3
        )

        assert result["num_origins"] == 1
        assert result["num_destinations"] == len(destinations)
        assert origin in result["accessibility"]

    def test_calculate_spatial_weights(self, spatial_methods):
        """Test spatial weights calculation."""
        cell = spatial_methods.h3.latlng_to_cell(37.7749, -122.4194, 8)
        neighbors = list(spatial_methods.h3.get_cell_neighbors(cell, k=1))
        cells = [cell] + neighbors

        result = spatial_methods.calculate_spatial_weights(
            cells, weight_type="queen", k=1
        )

        assert result["num_cells"] == len(cells)
        assert "weights" in result
        assert result["summary"]["avg_neighbors"] > 0
