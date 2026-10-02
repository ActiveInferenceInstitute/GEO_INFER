"""Observed values retain explicit H3/time identity and missingness."""

import h3
import numpy as np
import pandas as pd
import pytest
from datetime import UTC, datetime

from geo_infer_space import H3StateSpace, align_h3_observations
from geo_infer_space.analytics.spatiotemporal import SpatioTemporalAnalyzer
from geo_infer_space.models.data_models import SpatialMetadata


@pytest.fixture
def cells():
    center = h3.latlng_to_cell(41.75, -124.2, 8)
    return sorted(h3.grid_disk(center, 1), reverse=True)[:2]


def test_explicit_axes_preserve_zero_missing_values_and_input_ownership(cells):
    space = H3StateSpace(cells)
    timestamps = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]
    data = pd.DataFrame(
        [
            {"cell": cells[1], "timestamp": timestamps[1], "value": 9},
            {"cell": cells[0], "timestamp": "2023-12-31T16:00:00-08:00", "value": 0},
            {"cell": cells[1], "timestamp": timestamps[0], "value": 3},
        ]
    )
    series = align_h3_observations(data, state_space=space, timestamps=timestamps)
    expected = pd.DataFrame(
        [[0.0, 3.0], [np.nan, 9.0]], columns=cells, index=pd.DatetimeIndex(timestamps)
    )
    pd.testing.assert_frame_equal(series.to_dataframe(), expected)
    data.loc[0, "value"] = 99
    assert series.to_dataframe().iloc[1, 1] == 9
    reordered = align_h3_observations(
        data, state_space=H3StateSpace(cells[::-1]), timestamps=timestamps
    )
    assert list(reordered.to_dataframe().columns) == cells[::-1]
    assert reordered.to_dataframe().iloc[0].tolist() == [3.0, 0.0]


def test_duplicate_normalized_cell_instants_rejected(cells):
    data = pd.DataFrame(
        {
            "cell": [cells[0]] * 2,
            "timestamp": ["2024-01-01T00:00:00Z", "2023-12-31T16:00:00-08:00"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        align_h3_observations(
            data, state_space=H3StateSpace(cells), timestamps=["2024-01-01T00:00:00Z"]
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_observations_are_not_converted_to_missingness(cells, value):
    data = pd.DataFrame(
        {"cell": [cells[0]], "timestamp": ["2024-01-01T00:00:00Z"], "value": [value]}
    )
    with pytest.raises(ValueError, match="finite"):
        align_h3_observations(
            data, state_space=H3StateSpace(cells), timestamps=["2024-01-01T00:00:00Z"]
        )


@pytest.mark.parametrize(
    "column,value", [("cell", "invalid"), ("timestamp", "2024-01-01T01:00:00Z")]
)
def test_outside_axis_observation_rejected(cells, column, value):
    data = pd.DataFrame(
        {"cell": [cells[0]], "timestamp": ["2024-01-01T00:00:00Z"], "value": [2.0]}
    )
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="outside"):
        align_h3_observations(
            data, state_space=H3StateSpace(cells), timestamps=["2024-01-01T00:00:00Z"]
        )


def test_budget_checked_without_consuming_unbounded_time_axis(cells):
    seen = []

    def timestamps():
        for hour in range(100):
            seen.append(hour)
            yield pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(hours=hour)

    with pytest.raises(ValueError, match="max_entries"):
        align_h3_observations(
            pd.DataFrame(columns=["cell", "timestamp", "value"]),
            state_space=H3StateSpace(cells),
            timestamps=timestamps(),
            max_entries=4,
        )
    assert seen == [0, 1, 2]


def test_utc_cube_bins_are_invariant_to_offset_representation(cells):
    analyzer = SpatioTemporalAnalyzer()
    data = [
        {"cell": cells[0], "timestamp": stamp, "value": value}
        for stamp, value in [
            ("2024-01-01T00:00:00Z", 2.0),
            ("2023-12-31T16:00:00-08:00", 4.0),
        ]
    ]
    result = analyzer.compute_space_time_cube(data, "cell", "timestamp", "value")
    assert result["time_bins"] == ["2024-01-01"]
    assert result["time_slices"]["2024-01-01"]["values"][cells[0]] == 3
    series = analyzer.analyze_spatial_time_series(data, "cell", "timestamp", "value")
    assert series["cell_analyses"][cells[0]]["mean"] == 3


def test_week_cube_uses_iso_week_year(cells):
    data = [{"cell": cells[0], "timestamp": "2019-12-30T00:00:00Z", "value": 2.0}]
    result = SpatioTemporalAnalyzer().compute_space_time_cube(
        data, "cell", "timestamp", "value", temporal_bin_size="week"
    )
    assert result["time_bins"] == ["2020-W01"]


@pytest.mark.parametrize(
    "values", [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [-5.0, -5.001, -5.002]]
)
def test_signed_values_use_nonnegative_trend_threshold(values):
    assert SpatioTemporalAnalyzer()._detect_trend(values)["direction"] == "stable"


def test_weighted_interpolation_has_an_independent_value_oracle(cells):
    analyzer = SpatioTemporalAnalyzer()
    records = [
        {"cell": cells[0], "timestamp": stamp, "value": value}
        for stamp, value in [
            ("2024-01-01T00:00:00Z", 2.0),
            ("2024-01-01T02:00:00Z", 6.0),
        ]
    ]
    result = analyzer.interpolate_spatiotemporal(
        records,
        [cells[0]],
        pd.Timestamp("2024-01-01T01:00:00Z"),
        "cell",
        "timestamp",
        "value",
    )
    assert result["method"] == "softened_inverse_distance"
    assert result["interpolated"][cells[0]]["value"] == 4


def test_pentagon_clustering_uses_real_neighbor_topology():
    pentagon = h3.get_pentagons(8)[0]
    neighbors = sorted(set(h3.grid_disk(pentagon, 1)) - {pentagon})
    records = [
        {"cell": cell, "timestamp": "2024-01-01T00:00:00Z"}
        for cell in [pentagon, *neighbors]
    ]
    result = SpatioTemporalAnalyzer().detect_spatiotemporal_clusters(
        records, "cell", "timestamp", spatial_eps=1, min_points=2
    )
    assert result["num_clusters"] == 1
    assert result["noise_points"] == 0
    assert result["clusters"][0]["size"] == 6


def test_invalid_and_mixed_resolution_cells_fail_before_analysis(cells):
    analyzer = SpatioTemporalAnalyzer()
    for other in ["invalid", h3.cell_to_parent(cells[0], 7)]:
        data = [
            {"cell": cell, "timestamp": "2024-01-01T00:00:00Z", "value": 2.0}
            for cell in [cells[0], other]
        ]
        with pytest.raises(ValueError, match="H3"):
            analyzer.compute_space_time_cube(data, "cell", "timestamp", "value")


def test_spatial_metadata_uses_the_same_utc_boundary():
    metadata = SpatialMetadata(name="observations")
    assert metadata.created_at.tzinfo == UTC
    metadata.updated_at = datetime.fromisoformat("2023-12-31T16:00:00-08:00")
    assert metadata.updated_at == datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        SpatialMetadata(name="ambiguous", created_at=datetime(2024, 1, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        metadata.updated_at = datetime(2024, 1, 1)
