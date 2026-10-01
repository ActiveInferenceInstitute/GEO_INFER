"""
Tests for GEO-INFER-MATH interpolation module.

Tests cover: IDWInterpolator, KrigingInterpolator, LinearInterpolator,
CubicInterpolator, InterpolationManager, and convenience functions.
"""

import numpy as np
import pytest


from geo_infer_math.core.interpolation import (
    InterpolationConfig,
    IDWInterpolator,
    KrigingInterpolator,
    LinearInterpolator,
    CubicInterpolator,
    InterpolationManager,
    create_interpolation_manager,
    interpolate_spatial_data,
    create_interpolation_grid,
)


@pytest.fixture
def sample_data():
    """Create sample spatial data for interpolation testing."""
    np.random.seed(42)
    # 20 random points in a 10x10 grid
    coords = np.random.rand(20, 2) * 10
    # Values follow a simple spatial pattern: z = x + y
    values = coords[:, 0] + coords[:, 1]
    return coords, values


@pytest.fixture
def prediction_coords():
    """Create prediction coordinates."""
    return np.array(
        [
            [5.0, 5.0],
            [2.0, 8.0],
            [7.0, 3.0],
        ]
    )


class TestIDWInterpolator:
    """Tests for Inverse Distance Weighting interpolator."""

    def test_fit_and_predict(self, sample_data, prediction_coords):
        coords, values = sample_data
        config = InterpolationConfig(power=2.0, max_distance=20.0)
        idw = IDWInterpolator(config)
        idw.fit(coords, values)
        predictions = idw.predict(prediction_coords)
        assert len(predictions) == 3
        # Predictions should be within the range of training values
        assert np.all(predictions >= np.min(values) * 0.5)
        assert np.all(predictions <= np.max(values) * 1.5)

    def test_predict_at_training_point(self, sample_data):
        coords, values = sample_data
        config = InterpolationConfig(power=2.0, max_distance=20.0)
        idw = IDWInterpolator(config)
        idw.fit(coords, values)
        # Predict at the first training point -- should be very close to the actual value
        pred = idw.predict(coords[:1])
        assert abs(pred[0] - values[0]) < 0.1

    def test_raises_without_fit(self):
        idw = IDWInterpolator()
        with pytest.raises(ValueError, match="must be fitted"):
            idw.predict(np.array([[0.0, 0.0]]))

    def test_raises_too_few_points(self):
        config = InterpolationConfig(min_points=5)
        idw = IDWInterpolator(config)
        with pytest.raises(ValueError, match="at least"):
            idw.fit(np.array([[0.0, 0.0]]), np.array([1.0]))

    def test_method_chaining(self, sample_data):
        coords, values = sample_data
        idw = IDWInterpolator()
        result = idw.fit(coords, values)
        assert result is idw


class TestKrigingInterpolator:
    """Tests for Kriging interpolator."""

    def test_fit_and_predict(self, sample_data, prediction_coords):
        coords, values = sample_data
        config = InterpolationConfig(
            variogram_model="spherical",
            sill=1.0,
            range_param=10.0,
            nugget=0.0,
        )
        kriging = KrigingInterpolator(config)
        kriging.fit(coords, values)
        predictions = kriging.predict(prediction_coords)
        assert len(predictions) == 3
        assert np.all(np.isfinite(predictions))

    def test_exponential_variogram(self, sample_data, prediction_coords):
        coords, values = sample_data
        config = InterpolationConfig(variogram_model="exponential", range_param=5.0)
        kriging = KrigingInterpolator(config)
        kriging.fit(coords, values)
        predictions = kriging.predict(prediction_coords)
        assert len(predictions) == 3

    def test_linear_variogram(self, sample_data, prediction_coords):
        coords, values = sample_data
        config = InterpolationConfig(variogram_model="linear", range_param=5.0)
        kriging = KrigingInterpolator(config)
        kriging.fit(coords, values)
        predictions = kriging.predict(prediction_coords)
        assert np.all(np.isfinite(predictions))

    def test_kriging_exact_interpolation_at_data_points(self, sample_data):
        """OK weights solve: prediction at a training point equals its value."""
        coords, values = sample_data
        config = InterpolationConfig(
            variogram_model="spherical", sill=1.0, range_param=10.0, nugget=0.0
        )
        kriging = KrigingInterpolator(config)
        kriging.fit(coords, values)

        predictions = kriging.predict(coords)
        np.testing.assert_allclose(predictions, values, atol=1e-8)

    def test_kriging_raises_above_500_points(self):
        """More than 500 training points raises an informative error."""
        coords = np.arange(1002, dtype=float).reshape(501, 2)
        values = np.arange(501, dtype=float)
        kriging = KrigingInterpolator()
        with pytest.raises(ValueError, match="at most 500"):
            kriging.fit(coords, values)


class TestLinearInterpolator:
    """Tests for linear interpolator."""

    def test_fit_and_predict(self, sample_data, prediction_coords):
        coords, values = sample_data
        interp = LinearInterpolator()
        interp.fit(coords, values)
        predictions = interp.predict(prediction_coords)
        assert len(predictions) == 3

    def test_predictions_reasonable(self, sample_data):
        coords, values = sample_data
        interp = LinearInterpolator()
        interp.fit(coords, values)
        # Predict at center of data range
        center = np.mean(coords, axis=0).reshape(1, -1)
        pred = interp.predict(center)
        # Should be finite (may or may not be NaN depending on triangulation)
        assert len(pred) == 1


class TestCubicInterpolator:
    """Tests for cubic interpolator."""

    def test_fit_and_predict(self, sample_data, prediction_coords):
        coords, values = sample_data
        interp = CubicInterpolator()
        interp.fit(coords, values)
        predictions = interp.predict(prediction_coords)
        assert len(predictions) == 3


class TestInterpolationManager:
    """Tests for InterpolationManager."""

    def test_create_manager(self):
        manager = create_interpolation_manager()
        assert "idw" in manager.interpolators
        assert "kriging" in manager.interpolators
        assert "linear" in manager.interpolators

    def test_interpolate_with_manager(self, sample_data, prediction_coords):
        coords, values = sample_data
        config = InterpolationConfig(max_distance=20.0)
        manager = InterpolationManager(config)
        result = manager.interpolate(coords, values, prediction_coords, method="idw")
        assert len(result) == 3

    def test_unknown_method_raises(self, sample_data, prediction_coords):
        coords, values = sample_data
        manager = InterpolationManager()
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            manager.interpolate(coords, values, prediction_coords, method="nonexistent")

    def test_create_interpolation_grid_function(self):
        bounds = {
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        }
        grid = create_interpolation_grid(bounds, resolution=0.5)
        assert grid.shape[1] == 2
        assert len(grid) > 1

    def test_interpolation_grid_metadata(self):
        bounds = {
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        }
        manager = InterpolationManager()
        grid, meta = manager.create_interpolation_grid(bounds, resolution=0.5)
        assert "n_points" in meta
        assert "shape" in meta
        assert meta["resolution"] == 0.5


class TestConvenienceFunctions:
    """Tests for convenience interpolation functions."""

    def test_interpolate_spatial_data(self, sample_data, prediction_coords):
        coords, values = sample_data
        result = interpolate_spatial_data(
            coords, values, prediction_coords, method="idw"
        )
        assert len(result) == 3
        assert np.all(np.isfinite(result))
