"""
Unit tests for model evaluation functionality.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from geo_infer_ai.core.model_evaluation import GeospatialModelEvaluator


class TestGeospatialModelEvaluator:
    """Test GeospatialModelEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> GeospatialModelEvaluator:
        return GeospatialModelEvaluator()

    @pytest.fixture
    def binary_data(self) -> tuple:
        np.random.seed(42)
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        return y_true, y_pred

    @pytest.fixture
    def regression_data(self) -> tuple:
        np.random.seed(42)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        return y_true, y_pred

    def test_evaluate_classification(
        self, evaluator: GeospatialModelEvaluator, binary_data: tuple
    ) -> None:
        y_true, y_pred = binary_data
        metrics = evaluator.evaluate_classification(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0

    def test_evaluate_regression(
        self, evaluator: GeospatialModelEvaluator, regression_data: tuple
    ) -> None:
        y_true, y_pred = regression_data
        metrics = evaluator.evaluate_regression(y_true, y_pred)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0.0
        assert metrics["mse"] >= 0.0
        assert metrics["rmse"] >= 0.0

    def test_evaluate_value_tolerance(
        self, evaluator: GeospatialModelEvaluator
    ) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.3, 2.9, 4.2, 5.1])
        metrics = evaluator.evaluate_value_tolerance(y_true, y_pred, tolerance=0.15)

        assert metrics["mean_absolute_error"] == pytest.approx(0.16)
        assert metrics["median_absolute_error"] == pytest.approx(0.1)
        assert metrics["max_absolute_error"] == pytest.approx(0.3)
        assert metrics["within_tolerance_percentage"] == 60.0

    @pytest.mark.parametrize(
        "actual,predicted,tolerance",
        [
            ([], [], 1),
            ([1], [1, 2], 1),
            ([np.nan], [1], 1),
            ([1], [np.inf], 1),
            ([1], [1], -1),
            ([1], [1], np.inf),
            ([1], [1], True),
        ],
    )
    def test_value_tolerance_rejects_invalid_inputs(
        self, evaluator, actual, predicted, tolerance
    ):
        """Reject invalid scientific inputs before calculating a percentage."""
        with pytest.raises(ValueError):
            evaluator.evaluate_value_tolerance(actual, predicted, tolerance)

    def test_confusion_matrix(
        self, evaluator: GeospatialModelEvaluator, binary_data: tuple
    ) -> None:
        y_true, y_pred = binary_data
        result = evaluator.compute_confusion_matrix(y_true, y_pred)

        assert "confusion_matrix" in result
        assert "labels" in result
        assert "per_class" in result
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_confusion_matrix_normalized(
        self, evaluator: GeospatialModelEvaluator, binary_data: tuple
    ) -> None:
        y_true, y_pred = binary_data
        result = evaluator.compute_confusion_matrix(y_true, y_pred, normalize="true")

        assert "confusion_matrix_normalized" in result
        cm_norm = np.array(result["confusion_matrix_normalized"])
        # Each row should sum to approximately 1 (or 0 if no samples)
        row_sums = cm_norm.sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-6 or s == 0.0

    def test_roc_auc_binary(self, evaluator: GeospatialModelEvaluator) -> None:
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_score = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7, 0.4, 0.6, 0.85, 0.15])
        result = evaluator.compute_roc_auc(y_true, y_score)

        assert "roc_auc" in result
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_cross_validate_spatial(self, evaluator: GeospatialModelEvaluator) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        coords = np.random.randn(50, 2) * 100

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        result = evaluator.cross_validate_spatial(model, X, y, coords, n_splits=3)

        assert "mean_score" in result
        assert "std_score" in result
        assert "scores" in result
        assert len(result["scores"]) == 3

    def test_cross_validate_spatial_blocks_prevent_leakage(
        self, evaluator: GeospatialModelEvaluator
    ) -> None:
        """Test folds hold out contiguous spatial blocks, not random rows."""
        np.random.seed(42)
        centers = np.array([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]])
        coords = np.vstack(
            [c + np.random.uniform(-10, 10, size=(10, 2)) for c in centers]
        )
        y = np.repeat([0.0, 100.0, 200.0], 10) + np.random.uniform(0, 5, size=30)

        folds: list[list[int]] = []

        class SpyModel:
            """Echoes the test rows back so folds can be attributed."""

            def fit(self, X: np.ndarray, y: np.ndarray) -> "SpyModel":
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                folds.append(
                    [int(np.argmin(np.sum((centers - row) ** 2, axis=1))) for row in X]
                )
                return np.full(len(X), 50.0)

        result = evaluator.cross_validate_spatial(
            SpyModel(), coords, y, coords, n_splits=3
        )

        assert len(result["scores"]) == 3
        # Every sample is tested exactly once across folds.
        assert (
            sorted(count for fold in folds for count in fold)
            == [0] * 10 + [1] * 10 + [2] * 10
        )
        # Each fold's test set is spatially contiguous: one cluster only.
        for fold in folds:
            assert len(set(fold)) == 1

    def test_cross_validate_spatial_validates_splits(
        self, evaluator: GeospatialModelEvaluator
    ) -> None:
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        coords = np.random.randn(20, 2)
        model = RandomForestRegressor(n_estimators=5, random_state=42)

        with pytest.raises(ValueError):
            evaluator.cross_validate_spatial(model, X, y, coords, n_splits=1)
        with pytest.raises(ValueError):
            evaluator.cross_validate_spatial(model, X, y, coords, n_splits=21)
