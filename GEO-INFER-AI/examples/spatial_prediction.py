"""
Spatial prediction example using GEO-INFER-AI.

This example demonstrates how to use the SpatialPredictor for
geospatial regression and forecasting tasks.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from geo_infer_ai.core.training import ModelTrainer, TrainingConfig
from geo_infer_ai.models.predictive.spatial_predictor import SpatialPredictor
from geo_infer_ai.preprocessing.feature_engineering import GeospatialFeatureEngineer


def main() -> None:
    """Run spatial prediction example."""
    print("GEO-INFER-AI: Spatial Prediction Example")
    print("=" * 50)

    # Generate synthetic geospatial data
    # In practice, this would be real geospatial features
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # Features (e.g., elevation, slope, distance to water, etc.)
    X = np.random.randn(n_samples, n_features)

    # Spatial coordinates (longitude, latitude)
    coordinates = np.random.randn(n_samples, 2) * 10  # Scale to reasonable lat/lon

    # Target variable (e.g., temperature, precipitation, crop yield)
    # Create a relationship with both features and spatial location
    y = (
        X[:, 0] * 2.0
        + X[:, 1] * 1.5
        + coordinates[:, 0] * 0.5
        + coordinates[:, 1] * 0.3
        + np.random.randn(n_samples) * 0.5
    )

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Feature engineering
    print("\nPerforming feature engineering...")
    engineer = GeospatialFeatureEngineer(normalize=True, handle_spatial_autocorr=True)
    X_processed = engineer.fit_transform(X, coordinates=coordinates)

    print(f"Original features: {X.shape[1]}")
    print(f"Processed features: {X_processed.shape[1]}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Create predictor
    predictor = SpatialPredictor(
        model_type="random_forest",
        include_spatial_features=False,  # Already included in preprocessing
    )

    # Configure training
    config = TrainingConfig(
        validation_split=0.2,
        save_best_model=False,
        verbose=1,
    )
    trainer = ModelTrainer(config)

    # Train the model
    print("\nTraining predictor...")
    results = trainer.train_regressor(predictor, X_train, y_train)

    print("\nTraining Results:")
    print(f"  Validation R²: {results['r2']:.4f}")
    print(f"  Validation RMSE: {results['rmse']:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate_model(
        results["model"], X_test, y_test, task_type="regression"
    )

    print("\nTest Results:")
    print(f"  Test R²: {test_results['r2']:.4f}")
    print(f"  Test RMSE: {test_results['rmse']:.4f}")
    print(f"  Test MAE: {test_results['mae']:.4f}")

    # Show feature importance
    feature_importance = predictor.get_feature_importance()
    if feature_importance is not None:
        print("\nTop 5 Most Important Features:")
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        feature_names = engineer.get_feature_names()
        if feature_names:
            for idx in top_indices:
                print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
