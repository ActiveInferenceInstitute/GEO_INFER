"""
Basic image classification example using GEO-INFER-AI.

This example demonstrates how to use the ImageClassifier for
geospatial image classification tasks.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from geo_infer_ai.core.training import ModelTrainer, TrainingConfig
from geo_infer_ai.models.cv.image_classifier import ImageClassifier


def main() -> None:
    """Run basic classification example."""
    print("GEO-INFER-AI: Basic Image Classification Example")
    print("=" * 50)

    # Generate synthetic satellite image data
    # In practice, this would be real satellite imagery
    np.random.seed(42)
    n_samples = 200
    n_features = 100  # Flattened image features

    X = np.random.randn(n_samples, n_features)
    # Create 3 classes: water (0), vegetation (1), urban (2)
    y = np.random.randint(0, 3, n_samples)

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Classes: {len(np.unique(y))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Create classifier
    classifier = ImageClassifier(
        model_type="random_forest",
        n_classes=3,
    )

    # Configure training
    config = TrainingConfig(
        validation_split=0.2,
        save_best_model=False,
        verbose=1,
    )
    trainer = ModelTrainer(config)

    # Train the model
    print("\nTraining classifier...")
    results = trainer.train_classifier(classifier, X_train, y_train)

    print("\nTraining Results:")
    print(f"  Validation Accuracy: {results['accuracy']:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate_model(
        results["model"], X_test, y_test, task_type="classification"
    )

    print("\nTest Results:")
    print(f"  Test Accuracy: {test_results['accuracy']:.4f}")

    # Show classification report
    print("\nClassification Report:")
    report = test_results["classification_report"]
    for class_label, metrics in report.items():
        if isinstance(metrics, dict) and "precision" in metrics:
            print(f"  Class {class_label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
