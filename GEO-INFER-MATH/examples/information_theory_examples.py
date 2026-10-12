"""
Information Theory Examples

Examples demonstrating information theory capabilities for spatial data.
"""

import numpy as np
from geo_infer_math.core.information_theory import (
    spatial_entropy,
)


def example_spatial_entropy():
    """Example: Calculate spatial entropy."""
    # Create spatial data
    coordinates = np.random.rand(100, 2) * 100
    values = np.random.rand(100)

    # Calculate spatial entropy
    entropy = spatial_entropy(coordinates, values)
    print(f"Spatial entropy: {entropy:.4f}")

    return entropy


def example_mutual_information():
    """Example: Calculate mutual information."""
    # Create two datasets
    coords_x = np.random.rand(50, 2) * 100
    values_x = np.random.rand(50)
    coords_y = np.random.rand(50, 2) * 100
    values_y = values_x + np.random.rand(50) * 0.1  # Correlated

    from geo_infer_math.core.information_theory import spatial_mutual_information

    mi = spatial_mutual_information(coords_x, values_x, coords_y, values_y)
    print(f"Mutual information: {mi:.4f}")

    return mi


if __name__ == "__main__":
    print("Information Theory Examples")
    print("=" * 50)
    example_spatial_entropy()
    example_mutual_information()
