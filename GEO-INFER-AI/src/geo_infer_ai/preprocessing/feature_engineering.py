"""
Feature engineering for geospatial machine learning.

This module provides specialized preprocessing and feature engineering
for geospatial data, including handling spatial autocorrelation, creating
spatial features, and data augmentation.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GeospatialFeatureEngineer:
    """
    Feature engineering for geospatial ML datasets.

    Provides methods for creating spatial features, handling spatial
    autocorrelation, and preparing geospatial data for ML models.
    """

    def __init__(
        self, normalize: bool = True, handle_spatial_autocorr: bool = True
    ) -> None:
        """
        Initialize the feature engineer.

        Args:
            normalize: Whether to normalize features
            handle_spatial_autocorr: Accepted for API compatibility. Advisory
                only: it is stored but does not alter feature computation.
                Spatial autocorrelation is instead addressed downstream by
                spatially aware models (e.g. ``SpatialPredictor``) and by
                spatial block cross-validation during evaluation.
        """
        self.normalize = normalize
        self.handle_spatial_autocorr = handle_spatial_autocorr
        self.scaler: StandardScaler | None = None
        self.feature_names_: list[str] | None = None
        self.spatial_centroid_: np.ndarray | None = None

    def create_spatial_features(
        self,
        coordinates: np.ndarray,
        include_distances: bool = True,
        include_angles: bool = False,
        centroid: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Create spatial features from coordinates.

        Args:
            coordinates: Spatial coordinates (n_samples, 2) [lon, lat]
            include_distances: Whether to include distance-based features
            include_angles: Whether to include angular features
            centroid: Optional fitted ``[longitude, latitude]`` reference.
                When omitted, derive the centroid from ``coordinates``.

        Returns:
            DataFrame with spatial features
        """
        logger.info(f"Creating spatial features for {len(coordinates)} samples")

        lon = coordinates[:, 0]
        lat = coordinates[:, 1]
        reference = (
            np.asarray(centroid, dtype=float)
            if centroid is not None
            else np.mean(coordinates, axis=0)
        )
        if reference.shape != (2,) or not np.all(np.isfinite(reference)):
            raise ValueError("centroid must contain two finite coordinates")
        centroid_lon, centroid_lat = reference

        features: dict[str, np.ndarray] = {
            "longitude": lon,
            "latitude": lat,
        }

        if include_distances:
            # Distance from centroid
            features["distance_from_centroid"] = np.sqrt(
                (lon - centroid_lon) ** 2 + (lat - centroid_lat) ** 2
            )

            # Distance from origin
            features["distance_from_origin"] = np.sqrt(lon**2 + lat**2)

        if include_angles:
            # Angle from centroid
            features["angle_from_centroid"] = np.arctan2(
                lat - centroid_lat, lon - centroid_lon
            )

        df = pd.DataFrame(features)
        return df

    def create_temporal_features(
        self, timestamps: np.ndarray | pd.Series | list
    ) -> pd.DataFrame:
        """
        Create temporal features from timestamps.

        Args:
            timestamps: Timestamp data

        Returns:
            DataFrame with temporal features
        """
        logger.info(f"Creating temporal features for {len(timestamps)} samples")

        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)

        features = pd.DataFrame(
            {
                "year": timestamps.dt.year,
                "month": timestamps.dt.month,
                "day": timestamps.dt.day,
                "dayofweek": timestamps.dt.dayofweek,
                "dayofyear": timestamps.dt.dayofyear,
                "quarter": timestamps.dt.quarter,
                "is_weekend": (timestamps.dt.dayofweek >= 5).astype(int),
            }
        )

        # Cyclical encoding for periodic features
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        features["dayofweek_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7)
        features["dayofweek_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7)

        return features

    def fit_transform(
        self,
        X: np.ndarray | pd.DataFrame,
        coordinates: np.ndarray | None = None,
        timestamps: np.ndarray | pd.Series | list | None = None,
    ) -> np.ndarray:
        """
        Fit the feature engineer and transform data.

        Args:
            X: Input features
            coordinates: Optional spatial coordinates
            timestamps: Optional temporal data

        Returns:
            Transformed feature matrix
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        # Reset index to ensure alignment
        X_df = X_df.reset_index(drop=True)

        # Every fit establishes a new spatial contract. Do not retain a
        # centroid from an earlier dataset when this fit has no coordinates.
        self.spatial_centroid_ = None

        # Add spatial features
        if coordinates is not None:
            self.spatial_centroid_ = np.mean(coordinates, axis=0)
            spatial_features = self.create_spatial_features(
                coordinates, centroid=self.spatial_centroid_
            )
            # Reset index to ensure alignment
            spatial_features = spatial_features.reset_index(drop=True)
            X_df = pd.concat([X_df, spatial_features], axis=1, ignore_index=False)

        # Add temporal features
        if timestamps is not None:
            temporal_features = self.create_temporal_features(timestamps)
            # Reset index to ensure alignment
            temporal_features = temporal_features.reset_index(drop=True)
            X_df = pd.concat([X_df, temporal_features], axis=1, ignore_index=False)

        # Convert all column names to strings for sklearn compatibility
        X_df.columns = X_df.columns.astype(str)

        # Store feature names
        self.feature_names_ = list(X_df.columns)

        # Normalize if requested
        if self.normalize:
            self.scaler = StandardScaler()
            X_transformed = self.scaler.fit_transform(X_df)
        else:
            X_transformed = X_df.values

        if isinstance(X, np.ndarray):
            orig_num_features = X.shape[1] if X.ndim > 1 else 1
        elif isinstance(X, pd.DataFrame):
            orig_num_features = len(X.columns)
        else:
            orig_num_features = len(X)
        logger.info(
            f"Feature engineering completed. "
            f"Original features: {orig_num_features}, "
            f"Final features: {X_transformed.shape[1]}"
        )

        return np.asarray(X_transformed)

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
        coordinates: np.ndarray | None = None,
        timestamps: np.ndarray | pd.Series | list | None = None,
    ) -> np.ndarray:
        """
        Transform data using fitted feature engineer.

        Args:
            X: Input features
            coordinates: Optional spatial coordinates
            timestamps: Optional temporal data

        Returns:
            Transformed feature matrix
        """
        if self.scaler is None and self.normalize:
            raise ValueError(
                "Feature engineer must be fitted before transform. Use fit_transform first."
            )

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        # Reset index to ensure alignment
        X_df = X_df.reset_index(drop=True)

        # Add spatial features
        if coordinates is not None:
            if self.spatial_centroid_ is None:
                raise ValueError(
                    "Spatial coordinates were not fitted. Fit with training coordinates "
                    "before transforming held-out coordinates."
                )
            spatial_features = self.create_spatial_features(
                coordinates, centroid=self.spatial_centroid_
            )
            # Reset index to ensure alignment
            spatial_features = spatial_features.reset_index(drop=True)
            X_df = pd.concat([X_df, spatial_features], axis=1, ignore_index=False)

        # Add temporal features
        if timestamps is not None:
            temporal_features = self.create_temporal_features(timestamps)
            # Reset index to ensure alignment
            temporal_features = temporal_features.reset_index(drop=True)
            X_df = pd.concat([X_df, temporal_features], axis=1, ignore_index=False)

        # Convert all column names to strings for sklearn compatibility
        X_df.columns = X_df.columns.astype(str)

        # Normalize if fitted
        if self.normalize and self.scaler is not None:
            X_transformed = self.scaler.transform(X_df)
        else:
            X_transformed = X_df.values

        return np.asarray(X_transformed)

    def get_feature_names(self) -> list[str] | None:
        """
        Get feature names.

        Returns:
            List of feature names or None
        """
        return self.feature_names_

    def create_spatial_lag_features(
        self,
        values: np.ndarray,
        coordinates: np.ndarray,
        k_neighbors: int = 5,
    ) -> pd.DataFrame:
        """
        Create spatial lag features based on k-nearest neighbors.

        Spatial lag is the weighted average of a variable at neighboring
        locations, capturing spatial autocorrelation.

        Args:
            values: Values to compute lags for (n_samples,) or (n_samples, n_vars)
            coordinates: Spatial coordinates (n_samples, 2)
            k_neighbors: Number of nearest neighbors

        Returns:
            DataFrame with spatial lag features
        """
        logger.info(
            f"Creating spatial lag features with k={k_neighbors} for {len(coordinates)} samples"
        )

        n_samples = coordinates.shape[0]

        if values.ndim == 1:
            values = values.reshape(-1, 1)

        n_vars = values.shape[1]

        # Compute pairwise distance matrix
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

        lag_features: dict[str, np.ndarray] = {}

        for v in range(n_vars):
            var_lags = np.zeros(n_samples)
            var_lag_std = np.zeros(n_samples)

            for i in range(n_samples):
                distances = dist_matrix[i].copy()
                distances[i] = np.inf  # Exclude self

                # Find k nearest neighbors
                k_actual = min(k_neighbors, n_samples - 1)
                neighbor_idx = np.argsort(distances)[:k_actual]
                neighbor_dists = distances[neighbor_idx]

                # Inverse-distance weights
                weights = 1.0 / (neighbor_dists + 1e-10)
                weights /= weights.sum()

                neighbor_vals = values[neighbor_idx, v]
                var_lags[i] = np.sum(weights * neighbor_vals)
                var_lag_std[i] = np.sqrt(
                    np.sum(weights * (neighbor_vals - var_lags[i]) ** 2)
                )

            lag_features[f"spatial_lag_v{v}_mean"] = var_lags
            lag_features[f"spatial_lag_v{v}_std"] = var_lag_std

        return pd.DataFrame(lag_features)

    def create_distance_features(
        self,
        coordinates: np.ndarray,
        reference_points: np.ndarray,
        reference_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Create distance features from each sample to a set of reference points.

        Args:
            coordinates: Sample coordinates (n_samples, 2)
            reference_points: Reference coordinates (n_refs, 2)
            reference_names: Optional names for reference points

        Returns:
            DataFrame with distance features
        """
        logger.info(
            f"Creating distance features to {len(reference_points)} reference points"
        )

        n_refs = reference_points.shape[0]
        if reference_names is None:
            reference_names = [f"ref_{i}" for i in range(n_refs)]

        features: dict[str, np.ndarray] = {}

        for j in range(n_refs):
            diff = coordinates - reference_points[j]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            features[f"dist_to_{reference_names[j]}"] = dist
            # Log distance (useful for distance-decay relationships)
            features[f"log_dist_to_{reference_names[j]}"] = np.log1p(dist)

        return pd.DataFrame(features)

    def create_temporal_aggregation_features(
        self,
        values: np.ndarray,
        timestamps: np.ndarray | pd.Series | list,
        window_sizes: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Create rolling window aggregation features from time-series values.

        Args:
            values: Values to aggregate (n_samples,)
            timestamps: Timestamps for ordering
            window_sizes: List of rolling window sizes (default: [3, 7, 14])

        Returns:
            DataFrame with temporal aggregation features
        """
        if window_sizes is None:
            window_sizes = [3, 7, 14]

        logger.info(
            f"Creating temporal aggregation features with windows {window_sizes}"
        )

        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)

        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)

        # Sort by time
        sort_idx = timestamps.argsort()
        sorted_values = values[sort_idx]

        features: dict[str, np.ndarray] = {}

        for window in window_sizes:
            series = pd.Series(sorted_values)
            rolling = series.rolling(window=window, min_periods=1)

            roll_mean = rolling.mean().values
            roll_std = rolling.std().fillna(0.0).values
            roll_min = rolling.min().values
            roll_max = rolling.max().values

            # Unsort back to original order
            unsort_idx = np.argsort(sort_idx)
            features[f"rolling_mean_{window}"] = roll_mean[unsort_idx]
            features[f"rolling_std_{window}"] = roll_std[unsort_idx]
            features[f"rolling_min_{window}"] = roll_min[unsort_idx]
            features[f"rolling_max_{window}"] = roll_max[unsort_idx]

        return pd.DataFrame(features)
