"""
Spatial analysis tools for Statistical Parametric Mapping

This module provides spatial analysis capabilities for SPM, including:
- Spatial autocorrelation modeling and correction
- Cluster detection and analysis
- Spatial basis function generation
- Geographically weighted regression
- Spatial dependence diagnostics

The implementation supports various spatial models including:
- Exponential decay models
- Gaussian correlation structures
- Spherical variograms
- Non-stationary spatial processes

Mathematical Foundation:
Spatial autocorrelation is modeled using variograms:
γ(h) = (1/(2N(h))) * Σ [z(s_i) - z(s_i+h)]²

where γ(h) is the semivariogram, h is spatial lag, N(h) is number of pairs.
"""

import numpy as np
from typing import Any, cast
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

from ..models.data_models import SPMData, SPMResult


from ..utils.rng import resolve_rng


class SpatialAnalyzer:
    """
    Spatial analysis tools for SPM data.

    This class provides methods for analyzing and modeling spatial structure
    in geospatial SPM data, including autocorrelation estimation, cluster
    detection, and spatial regularization.

    Attributes:
        coordinates: Spatial coordinates of data points
        distance_matrix: Pre-computed distance matrix
        variogram_model: Fitted variogram model parameters
    """

    def __init__(self, coordinates: np.ndarray) -> None:
        """
        Initialize spatial analyzer.

        Args:
            coordinates: Spatial coordinates (n_points x 2)
        """
        self.coordinates = coordinates
        self.distance_matrix: np.ndarray | None = None
        self.variogram_model: dict[str, Any] | None = None
        self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> None:
        """Compute pairwise distance matrix."""
        self.distance_matrix = squareform(pdist(self.coordinates))

    def estimate_variogram(
        self,
        residuals: np.ndarray,
        n_bins: int = 20,
        max_distance: float | None = None,
    ) -> dict[str, Any]:
        """
        Estimate empirical variogram from residuals.

        Args:
            residuals: Model residuals
            n_bins: Number of distance bins
            max_distance: Maximum distance for variogram

        Returns:
            Dictionary with variogram parameters and fitted model
        """
        if self.distance_matrix is None:
            self._compute_distance_matrix()
        assert self.distance_matrix is not None

        distances = self.distance_matrix.flatten()
        residual_diffs = residuals[:, np.newaxis] - residuals[np.newaxis, :]
        residual_diffs = residual_diffs.flatten()

        # Remove self-comparisons (distance = 0)
        mask = distances > 0
        distances = distances[mask]
        residual_diffs = residual_diffs[mask]

        if max_distance is None:
            max_distance = np.max(distances)

        # Bin distances
        bin_edges = np.linspace(0, max_distance, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        variogram_values = np.zeros(n_bins)
        variogram_counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            variogram_counts[i] = int(np.sum(mask))
            if variogram_counts[i] > 0:
                variogram_values[i] = np.mean(residual_diffs[mask] ** 2) / 2

        if np.any(variogram_counts > 0):
            variogram_values = np.maximum.accumulate(variogram_values)

        # Fit theoretical variogram model
        model_params = self._fit_variogram_model(bin_centers, variogram_values)

        result = {
            "distances": bin_centers,
            "variogram": np.array(variogram_values),
            "counts": np.array(variogram_counts),
            "model": model_params,
            "nugget": model_params.get("nugget", 0),
            "sill": model_params.get("sill", np.var(residuals)),
            "range": model_params.get("range", max_distance / 3),
        }

        self.variogram_model = result
        return result

    def _fit_variogram_model(
        self, distances: np.ndarray, values: np.ndarray
    ) -> dict[str, Any]:
        """
        Fit theoretical variogram model to empirical values.

        Fits spherical, exponential, or Gaussian variogram models.
        """

        def spherical_model(
            h: np.ndarray, nugget: float, sill: float, range_: float
        ) -> np.ndarray:
            """Spherical variogram model."""
            range_ = np.maximum(range_, np.finfo(float).eps)
            h_norm = h / range_
            return np.where(
                h_norm <= 1,
                nugget + sill * (1.5 * h_norm - 0.5 * h_norm**3),
                nugget + sill,
            )

        def exponential_model(
            h: np.ndarray, nugget: float, sill: float, range_: float
        ) -> np.ndarray:
            """Exponential variogram model."""
            range_ = np.maximum(range_, np.finfo(float).eps)
            return cast(np.ndarray, nugget + sill * (1 - np.exp(-h / range_)))

        def gaussian_model(
            h: np.ndarray, nugget: float, sill: float, range_: float
        ) -> np.ndarray:
            """Gaussian variogram model."""
            range_ = np.maximum(range_, np.finfo(float).eps)
            return cast(np.ndarray, nugget + sill * (1 - np.exp(-((h / range_) ** 2))))

        # Try different models
        models = [
            ("spherical", spherical_model),
            ("exponential", exponential_model),
            ("gaussian", gaussian_model),
        ]

        best_params = None
        best_error = np.inf

        for model_name, model_func in models:
            try:
                # Initial parameter guesses
                sill_guess = np.max(values)
                above_half = np.where(values > sill_guess * 0.5)[0]
                range_guess = (
                    distances[above_half[0]]
                    if len(above_half)
                    else distances[len(distances) // 2]
                )
                nugget_guess = np.min(values)

                def objective(params: np.ndarray) -> float:
                    nugget, sill, range_ = params
                    predicted = model_func(distances, nugget, sill, range_)
                    return float(np.sum((values - predicted) ** 2))

                minimum_range = max(
                    float(np.finfo(float).eps), float(distances[-1]) * 1e-9
                )
                bounds = [
                    (0, sill_guess),
                    (nugget_guess, sill_guess * 2),
                    (minimum_range, distances[-1] * 2),
                ]
                range_guess = max(float(range_guess), minimum_range)
                result = minimize(
                    objective,
                    [nugget_guess, sill_guess, range_guess],
                    bounds=bounds,
                    method="L-BFGS-B",
                )

                if result.success and result.fun < best_error:
                    best_error = result.fun
                    best_params = {
                        "nugget": result.x[0],
                        "sill": result.x[1],
                        "range": result.x[2],
                        "model": model_name,
                    }

            except (ValueError, FloatingPointError, RuntimeError):
                continue

        if best_params is None:
            # Return simple exponential model with default parameters
            return {
                "nugget": 0,
                "sill": np.var(values),
                "range": distances[-1] / 3,
                "model": "exponential",
            }

        return best_params

    def create_spatial_weights(
        self, model_type: str = "exponential", **kwargs: Any
    ) -> np.ndarray:
        """
        Create spatial weights matrix based on fitted variogram.

        Args:
            model_type: Type of spatial correlation model
            **kwargs: Additional parameters for weight calculation

        Returns:
            Spatial weights matrix (n_points x n_points)
        """
        if self.variogram_model is None:
            raise ValueError("Variogram must be estimated before creating weights")

        if self.distance_matrix is None:
            self._compute_distance_matrix()
        assert self.distance_matrix is not None

        nugget = self.variogram_model["nugget"]
        sill = self.variogram_model["sill"]
        range_param = self.variogram_model["range"]

        # Compute correlation matrix
        if model_type == "exponential":
            correlation = np.exp(-self.distance_matrix / range_param)
        elif model_type == "gaussian":
            correlation = np.exp(-((self.distance_matrix / range_param) ** 2))
        elif model_type == "spherical":
            h_norm = self.distance_matrix / range_param
            correlation = np.where(h_norm <= 1, 1 - 1.5 * h_norm + 0.5 * h_norm**3, 0)
        else:
            raise ValueError(f"Unknown spatial model: {model_type}")

        # Convert correlation to covariance
        variance = sill - nugget
        covariance = nugget + variance * correlation

        # Add small regularization for numerical stability
        regularization = kwargs.get("regularization", 1e-6)
        covariance += regularization * np.eye(len(covariance))

        return cast(np.ndarray, covariance)

    def detect_clusters(
        self, statistical_map: np.ndarray, threshold: float, min_cluster_size: int = 1
    ) -> dict[str, Any]:
        """
        Detect significant clusters in statistical parametric map.

        Args:
            statistical_map: SPM statistical map
            threshold: Statistical threshold for cluster formation
            min_cluster_size: Minimum cluster size (in voxels)

        Returns:
            Dictionary with cluster information
        """
        from scipy import ndimage

        # Threshold the statistical map
        thresholded = np.abs(statistical_map) > threshold

        # Label connected components (clusters)
        labeled_clusters, n_clusters = ndimage.label(thresholded)

        # Extract cluster properties
        clusters = []
        for cluster_id in range(1, n_clusters + 1):
            cluster_mask = labeled_clusters == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= min_cluster_size:
                # Compute cluster statistics
                cluster_values = statistical_map[cluster_mask]
                max_stat = np.max(np.abs(cluster_values))
                mean_stat = np.mean(cluster_values)

                # Find cluster center of mass
                center_of_mass = ndimage.center_of_mass(cluster_mask.astype(float))

                clusters.append(
                    {
                        "id": cluster_id,
                        "size": cluster_size,
                        "max_statistic": max_stat,
                        "mean_statistic": mean_stat,
                        "center_of_mass": center_of_mass,
                        "mask": cluster_mask,
                    }
                )

        # Sort clusters by size
        clusters.sort(key=lambda x: x["size"], reverse=True)

        return {
            "n_clusters": len(clusters),
            "clusters": clusters,
            "threshold": threshold,
            "min_cluster_size": min_cluster_size,
        }

    def geographically_weighted_regression(
        self,
        data: SPMData,
        design_matrix: Any = None,
        bandwidth: float | None = None,
    ) -> SPMResult:
        """
        Perform geographically weighted regression (GWR).

        GWR allows regression coefficients to vary spatially, providing
        local parameter estimates that can reveal spatial non-stationarity.

        Args:
            data: SPMData with response and covariates
            bandwidth: Bandwidth for spatial weighting

        Returns:
            SPMResult with local regression coefficients
        """
        if design_matrix is None and data.covariates is None:
            raise ValueError("Covariates required for GWR")

        n_points = data.n_points

        # Prepare design matrix from covariates
        if design_matrix is not None:
            X = np.asarray(design_matrix.matrix)
            names = design_matrix.names or [
                f"regressor_{idx}" for idx in range(X.shape[1])
            ]
        else:
            covariates = data.covariates
            assert covariates is not None
            X = np.column_stack([np.ones(n_points)] + list(covariates.values()))
            names = ["intercept"] + list(covariates.keys())
        y = data.data

        if bandwidth is None:
            # Use cross-validation to select bandwidth
            assert self.distance_matrix is not None
            bandwidth = self._select_gwr_bandwidth(X, y)

        # Compute spatial weights for each point
        assert self.distance_matrix is not None
        local_coefficients = np.zeros((n_points, X.shape[1]))
        predictions = np.zeros(n_points)

        for i in range(n_points):
            # Gaussian kernel weights
            distances = self.distance_matrix[i, :]
            weights = np.exp(-((distances / bandwidth) ** 2))

            # Weighted least squares
            W = np.diag(weights)
            XtW = X.T @ W
            beta_i = np.linalg.pinv(XtW @ X) @ (XtW @ y)
            local_coefficients[i, :] = beta_i
            predictions[i] = X[i] @ beta_i

        # Build the result from the fitted local coefficients and predictions.
        from ..models.data_models import DesignMatrix

        design = DesignMatrix(matrix=X, names=names)

        result = SPMResult(
            spm_data=data,
            design_matrix=design,
            beta_coefficients=local_coefficients.T,  # Transpose for compatibility
            residuals=y - predictions,
            model_diagnostics={
                "method": "GWR",
                "bandwidth": bandwidth,
                "local_coefficients": local_coefficients,
            },
        )

        return result

    def _select_gwr_bandwidth(
        self, X: np.ndarray, y: np.ndarray, bandwidths: np.ndarray | None = None
    ) -> float:
        """
        Select optimal bandwidth using cross-validation.

        Args:
            X: Design matrix
            y: Response variable
            bandwidths: Candidate bandwidth values

        Returns:
            Optimal bandwidth
        """
        if bandwidths is None:
            assert self.distance_matrix is not None
            distances = self.distance_matrix.flatten()
            distances = distances[distances > 0]
            bandwidths = np.percentile(distances, [10, 25, 50, 75, 90])

        best_bandwidth = bandwidths[0]
        best_cv_error = np.inf

        for bw in bandwidths:
            cv_errors = []

            # Leave-one-out cross-validation
            for i in range(len(X)):
                # Training data (exclude point i)
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i, axis=0)
                coords_train = np.delete(self.coordinates, i, axis=0)

                # Compute weights for training points relative to left-out point
                distances = np.linalg.norm(coords_train - self.coordinates[i], axis=1)
                weights = np.exp(-((distances / bw) ** 2))

                # Weighted regression
                W = np.diag(weights)
                beta = np.linalg.pinv(X_train.T @ W @ X_train) @ (
                    X_train.T @ W @ y_train
                )

                # Predict for left-out point
                y_pred = X[i] @ beta
                cv_errors.append((y[i] - y_pred) ** 2)

            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_cv_error:
                best_cv_error = mean_cv_error
                best_bandwidth = bw

        return float(best_bandwidth)

    def spatial_basis_functions(
        self,
        n_basis: int = 10,
        basis_type: str = "gaussian",
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate spatial basis functions for modeling spatial variation.

        Args:
            n_basis: Number of basis functions
            basis_type: Type of basis functions ('gaussian', 'polynomial')
            random_seed: Optional seed for reproducible Gaussian center
                selection. When ``None`` (default) the legacy behaviour is
                kept: the global ``np.random`` state is seeded with 42.

        Returns:
            Basis function matrix (n_points x n_basis)
        """
        if basis_type == "gaussian":
            # Random Gaussian basis functions
            rng = resolve_rng(random_seed)
            center_indices = rng.choice(
                len(self.coordinates), size=n_basis, replace=False
            )
            centers = self.coordinates[center_indices]
            scale = np.std(self.coordinates) / np.sqrt(n_basis)

            basis = np.zeros((len(self.coordinates), n_basis))
            for i in range(n_basis):
                distances = np.linalg.norm(self.coordinates - centers[i], axis=1)
                basis[:, i] = np.exp(-((distances / scale) ** 2))

        elif basis_type == "polynomial":
            # Polynomial basis functions
            x, y = self.coordinates[:, 0], self.coordinates[:, 1]

            # Normalize coordinates
            x_norm = (x - np.mean(x)) / np.std(x)
            y_norm = (y - np.mean(y)) / np.std(y)

            basis_list = [np.ones(len(x))]  # Intercept

            for degree in range(1, n_basis // 2 + 1):
                basis_list.extend([x_norm**degree, y_norm**degree])

            basis = np.column_stack(basis_list[:n_basis])

        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

        return basis
