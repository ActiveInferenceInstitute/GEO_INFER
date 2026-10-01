"""
Linear Algebra and Tensor Operations Module

This module provides functions and structures for handling linear algebra
operations and multi-dimensional geospatial data (tensors) for spatial analysis.
"""

import numpy as np
from typing import Any, cast
from dataclasses import dataclass, field
import logging

from geo_infer_math.core.spatial_statistics import morans_i_variance
from geo_infer_math.utils.rng import resolve_rng

logger = logging.getLogger(__name__)


@dataclass
class TensorData:
    """Container for multi-dimensional geospatial data."""

    data: np.ndarray
    coordinates: np.ndarray | None = None
    dimensions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dimensions:
            self.dimensions = [f"dim_{i}" for i in range(self.data.ndim)]


class MatrixOperations:
    """Linear algebra operations for geospatial matrices."""

    @staticmethod
    def condition_number(matrix: np.ndarray) -> float:
        """
        Calculate the condition number of a matrix.

        Args:
            matrix: Input matrix

        Returns:
            Condition number (ratio of largest to smallest singular value)
        """
        try:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            return float(singular_values[0] / singular_values[-1])
        except np.linalg.LinAlgError:
            logger.warning("Matrix is singular or nearly singular")
            return float("inf")

    @staticmethod
    def is_positive_definite(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
        """
        Check if a matrix is positive definite.

        Args:
            matrix: Input matrix
            tolerance: Numerical tolerance

        Returns:
            True if matrix is positive definite
        """
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return bool(np.all(eigenvalues > -tolerance))
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def nearest_positive_definite(
        matrix: np.ndarray, epsilon: float = 1e-8
    ) -> np.ndarray:
        """
        Find the nearest positive definite matrix.

        Args:
            matrix: Input matrix
            epsilon: Minimum eigenvalue threshold

        Returns:
            Nearest positive definite matrix
        """
        # Ensure matrix is symmetric
        B = (matrix + matrix.T) / 2

        # Compute eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)

        # Replace negative or very small eigenvalues with epsilon
        eigenvalues[eigenvalues <= epsilon] = epsilon

        # Reconstruct matrix
        return cast(np.ndarray, eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)

    @staticmethod
    def spatial_weights_matrix(
        points: np.ndarray,
        method: str = "inverse_distance",
        k: int = 5,
        threshold: float | None = None,
    ) -> np.ndarray:
        """
        Create spatial weights matrix from point coordinates.

        Args:
            points: Point coordinates (n_points, 2)
            method: Weighting method ('inverse_distance', 'knn', 'gaussian')
            k: Number of nearest neighbors (for knn method)
            threshold: Distance threshold for binary weights

        Returns:
            Spatial weights matrix (n_points, n_points)
        """
        n_points = len(points)
        weights = np.zeros((n_points, n_points))

        # Calculate pairwise distances
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))

        if method == "inverse_distance":
            # Inverse distance weighting
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:
                        weights[i, j] = 1.0 / (distances[i, j] + 1e-10)

        elif method == "knn":
            # K-nearest neighbors
            for i in range(n_points):
                # Find k nearest neighbors
                neighbor_indices = np.argsort(distances[i])[1 : k + 1]  # Exclude self
                weights[i, neighbor_indices] = 1.0

        elif method == "gaussian":
            # Gaussian kernel weights
            sigma = np.std(
                distances[distances > 0]
            )  # Use distance standard deviation as sigma
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:
                        weights[i, j] = np.exp(-(distances[i, j] ** 2) / (2 * sigma**2))

        elif method == "binary":
            # Binary weights based on threshold (NOT row-standardized)
            if threshold is None:
                threshold = np.mean(distances[distances > 0])
            weights = (distances <= threshold).astype(float)
            np.fill_diagonal(weights, 0)  # No self-weights
            # Return binary weights directly (skip general row-standardization)
            return weights

        # Row-standardize the weights (inverse_distance, knn, gaussian)
        row_sums = weights.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums[:, np.newaxis]

        return cast(np.ndarray, weights)

    @staticmethod
    def moran_i_matrix(
        values: np.ndarray, weights_matrix: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate Moran's I statistic using matrix operations.

        Args:
            values: Value array
            weights_matrix: Spatial weights matrix

        Returns:
            Moran's I statistics
        """
        n = len(values)

        # Standardize values
        z = (values - np.mean(values)) / np.std(values)

        # Calculate Moran's I
        numerator = z.T @ weights_matrix @ z
        denominator = np.sum(z**2)

        I = (n / np.sum(weights_matrix)) * (numerator / denominator)

        # Expected value and variance
        expected_I = -1.0 / (n - 1)

        var_I = morans_i_variance(values, weights_matrix)

        # Z-score and p-value (two-tailed using erfc)
        z_score = (I - expected_I) / np.sqrt(var_I) if var_I > 0 else 0.0
        from math import erfc, sqrt

        p_value = erfc(abs(z_score) / sqrt(2))

        return {
            "I": I,
            "expected_I": expected_I,
            "variance": var_I,
            "z_score": z_score,
            "p_value": p_value,
        }


class TensorOperations:
    """Operations for multi-dimensional geospatial data."""

    @staticmethod
    def create_spatiotemporal_tensor(
        spatial_data: list[np.ndarray],
        temporal_indices: list[float],
        spatial_coords: np.ndarray | None = None,
    ) -> TensorData:
        """
        Create a spatiotemporal tensor from spatial data over time.

        Args:
            spatial_data: List of spatial arrays (one per time step)
            temporal_indices: Time indices
            spatial_coords: Spatial coordinates

        Returns:
            TensorData object
        """
        # Stack spatial data into 3D tensor (time, height, width)
        tensor_data = np.stack(spatial_data, axis=0)

        dimensions = ["time", "latitude", "longitude"]
        metadata = {
            "temporal_indices": temporal_indices,
            "n_time_steps": len(temporal_indices),
            "spatial_shape": spatial_data[0].shape,
        }

        return TensorData(
            data=tensor_data,
            coordinates=spatial_coords,
            dimensions=dimensions,
            metadata=metadata,
        )

    @staticmethod
    def tensor_unfold(
        tensor: TensorData, mode: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Unfold tensor along a specific mode (MATRICIZATION).

        Args:
            tensor: Input tensor
            mode: Mode along which to unfold (0, 1, or 2 for 3D tensor)

        Returns:
            Tuple of (unfolded_matrix, unfolding_info)
        """
        data = tensor.data

        if data.ndim != 3:
            raise ValueError("Tensor must be 3-dimensional for unfolding")

        n0, n1, n2 = data.shape

        if mode == 0:
            # Unfold along mode 0 (time)
            unfolded = data.reshape(n0, n1 * n2).T
            shape_info = {"original_shape": (n0, n1, n2), "mode": 0}
        elif mode == 1:
            # Unfold along mode 1 (latitude)
            unfolded = data.transpose(1, 0, 2).reshape(n1, n0 * n2)
            shape_info = {"original_shape": (n0, n1, n2), "mode": 1}
        elif mode == 2:
            # Unfold along mode 2 (longitude)
            unfolded = data.transpose(2, 0, 1).reshape(n2, n0 * n1).T
            shape_info = {"original_shape": (n0, n1, n2), "mode": 2}
        else:
            raise ValueError("Mode must be 0, 1, or 2 for 3D tensor")

        return unfolded, shape_info

    @staticmethod
    def tensor_fold(
        unfolded_matrix: np.ndarray, shape_info: dict[str, Any]
    ) -> np.ndarray:
        """
        Fold unfolded matrix back into tensor.

        Args:
            unfolded_matrix: Unfolded matrix
            shape_info: Information from unfolding operation

        Returns:
            Reconstructed tensor
        """
        original_shape = shape_info["original_shape"]
        mode = shape_info["mode"]

        n0, n1, n2 = original_shape

        if mode == 0:
            folded = unfolded_matrix.T.reshape(original_shape)
        elif mode == 1:
            temp = unfolded_matrix.reshape((n1, n0, n2))
            folded = temp.transpose(1, 0, 2)
        elif mode == 2:
            temp = unfolded_matrix.T.reshape((n2, n0, n1))
            folded = temp.transpose(1, 2, 0)

        return folded

    @staticmethod
    def principal_component_analysis(
        tensor: TensorData, n_components: int | None = None
    ) -> dict[str, Any]:
        """
        Perform PCA on tensor data.

        Args:
            tensor: Input tensor data
            n_components: Number of principal components

        Returns:
            PCA results
        """
        # Unfold tensor along spatial mode (combine space-time)
        unfolded, shape_info = TensorOperations.tensor_unfold(tensor, mode=0)

        # Perform PCA
        if n_components is None:
            n_components = min(unfolded.shape)

        # Center the data
        unfolded_centered = unfolded - np.mean(unfolded, axis=0)

        # Compute SVD
        U, s, Vt = np.linalg.svd(unfolded_centered, full_matrices=False)

        # Select components
        U_reduced = U[:, :n_components]
        s_reduced = s[:n_components]
        Vt_reduced = Vt[:n_components]

        # Reconstruct principal components
        principal_components = U_reduced @ np.diag(s_reduced)

        # Calculate explained variance
        explained_variance = s_reduced**2 / np.sum(s**2)
        cumulative_variance = np.cumsum(explained_variance)

        return {
            "principal_components": principal_components,
            "explained_variance": explained_variance,
            "cumulative_variance": cumulative_variance,
            "eigenvalues": s_reduced**2,
            "eigenvectors": Vt_reduced.T,
            "projection_matrix": U_reduced,
            "singular_values": s_reduced,
        }

    @staticmethod
    def tensor_decomposition(
        tensor: TensorData, rank: int, method: str = "cp", rng: Any | None = None
    ) -> dict[str, Any]:
        """
        Perform tensor decomposition (CP or Tucker).

        Args:
            tensor: Input tensor
            rank: Decomposition rank
            method: Decomposition method ('cp' or 'tucker')
            rng: Optional seed or np.random.Generator for the CP-ALS
                initialization (resolved via ``resolve_rng``). Only used by
                method='cp'; the Tucker/HOSVD path is fully deterministic.

        Returns:
            Decomposition results
        """
        if method == "cp":
            return TensorOperations._cp_decomposition(tensor, rank, rng=rng)
        elif method == "tucker":
            return TensorOperations._tucker_decomposition(tensor, rank)
        else:
            raise ValueError("Method must be 'cp' or 'tucker'")

    @staticmethod
    def _khatri_rao(columns_a: np.ndarray, columns_b: np.ndarray) -> np.ndarray:
        """Column-wise Khatri-Rao product with rows indexed (a-index, b-index)."""
        rank = columns_a.shape[1]
        product = np.zeros((columns_a.shape[0] * columns_b.shape[0], rank))
        for r in range(rank):
            product[:, r] = np.kron(columns_a[:, r], columns_b[:, r])
        return product

    @staticmethod
    def _cp_decomposition(
        tensor: TensorData,
        rank: int,
        rng: Any | None = None,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> dict[str, Any]:
        """CP (CANDECOMP/PARAFAC) decomposition via alternating least squares.

        Factor matrices are initialized deterministically from uniform draws
        of the resolved generator, then each mode is updated in turn by
        solving the mode-k least-squares problem against the Khatri-Rao
        product of the other two factors. Iteration stops when the relative
        Frobenius reconstruction error falls below ``tol`` or after
        ``max_iter`` sweeps.
        """
        data = np.asarray(tensor.data, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError("CP decomposition requires a 3-dimensional tensor")
        n0, n1, n2 = data.shape
        if rank < 1:
            raise ValueError("CP decomposition rank must be at least 1")

        gen = resolve_rng(rng)
        A = gen.uniform(size=(n0, rank))
        B = gen.uniform(size=(n1, rank))
        C = gen.uniform(size=(n2, rank))

        norm_data = np.linalg.norm(data)
        unfolded_0 = data.reshape(n0, -1)
        unfolded_1 = np.moveaxis(data, 1, 0).reshape(n1, -1)
        unfolded_2 = np.moveaxis(data, 2, 0).reshape(n2, -1)

        errors: list[float] = []
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            kr_BC = TensorOperations._khatri_rao(B, C)  # rows (i1, i2)
            A = np.linalg.lstsq(kr_BC, unfolded_0.T, rcond=None)[0].T

            kr_AC = TensorOperations._khatri_rao(A, C)  # rows (i0, i2)
            B = np.linalg.lstsq(kr_AC, unfolded_1.T, rcond=None)[0].T

            kr_AB = TensorOperations._khatri_rao(A, B)  # rows (i0, i1)
            C = np.linalg.lstsq(kr_AB, unfolded_2.T, rcond=None)[0].T

            reconstruction = np.einsum("ir,jr,kr->ijk", A, B, C)
            rel_error = float(np.linalg.norm(data - reconstruction) / norm_data)
            errors.append(rel_error)
            if rel_error < tol:
                break

        # Column-normalize the leading factor and carry the scales as weights.
        weights = np.linalg.norm(A, axis=0)
        weights[weights == 0.0] = 1.0
        A = A / weights

        return {
            "factor_matrices": [A, B, C],
            "weights": weights,
            "rank": rank,
            "method": "cp",
            "errors": errors,
            "n_iter": n_iter,
            "converged": bool(errors[-1] < tol),
        }

    @staticmethod
    def _tucker_decomposition(tensor: TensorData, rank: int) -> dict[str, Any]:
        """Tucker decomposition via HOSVD (higher-order SVD).

        Each mode-k unfolding is decomposed with a real SVD; the leading
        left singular vectors form the factor matrices and the core tensor
        is the original tensor projected onto all three factor bases.
        """
        data = np.asarray(tensor.data, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError("Tucker decomposition requires a 3-dimensional tensor")
        n0, n1, n2 = data.shape

        factor_matrices: list[np.ndarray] = []
        for mode, dim in enumerate((n0, n1, n2)):
            if mode == 0:
                unfolded = data.reshape(dim, -1)
            elif mode == 1:
                unfolded = np.moveaxis(data, 1, 0).reshape(dim, -1)
            else:
                unfolded = np.moveaxis(data, 2, 0).reshape(dim, -1)
            u, _, _ = np.linalg.svd(unfolded, full_matrices=False)
            factor_matrices.append(u[:, : min(dim, rank)])

        core = np.einsum(
            "ia,jb,kc,ijk->abc",
            factor_matrices[0],
            factor_matrices[1],
            factor_matrices[2],
            data,
        )

        return {
            "core_tensor": core,
            "factor_matrices": factor_matrices,
            "rank": rank,
            "method": "tucker",
        }


class SpatialLinearAlgebra:
    """Specialized linear algebra for spatial problems."""

    @staticmethod
    def solve_spatial_regression(
        X: np.ndarray, y: np.ndarray, weights_matrix: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Solve spatial regression with optional spatial weights.

        Without ``weights_matrix`` this is ordinary least squares. With a
        weights matrix, a spatial lag (SAR) model is fitted:

            y = rho * W y + X beta + epsilon

        ``rho`` is estimated by maximizing the concentrated log-likelihood
        over a grid in [0, 0.95]; standard errors come from the inverse of
        the asymptotic information matrix of the full log-likelihood in
        (beta, rho).

        Args:
            X: Design matrix (n_samples, n_features)
            y: Target values (n_samples,)
            weights_matrix: Spatial weights matrix

        Returns:
            Regression results

        Raises:
            ValueError: If the information matrix is singular or a variance
                estimate is non-positive (standard errors undefined).
        """
        if weights_matrix is None:
            # Ordinary least squares
            coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ coefficients

            # Calculate statistics
            n, p = X.shape
            mse = np.sum(residuals**2) / (n - p)
            se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
            t_stats = coefficients / se
            r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y)) ** 2)
            rho_hat = 0.0

        else:
            W = np.asarray(weights_matrix, dtype=np.float64)
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64).ravel()
            n, p = X.shape
            identity_n = np.eye(n)

            def _sar_loglik(beta: np.ndarray, rho: float) -> float:
                """Full Gaussian log-likelihood of the SAR model."""
                a_matrix = identity_n - rho * W
                sign, log_det = np.linalg.slogdet(a_matrix)
                if sign <= 0:
                    return -np.inf
                u = a_matrix @ y - X @ beta
                sigma2 = float(u @ u) / n
                if sigma2 <= 0.0:
                    return -np.inf
                return float(
                    -0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2)) + log_det
                )

            # Profile-likelihood grid search over rho
            rho_grid = np.linspace(0.0, 0.95, 200)
            best_rho = 0.0
            best_ll = -np.inf
            for rho_candidate in rho_grid:
                e_transformed = y - rho_candidate * (W @ y)
                beta_candidate = np.linalg.lstsq(X, e_transformed, rcond=None)[0]
                ll = _sar_loglik(beta_candidate, rho_candidate)
                if ll > best_ll:
                    best_ll = ll
                    best_rho = float(rho_candidate)

            rho = best_rho
            e_transformed = y - rho * (W @ y)
            coefficients = np.linalg.lstsq(X, e_transformed, rcond=None)[0]
            residuals = e_transformed - X @ coefficients
            r_squared = 1 - np.sum(residuals**2) / np.sum(
                (e_transformed - np.mean(e_transformed)) ** 2
            )

            # Asymptotic information matrix of the concentrated SAR
            # log-likelihood in (beta, rho), computed in closed form at the
            # estimates (sigma^2 = u^T u / n):
            #   I_bb   = X^T X / sigma^2
            #   I_b,r  = X^T W y / sigma^2
            #   I_rr   = ||W y||^2 / sigma^2 - 2 (u^T W y)^2 / (n sigma^4)
            #            - tr((A^{-1} W)^2),  A = I - rho W
            sigma2 = float(np.sum(residuals**2)) / n
            if sigma2 <= 0.0 or not np.isfinite(sigma2):
                raise ValueError(
                    "SAR residual variance is degenerate; standard errors "
                    "are undefined for this data"
                )
            a_matrix = np.eye(n) - rho * W
            wy = W @ y
            u_wy = float(residuals @ wy)
            m_inv_w = np.linalg.solve(a_matrix, W)
            trace_inv_w_sq = float(np.sum(m_inv_w * m_inv_w.T))

            info_bb = (X.T @ X) / sigma2
            info_b_rho = (X.T @ wy) / sigma2
            info_rr = (
                float(wy @ wy) / sigma2
                - 2.0 * u_wy**2 / (n * sigma2**2)
                - trace_inv_w_sq
            )
            info_matrix = np.zeros((p + 1, p + 1))
            info_matrix[:p, :p] = info_bb
            info_matrix[:p, p] = info_matrix[p, :p] = info_b_rho
            info_matrix[p, p] = info_rr

            try:
                covariance = np.linalg.inv(info_matrix)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "SAR information matrix is singular; standard errors are "
                    f"undefined for this weights matrix and data (rho={rho:.4f})"
                ) from exc
            variances = np.diag(covariance)
            if np.any(variances <= 0.0) or not np.all(np.isfinite(variances)):
                raise ValueError(
                    "SAR information matrix is singular; standard errors are "
                    f"undefined for this weights matrix and data (rho={rho:.4f})"
                )
            se = np.sqrt(variances[:p])
            t_stats = coefficients / se
            rho_hat = rho

        return {
            "coefficients": coefficients,
            "standard_errors": se,
            "t_statistics": t_stats,
            "r_squared": r_squared,
            "residuals": residuals,
            "rho": rho_hat,
        }

    @staticmethod
    def spatial_eigen_analysis(
        weights_matrix: np.ndarray, n_eigenvectors: int = 10
    ) -> dict[str, Any]:
        """
        Perform eigen analysis of spatial weights matrix.

        Args:
            weights_matrix: Spatial weights matrix
            n_eigenvectors: Number of eigenvectors to compute

        Returns:
            Eigen analysis results
        """
        # Ensure matrix is symmetric
        W = (weights_matrix + weights_matrix.T) / 2

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(W)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select requested number of eigenvectors
        eigenvalues = eigenvalues[:n_eigenvectors]
        eigenvectors = eigenvectors[:, :n_eigenvectors]

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "n_eigenvectors": n_eigenvectors,
        }

    @staticmethod
    def cholesky_decomposition(matrix: np.ndarray) -> np.ndarray:
        """
        Perform Cholesky decomposition for positive definite matrices.

        Args:
            matrix: Positive definite matrix

        Returns:
            Lower triangular matrix L such that L*L^T = matrix
        """
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            logger.warning("Matrix is not positive definite, finding nearest PD matrix")
            pd_matrix = MatrixOperations.nearest_positive_definite(matrix)
            return np.linalg.cholesky(pd_matrix)

    @staticmethod
    def matrix_inverse(matrix: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        Compute matrix inverse using various methods.

        Args:
            matrix: Input matrix
            method: Inversion method ('standard', 'svd', 'iterative')

        Returns:
            Matrix inverse
        """
        if method == "standard":
            try:
                return np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                logger.warning("Matrix is singular, using pseudo-inverse")
                return np.linalg.pinv(matrix)

        elif method == "svd":
            U, s, Vt = np.linalg.svd(matrix)
            # Filter small singular values
            s_inv = np.where(s > 1e-10, 1.0 / s, 0.0)
            return cast(np.ndarray, Vt.T @ np.diag(s_inv) @ U.T)

        elif method == "iterative":
            # Richardson iteration for matrices close to the identity
            n = matrix.shape[0]
            X = np.eye(n)  # Initial guess
            max_iter = 100

            for _ in range(max_iter):
                X_new = X @ (2 * np.eye(n) - matrix @ X)
                if np.allclose(X, X_new, atol=1e-10):
                    break
                X = X_new

            return X

        else:
            raise ValueError(f"Unknown inversion method: {method}")


__all__ = ["TensorData", "MatrixOperations", "TensorOperations", "SpatialLinearAlgebra"]
