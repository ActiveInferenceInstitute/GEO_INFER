"""
Helper functions for GEO-INFER-SPM

This module provides utility functions for creating design matrices,
generating coordinates, and other common SPM analysis tasks.
"""

from typing import Any
from numbers import Real

import numpy as np
from scipy import stats

from ..models.data_models import DesignMatrix, SPMData
from .rng import SeedLike, resolve_rng


def create_design_matrix(
    data: SPMData,
    formula: str | None = None,
    factors: dict[str, list[str]] | None = None,
    covariates: list[str] | None = None,
    intercept: bool = True,
) -> DesignMatrix:
    """
    Create design matrix from SPMData and specification.

    Args:
        data: SPMData containing covariates
        formula: Formula string for design matrix (e.g., "y ~ x1 + x2 + factor1")
        factors: Dictionary of categorical factors and their levels
        covariates: List of continuous covariate names
        intercept: Whether to include intercept term

    Returns:
        DesignMatrix object

    Example:
        >>> # Simple design with intercept and one covariate
        >>> design = create_design_matrix(data, covariates=['elevation'])

        >>> # Design with categorical factor
        >>> design = create_design_matrix(data, factors={'season': ['winter', 'spring', 'summer']})
    """
    n_points = data.n_points

    if formula is not None:
        # Deferred: see docs/deferred_statistical_methods.md ("Formula parser").
        design_matrix, names = _parse_formula(formula, data, intercept)
    else:
        # Build from factors and covariates
        design_components = []
        names = []
        covariates_map = data.covariates or {}

        # Intercept
        if intercept:
            design_components.append(np.ones(n_points))
            names.append("intercept")

        # Covariates
        if covariates:
            for cov_name in covariates:
                if cov_name not in covariates_map:
                    raise ValueError(f"Covariate '{cov_name}' not found in data")
                design_components.append(covariates_map[cov_name])
                names.append(cov_name)

        # Factors (categorical variables)
        if factors:
            for factor_name, levels in factors.items():
                if factor_name in covariates_map:
                    # Convert categorical covariate to dummy variables
                    factor_values = covariates_map[factor_name]
                    dummy_matrix = _create_dummy_variables(factor_values, levels)
                    for i, level in enumerate(levels[:-1]):  # n-1 dummies
                        design_components.append(dummy_matrix[:, i])
                        names.append(f"{factor_name}_{level}")
                else:
                    raise ValueError(
                        f"Factor '{factor_name}' is not present in data.covariates"
                    )

        design_matrix = np.column_stack(design_components)

    return DesignMatrix(
        matrix=design_matrix, names=names, factors=factors, covariates=covariates
    )


def _parse_formula(
    formula: str, data: SPMData, intercept: bool
) -> tuple[np.ndarray, list[str]]:
    """Parse formula string to create design matrix (see register: "Formula parser")."""
    # Deferred: see docs/deferred_statistical_methods.md ("Formula parser").
    if formula.count("~") != 1:
        raise ValueError("Formula must contain exactly one '~' separator")

    response, predictors = formula.split("~", 1)
    if not response.strip() or not predictors.strip():
        raise ValueError("Formula needs a response and predictor terms")

    # Parse predictors
    terms = [term.strip() for term in predictors.split("+")]
    covariates_map = data.covariates or {}

    design_components = []
    names = []

    # Intercept
    if intercept and "0" not in terms:
        design_components.append(np.ones(data.n_points))
        names.append("intercept")

    for term in terms:
        term = term.strip()
        if term == "0":
            continue  # No intercept
        elif term in covariates_map:
            design_components.append(covariates_map[term])
            names.append(term)
        elif "*" in term:
            # Interaction term (see register: "Formula parser").
            var1, var2 = term.split("*", 1)
            var1, var2 = var1.strip(), var2.strip()
            if var1 not in covariates_map or var2 not in covariates_map:
                raise ValueError(f"Unknown interaction in formula: {term}")
            interaction = covariates_map[var1] * covariates_map[var2]
            design_components.append(interaction)
            names.append(f"{var1}:{var2}")
        else:
            raise ValueError(f"Unknown term in formula: {term}")

    if not design_components:
        raise ValueError("Formula must select at least one regressor")
    return np.column_stack(design_components), names


def _create_dummy_variables(values: np.ndarray, levels: list[str]) -> np.ndarray:
    """Create dummy variables for categorical factor."""
    n_points = len(values)
    n_levels = len(levels)
    if not levels or len(set(levels)) != n_levels:
        raise ValueError("Factor levels must be nonempty and unique")
    if np.asarray(values).ndim != 1:
        raise ValueError("Factor values must be one-dimensional")
    if any(value not in levels for value in values):
        raise ValueError("Factor values must belong to the declared levels")

    # Map string levels to indices
    level_to_idx = {level: i for i, level in enumerate(levels)}

    # Create dummy matrix (n-1 columns for n levels)
    dummy_matrix = np.zeros((n_points, n_levels - 1))

    for i, value in enumerate(values):
        if value in level_to_idx:
            level_idx = level_to_idx[value]
            if level_idx < n_levels - 1:  # Don't create dummy for last level
                dummy_matrix[i, level_idx] = 1

    return dummy_matrix


def generate_coordinates(
    grid_type: str = "regular",
    n_points: int = 100,
    bounds: tuple[float, float, float, float] | None = None,
    random_seed: SeedLike = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Generate synthetic coordinate arrays for testing and examples.

    Args:
        grid_type: Type of coordinate grid ('regular', 'random', 'clustered')
        n_points: Number of coordinate points to generate
        bounds: Spatial bounds (min_lon, max_lon, min_lat, max_lat)
        random_seed: Seed or generator for reproducible random/clustered grids.
            ``None`` draws fresh OS entropy without changing global random state.
        **kwargs: Additional parameters for grid generation

    Returns:
        Coordinate array of shape (n_points, 2)

    Example:
        >>> # Generate regular grid
        >>> coords = generate_coordinates('regular', n_points=100, bounds=(-180, 180, -90, 90))

        >>> # Generate random coordinates
        >>> coords = generate_coordinates('random', n_points=50)

        >>> # Reproducible random coordinates
        >>> coords = generate_coordinates('random', n_points=50, random_seed=7)
    """
    if (
        isinstance(n_points, (bool, np.bool_))
        or not isinstance(n_points, (int, np.integer))
        or n_points <= 0
    ):
        raise ValueError("n_points must be a positive integer")
    if grid_type not in {"regular", "random", "clustered"}:
        raise ValueError(f"Unknown grid type: {grid_type}")
    if bounds is None:
        bounds = (-180, 180, -90, 90)  # Global bounds

    bounds_array = np.asarray(bounds, dtype=float)
    if bounds_array.shape != (4,) or not np.isfinite(bounds_array).all():
        raise ValueError("bounds must contain four finite values")
    min_lon, max_lon, min_lat, max_lat = bounds_array
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError("bounds minima must not exceed maxima")
    if grid_type == "clustered":
        n_clusters = kwargs.get("n_clusters", 3)
        cluster_std = kwargs.get("cluster_std", 5.0)
        if (
            isinstance(n_clusters, (bool, np.bool_))
            or not isinstance(n_clusters, (int, np.integer))
            or n_clusters <= 0
        ):
            raise ValueError("n_clusters must be a positive integer")
        if (
            not isinstance(cluster_std, Real)
            or not np.isfinite(cluster_std)
            or cluster_std < 0
        ):
            raise ValueError("cluster_std must be finite and nonnegative")
    rng = resolve_rng(random_seed)

    if grid_type == "regular":
        # Create regular grid
        n_cols = int(np.sqrt(n_points))
        n_rows = (n_points + n_cols - 1) // n_cols  # Ceiling division

        lon_vals = np.linspace(min_lon, max_lon, n_cols)
        lat_vals = np.linspace(min_lat, max_lat, n_rows)

        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        coordinates = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])[:n_points]

    elif grid_type == "random":
        # Random coordinates within bounds
        lon_vals = rng.uniform(min_lon, max_lon, n_points)
        lat_vals = rng.uniform(min_lat, max_lat, n_points)
        coordinates = np.column_stack([lon_vals, lat_vals])

    elif grid_type == "clustered":
        # Generate clustered coordinates
        n_clusters = kwargs.get("n_clusters", 3)
        cluster_std = kwargs.get("cluster_std", 5.0)

        coordinates = np.zeros((n_points, 2))

        # Generate cluster centers
        cluster_centers_lon = rng.uniform(min_lon, max_lon, n_clusters)
        cluster_centers_lat = rng.uniform(min_lat, max_lat, n_clusters)

        points_per_cluster = n_points // n_clusters
        remaining_points = n_points % n_clusters

        idx = 0
        for cluster in range(n_clusters):
            cluster_size = points_per_cluster + (1 if cluster < remaining_points else 0)

            # Generate points around cluster center
            lon_points = rng.normal(
                cluster_centers_lon[cluster], cluster_std, cluster_size
            )
            lat_points = rng.normal(
                cluster_centers_lat[cluster], cluster_std, cluster_size
            )

            # Clip to bounds
            lon_points = np.clip(lon_points, min_lon, max_lon)
            lat_points = np.clip(lat_points, min_lat, max_lat)

            coordinates[idx : idx + cluster_size] = np.column_stack(
                [lon_points, lat_points]
            )
            idx += cluster_size

    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

    return coordinates


def generate_synthetic_data(
    coordinates: np.ndarray,
    effects: dict[str, Any] | None = None,
    noise_level: float = 0.1,
    temporal: bool = False,
    n_timepoints: int = 10,
    random_seed: SeedLike = None,
) -> SPMData:
    """
    Generate synthetic SPM data for testing and examples.

    Args:
        coordinates: Spatial coordinates
        effects: Dictionary specifying spatial effects to include
        noise_level: Standard deviation of noise
        temporal: Whether to include temporal dimension
        n_timepoints: Number of time points if temporal
        random_seed: Seed or generator for the noise draws; ``None`` (default)
            draws fresh OS entropy. See :func:`geo_infer_spm.utils.rng.resolve_rng`.

    Returns:
        SPMData with synthetic data

    Example:
        >>> coords = generate_coordinates('regular', 100)
        >>> data = generate_synthetic_data(coords, effects={'trend': 'north_south'})
    """
    rng = resolve_rng(random_seed)
    coordinates = np.asarray(coordinates, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not len(coordinates)
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("coordinates must be a nonempty finite (n_points, 2) array")
    n_points = len(coordinates)

    if effects is None:
        effects = {"intercept": 10, "trend": "east_west"}

    # Generate base signal
    signal = np.zeros(n_points)

    # Intercept
    if "intercept" in effects:
        signal += effects["intercept"]

    # Spatial trends
    if "trend" in effects:
        trend_type = effects["trend"]

        if trend_type == "east_west":
            # Linear trend from west to east
            lon_range = np.ptp(coordinates[:, 0]) or 1.0
            lon_norm = (coordinates[:, 0] - np.min(coordinates[:, 0])) / lon_range
            signal += 5 * lon_norm

        elif trend_type == "north_south":
            # Linear trend from south to north
            lat_range = np.ptp(coordinates[:, 1]) or 1.0
            lat_norm = (coordinates[:, 1] - np.min(coordinates[:, 1])) / lat_range
            signal += 5 * lat_norm

        elif trend_type == "radial":
            # Radial pattern from center
            center = np.mean(coordinates, axis=0)
            distances = np.linalg.norm(coordinates - center, axis=1)
            dist_norm = distances / (np.max(distances) or 1.0)
            signal += 5 * (1 - dist_norm)  # Higher values near center

    # Spatial clusters
    if "clusters" in effects:
        n_clusters = effects["clusters"].get("n_clusters", 3)
        cluster_effect = effects["clusters"].get("effect_size", 3.0)

        # Simple cluster generation
        for _ in range(n_clusters):
            # Random cluster center
            center_idx = rng.integers(0, n_points)
            center = coordinates[center_idx]

            # Points within cluster radius
            distances = np.linalg.norm(coordinates - center, axis=1)
            cluster_radius = np.percentile(distances, 10)  # 10th percentile distance

            cluster_mask = distances < cluster_radius
            signal[cluster_mask] += cluster_effect

    # Temporal component
    time_coords = None
    if temporal:
        time_coords = np.arange(n_timepoints)
        temporal_signal = np.zeros((n_timepoints, n_points))

        for t in range(n_timepoints):
            # Temporal evolution (e.g., linear trend over time)
            time_effect = 1 + 0.1 * t  # Increasing over time
            temporal_signal[t] = signal * time_effect

        # Add temporal noise
        temporal_noise = rng.normal(0, noise_level, (n_timepoints, n_points))
        data = temporal_signal + temporal_noise

        # Flatten for SPMData format
        data_flat = data.T  # (n_points, n_timepoints)

    else:
        # Spatial only
        noise = rng.normal(0, noise_level, n_points)
        data = signal + noise
        data_flat = data

    # Generate covariates
    signal_scale = np.std(signal) or 1.0
    elevation = 500 + 100 * (signal - np.mean(signal)) / signal_scale
    elevation += rng.normal(0, max(noise_level, 0.05) * 10, n_points)
    response_summary = data_flat.mean(axis=-1) if temporal else data_flat
    covariates = {
        "elevation": elevation,
        "temperature": response_summary
        + rng.normal(0, max(noise_level, 0.05), n_points),
    }

    # Create metadata
    metadata = {
        "synthetic": True,
        "effects": effects,
        "noise_level": noise_level,
        "temporal": temporal,
        "n_timepoints": n_timepoints if temporal else None,
        "generation_timestamp": str(np.datetime64("now")),
    }

    return SPMData(
        data=data_flat,
        coordinates=coordinates,
        time=time_coords,
        covariates=covariates,
        metadata=metadata,
        crs="EPSG:4326",
    )


def create_spatial_basis_functions(
    coordinates: np.ndarray,
    n_basis: int = 10,
    method: str = "gaussian",
    random_seed: SeedLike = None,
) -> np.ndarray:
    """
    Create spatial basis functions for modeling spatial variation.

    Args:
        coordinates: Spatial coordinates (n_points, 2)
        n_basis: Number of basis functions
        method: Basis function method ('gaussian', 'polynomial', 'fourier').
            Polynomial columns follow increasing total degree, starting with
            the intercept, latitude, longitude, then quadratic terms.
        random_seed: Optional seed for reproducible Gaussian center selection.
            When ``None``, center selection uses a fresh entropy-backed
            generator without mutating the global ``np.random`` state.

    Returns:
        Basis function matrix (n_points, n_basis)
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not len(coordinates)
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("coordinates must be a nonempty finite (n_points, 2) array")
    if (
        isinstance(n_basis, (bool, np.bool_))
        or not isinstance(n_basis, (int, np.integer))
        or n_basis <= 0
    ):
        raise ValueError("n_basis must be a positive integer")
    n_points = len(coordinates)

    if method == "gaussian":
        # Gaussian radial basis functions
        rng = resolve_rng(random_seed)
        center_indices = rng.choice(
            n_points, size=min(n_basis, n_points), replace=False
        )
        centers = coordinates[center_indices]

        # Width based on median distance
        distances = np.linalg.norm(
            coordinates[:, np.newaxis] - centers[np.newaxis, :], axis=2
        )
        median_dist = np.median(distances)
        width = (median_dist or 1.0) / np.sqrt(n_basis)

        basis = np.zeros((n_points, n_basis))
        for i in range(n_basis):
            distances_to_center = np.linalg.norm(
                coordinates - centers[i % len(centers)], axis=1
            )
            basis[:, i] = np.exp(-(distances_to_center**2) / (2 * width**2))

    elif method == "polynomial":
        # Polynomial basis functions
        lon, lat = coordinates[:, 0], coordinates[:, 1]

        # Normalize coordinates
        lon_norm = (lon - np.mean(lon)) / (np.std(lon) or 1.0)
        lat_norm = (lat - np.mean(lat)) / (np.std(lat) or 1.0)

        basis_list = [np.ones(n_points)]  # Constant

        degree = 1
        while len(basis_list) < n_basis:
            for i in range(degree + 1):
                j = degree - i
                basis_list.append(lon_norm**i * lat_norm**j)
                if len(basis_list) == n_basis:
                    break

            degree += 1

        basis = np.column_stack(basis_list[:n_basis])

    elif method == "fourier":
        # Fourier basis functions
        lon_rad = np.radians(coordinates[:, 0])
        lat_rad = np.radians(coordinates[:, 1])

        basis_list = [np.ones(n_points)]  # Constant

        max_freq = int(np.sqrt(n_basis)) + 1
        for freq_lon in range(max_freq):
            for freq_lat in range(max_freq):
                if len(basis_list) >= n_basis:
                    break
                if freq_lon == 0 and freq_lat == 0:
                    continue  # Already added constant

                basis_list.extend(
                    [
                        np.cos(freq_lon * lon_rad) * np.cos(freq_lat * lat_rad),
                        np.sin(freq_lon * lon_rad) * np.cos(freq_lat * lat_rad),
                    ]
                )

            if len(basis_list) >= n_basis:
                break

        basis = np.column_stack(basis_list[:n_basis])

    else:
        raise ValueError(f"Unknown basis method: {method}")

    return basis


def compute_power_analysis(
    effect_size: float,
    n_points: int,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    random_seed: SeedLike = None,
) -> dict[str, Any]:
    """
    Perform power analysis for SPM statistical tests.

    Args:
        effect_size: Expected effect size
        n_points: Number of spatial/temporal points
        alpha: Significance level
        n_simulations: Number of simulation runs
        random_seed: Seed or generator for the simulations; ``None`` (default)
            draws fresh OS entropy. See :func:`geo_infer_spm.utils.rng.resolve_rng`.

    Returns:
        Dictionary with power analysis results
    """
    # Deferred: see docs/deferred_statistical_methods.md
    # ("Power analysis with spatial autocorrelation").
    rng = resolve_rng(random_seed)

    # Degrees of freedom for one-sample t-test
    df = n_points - 1

    # Critical t-value
    t_critical = stats.t.ppf(1 - alpha / 2, df)

    # Power calculation: one-sample t-test for mean shift
    power_values = []

    for _ in range(n_simulations):
        # Simulate data with effect (mean = effect_size, std = 1)
        data = rng.normal(effect_size, 1.0, n_points)

        # One-sample t-test: t = mean(data) / (std(data) / sqrt(n))
        t_stat = np.mean(data) / (np.std(data, ddof=1) / np.sqrt(n_points))
        power_values.append(abs(t_stat) > t_critical)

    power = np.mean(power_values)

    return {
        "power": power,
        "effect_size": effect_size,
        "n_points": n_points,
        "alpha": alpha,
        "t_critical": t_critical,
        "n_simulations": n_simulations,
    }
