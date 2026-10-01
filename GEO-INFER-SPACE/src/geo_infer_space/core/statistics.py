"""
Spatial Statistics Module for GEO-INFER-SPACE.

Provides statistical methods for spatial analysis including spatial autocorrelation,
point pattern analysis, and clustering statistics.
"""

import logging
from typing import Any, TYPE_CHECKING, cast
import numpy as np

if TYPE_CHECKING:
    from .dispatcher import SpatialBackendDispatcher

logger = logging.getLogger(__name__)


class SpatialStatistics:
    """
    Comprehensive spatial statistics for geospatial analysis.

    Provides methods for calculating spatial autocorrelation, clustering
    indices, and pattern detection statistics.
    """

    def __init__(self, backend: str | None = None) -> None:
        """Initialize spatial statistics with optional backend."""
        self.backend = backend
        self._dispatcher: SpatialBackendDispatcher | None = None

    @property
    def dispatcher(self) -> "SpatialBackendDispatcher":
        """Lazy load the dispatcher."""
        if self._dispatcher is None:
            from .dispatcher import get_backend_dispatcher

            self._dispatcher = get_backend_dispatcher()
        return self._dispatcher

    def moran_i(
        self, cells: list[str], values: list[float], weight_type: str = "queen"
    ) -> dict[str, Any]:
        """
        Calculate Moran's I spatial autocorrelation coefficient.

        Moran's I measures the degree to which similar values cluster together
        in space. Values range from -1 (dispersed) to +1 (clustered).

        Args:
            cells: List of spatial cell identifiers
            values: Numeric values at each cell location
            weight_type: Weight matrix type ('queen', 'rook', 'distance')

        Returns:
            Dictionary with:
                - moran_i: The Moran's I statistic
                - expected_i: Expected value under null hypothesis
                - variance: Variance of I
                - z_score: Standardized z-score
                - p_value: Two-tailed p-value
                - interpretation: Text interpretation
        """
        if len(cells) != len(values):
            raise ValueError(
                f"Cells ({len(cells)}) and values ({len(values)}) must have same length"
            )

        n = len(values)
        if n < 3:
            return {
                "moran_i": None,
                "error": "Need at least 3 observations for Moran's I",
            }

        logger.info(
            f"Calculating Moran's I for {n} observations with {weight_type} weights"
        )

        values_arr = np.array(values)
        mean = np.mean(values_arr)
        deviations = values_arr - mean

        # Build weight matrix based on cell adjacency
        weights = self._build_weight_matrix(cells, weight_type)

        # Calculate Moran's I
        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * deviations[i] * deviations[j]

        denominator = np.sum(deviations**2)
        total_weight = np.sum(weights)

        if denominator == 0 or total_weight == 0:
            return {"moran_i": 0.0, "error": "Zero variance or no spatial weights"}

        moran_i = (n / total_weight) * (numerator / denominator)

        # Expected value under null hypothesis
        expected_i = -1.0 / (n - 1)

        # Variance calculation (randomization assumption, Cliff & Ord 1973).
        # S2 = sum over locations of (row sum_i + column sum_i)^2, matching
        # geo_infer_math.core.spatial_statistics.morans_i_variance.
        s1 = 0.5 * np.sum((weights + weights.T) ** 2)
        s2 = np.sum((np.sum(weights, axis=1) + np.sum(weights, axis=0)) ** 2)
        s0 = total_weight

        denominator = (n - 1) * (n - 2) * (n - 3) * s0**2
        if denominator == 0:
            return {
                "moran_i": float(moran_i),
                "expected_i": float(expected_i),
                "error": (
                    "geo_infer_space: Moran's I randomization variance is "
                    f"undefined for n={n} observations (need n >= 4)"
                ),
            }

        b2 = n * np.sum(deviations**4) / np.sum(deviations**2) ** 2

        variance = (
            n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * s0**2)
            - b2 * (n * (n - 1) * s1 - 2 * n * s2 + 6 * s0**2)
        ) / denominator - expected_i**2

        if variance <= 0:
            return {
                "moran_i": float(moran_i),
                "expected_i": float(expected_i),
                "error": (
                    "geo_infer_space: non-positive Moran's I variance "
                    f"({variance:.3e}); z-score and p-value are undefined"
                ),
            }

        z_score = (moran_i - expected_i) / np.sqrt(variance)

        # Two-tailed p-value using normal approximation
        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Interpretation
        if p_value > 0.05:
            interpretation = "No significant spatial autocorrelation (random pattern)"
        elif moran_i > 0:
            interpretation = f"Significant positive spatial autocorrelation (clustered pattern, p={p_value:.4f})"
        else:
            interpretation = f"Significant negative spatial autocorrelation (dispersed pattern, p={p_value:.4f})"

        return {
            "moran_i": float(moran_i),
            "expected_i": float(expected_i),
            "variance": float(variance),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "interpretation": interpretation,
            "n": n,
            "weight_type": weight_type,
        }

    def _build_weight_matrix(self, cells: list[str], weight_type: str) -> np.ndarray:
        """Build spatial weight matrix based on cell adjacency.

        Raises:
            ValueError: When no backend can be resolved for the weight
                construction; no fully-connected fallback matrix is ever
                fabricated.
        """
        n = len(cells)
        weights = np.zeros((n, n))

        backend_name = self.dispatcher._resolve_backend_name("indexing", self.backend)
        backend: Any = self.dispatcher.get_backend(backend_name)
        if backend is None:
            raise ValueError(
                f"geo_infer_space: backend '{backend_name}' resolved but not "
                f"retrievable; cannot build weight matrix"
            )

        for i, cell_i in enumerate(cells):
            try:
                if weight_type == "queen" or weight_type == "rook":
                    neighbors = backend.get_cell_neighbors(cell_i, k=1)
                else:  # distance
                    neighbors = backend.get_cell_neighbors(cell_i, k=2)
            except Exception as e:
                logger.warning(
                    f"Neighbor lookup failed for {cell_i}; treating it as "
                    f"isolated — Moran's I may be degraded: {e}"
                )
                continue

            neighbor_set = set(neighbors)

            for j, cell_j in enumerate(cells):
                if i != j and cell_j in neighbor_set:
                    weights[i, j] = 1.0

        # Row-standardize
        row_sums = np.sum(weights, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums

        return cast(np.ndarray, weights)

    def getis_ord_g(
        self, cells: list[str], values: list[float], distance: int = 1
    ) -> dict[str, Any]:
        """
        Calculate Getis-Ord G* statistic for hot/cold spot analysis.

        G* identifies statistically significant hot spots (high values clustered)
        and cold spots (low values clustered).

        Args:
            cells: List of spatial cell identifiers
            values: Numeric values at each cell location
            distance: Neighborhood distance in grid steps

        Returns:
            Dictionary with G* statistics for each cell
        """
        if len(cells) != len(values):
            raise ValueError("Cells and values must have same length")

        n = len(values)
        values_arr = np.array(values)
        mean = np.mean(values_arr)
        std = np.std(values_arr)

        logger.info(f"Calculating Getis-Ord G* for {n} observations")

        g_stars = {}
        hotspots = []
        coldspots = []

        try:
            backend_name = self.dispatcher._resolve_backend_name(
                "indexing", self.backend
            )
            backend: Any = self.dispatcher.get_backend(backend_name)
            cell_values = dict(zip(cells, values))
            cell_set = set(cells)

            for i, cell in enumerate(cells):
                try:
                    neighbors = backend.get_cell_neighbors(cell, k=distance)
                    neighborhood = [cell] + [n for n in neighbors if n in cell_set]
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"Neighbor lookup failed for {cell}; using self-only "
                        f"neighborhood — local statistics may be degraded: {e}"
                    )
                    neighborhood = [cell]

                # Get values in neighborhood
                neighbor_values = [
                    cell_values[c] for c in neighborhood if c in cell_values
                ]

                if not neighbor_values:
                    continue

                w = len(neighbor_values)
                local_sum = sum(neighbor_values)

                # G* statistic
                numerator = local_sum - mean * w
                s = std * np.sqrt((n * w - w**2) / (n - 1)) if n > 1 else 1.0

                g_star = numerator / (s + 1e-10)
                g_stars[cell] = float(g_star)

                # Classify
                if g_star > 1.96:
                    hotspots.append(
                        {
                            "cell": cell,
                            "g_star": float(g_star),
                            "value": values_arr[i],
                            "significance": "significant"
                            if g_star > 2.58
                            else "moderate",
                        }
                    )
                elif g_star < -1.96:
                    coldspots.append(
                        {
                            "cell": cell,
                            "g_star": float(g_star),
                            "value": values_arr[i],
                            "significance": "significant"
                            if g_star < -2.58
                            else "moderate",
                        }
                    )

        except Exception as e:
            logger.error(f"G* calculation failed: {e}")
            return {"error": str(e)}

        return {
            "g_stars": g_stars,
            "hotspots": hotspots,
            "coldspots": coldspots,
            "num_hotspots": len(hotspots),
            "num_coldspots": len(coldspots),
            "distance": distance,
            "global_mean": float(mean),
            "global_std": float(std),
        }

    def nearest_neighbor_index(self, cells: list[str]) -> dict[str, Any]:
        """
        Calculate Nearest Neighbor Index for point pattern analysis.

        NNI < 1 indicates clustering, NNI > 1 indicates dispersion,
        NNI = 1 indicates random distribution.

        Args:
            cells: List of spatial cell identifiers

        Returns:
            Dictionary with NNI statistics
        """
        n = len(cells)
        if n < 2:
            return {"error": "Need at least 2 observations"}

        logger.info(f"Calculating Nearest Neighbor Index for {n} cells")

        try:
            backend_name = self.dispatcher._resolve_backend_name(
                "indexing", self.backend
            )
            backend: Any = self.dispatcher.get_backend(backend_name)

            total_distance: float = 0
            valid_pairs = 0

            for i, cell in enumerate(cells):
                min_dist = float("inf")
                for j, other_cell in enumerate(cells):
                    if i == j:
                        continue
                    try:
                        dist = backend.get_cell_distance(cell, other_cell)
                        min_dist = min(min_dist, dist)
                    except Exception:
                        continue

                if min_dist < float("inf"):
                    total_distance += min_dist
                    valid_pairs += 1

            if valid_pairs == 0:
                return {"error": "Could not calculate distances"}

            observed_mean_distance = total_distance / valid_pairs

            # Expected mean distance for random pattern
            # Using simplified formula: E(d) = 0.5 / sqrt(density)
            # For H3, we approximate based on number of cells
            expected_mean_distance = 0.5 * np.sqrt(n / valid_pairs)

            nni = (
                observed_mean_distance / expected_mean_distance
                if expected_mean_distance > 0
                else 1.0
            )

            # Z-score
            se = 0.26136 / np.sqrt(n * (n / valid_pairs))
            z_score = (observed_mean_distance - expected_mean_distance) / (se + 1e-10)

            from scipy import stats

            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            if nni < 0.5:
                pattern = "highly clustered"
            elif nni < 1.0:
                pattern = "clustered"
            elif nni > 1.5:
                pattern = "highly dispersed"
            elif nni > 1.0:
                pattern = "dispersed"
            else:
                pattern = "random"

            return {
                "nni": float(nni),
                "observed_mean_distance": float(observed_mean_distance),
                "expected_mean_distance": float(expected_mean_distance),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "pattern": pattern,
                "n": n,
            }

        except Exception as e:
            logger.error(f"NNI calculation failed: {e}")
            return {"error": str(e)}

    def calculate_summary_statistics(self, values: list[float]) -> dict[str, Any]:
        """
        Calculate comprehensive summary statistics for spatial data.

        Args:
            values: Numeric values to summarize

        Returns:
            Dictionary with summary statistics
        """
        values_arr = np.array(values)
        n = len(values_arr)

        if n == 0:
            return {"error": "No values provided"}

        from scipy import stats as scipy_stats

        mean = np.mean(values_arr)
        median = np.median(values_arr)
        std = np.std(values_arr, ddof=1) if n > 1 else 0.0
        variance = np.var(values_arr, ddof=1) if n > 1 else 0.0

        # Coefficient of variation
        cv = (std / mean * 100) if mean != 0 else 0.0

        # Skewness and kurtosis
        if n > 2:
            skewness = scipy_stats.skew(values_arr)
            kurtosis = scipy_stats.kurtosis(values_arr)
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Quartiles
        q1, q2, q3 = np.percentile(values_arr, [25, 50, 75])
        iqr = q3 - q1

        # Range
        min_val = np.min(values_arr)
        max_val = np.max(values_arr)
        range_val = max_val - min_val

        return {
            "n": n,
            "mean": float(mean),
            "median": float(median),
            "std": float(std),
            "variance": float(variance),
            "cv": float(cv),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "min": float(min_val),
            "max": float(max_val),
            "range": float(range_val),
            "q1": float(q1),
            "q2": float(q2),
            "q3": float(q3),
            "iqr": float(iqr),
        }

    def variance_mean_ratio(self, values: list[float]) -> dict[str, Any]:
        """
        Calculate Variance-to-Mean Ratio (Index of Dispersion).

        VMR < 1 indicates underdispersion (more uniform than random)
        VMR = 1 indicates random Poisson distribution
        VMR > 1 indicates overdispersion (clustering)

        Args:
            values: Count or intensity values

        Returns:
            Dictionary with VMR statistics
        """
        values_arr = np.array(values)
        n = len(values_arr)

        if n == 0:
            return {"error": "No values provided"}

        mean = np.mean(values_arr)
        variance = np.var(values_arr, ddof=1) if n > 1 else 0.0

        vmr = variance / mean if mean > 0 else 0.0

        # Chi-square test
        chi_sq = (n - 1) * vmr
        from scipy import stats

        # Two-tailed p-value
        p_lower = stats.chi2.cdf(chi_sq, n - 1)
        p_upper = 1 - p_lower
        p_value = 2 * min(p_lower, p_upper)

        if vmr < 0.5:
            pattern = "highly underdispersed (regular/uniform)"
        elif vmr < 1.0:
            pattern = "underdispersed"
        elif vmr > 2.0:
            pattern = "highly overdispersed (clustered)"
        elif vmr > 1.0:
            pattern = "overdispersed"
        else:
            pattern = "random (Poisson)"

        return {
            "vmr": float(vmr),
            "variance": float(variance),
            "mean": float(mean),
            "chi_square": float(chi_sq),
            "df": n - 1,
            "p_value": float(p_value),
            "pattern": pattern,
            "n": n,
        }

    def quadrat_count(
        self,
        cells: list[str],
        values: list[float] | None = None,
        quadrat_size: int = 2,
    ) -> dict[str, Any]:
        """
        Perform quadrat count analysis.

        Groups cells into larger quadrats and analyzes count distribution.

        Args:
            cells: List of spatial cell identifiers
            values: Optional values (counts) at each cell
            quadrat_size: Size of quadrats in resolution steps

        Returns:
            Dictionary with quadrat analysis results
        """
        n = len(cells)
        if n == 0:
            return {"error": "No cells provided"}

        if values is None:
            values = [1] * n  # Point count

        logger.info(f"Performing quadrat count analysis for {n} cells")

        try:
            backend_name = self.dispatcher._resolve_backend_name(
                "indexing", self.backend
            )
            backend: Any = self.dispatcher.get_backend(backend_name)

            # Get parent cells as quadrats
            quadrat_counts: dict[str, float] = {}

            for cell, value in zip(cells, values):
                try:
                    current_res = backend.get_cell_resolution(cell)
                    parent_res = max(0, current_res - quadrat_size)
                    parent = backend.get_cell_parent(cell, parent_res)

                    if parent not in quadrat_counts:
                        quadrat_counts[parent] = 0
                    quadrat_counts[parent] += value
                except Exception:
                    continue

            counts = list(quadrat_counts.values())

            if not counts:
                return {"error": "Could not create quadrats"}

            # Calculate VMR for quadrats
            vmr_result = self.variance_mean_ratio(counts)

            return {
                "num_quadrats": len(quadrat_counts),
                "quadrat_size": quadrat_size,
                "counts": counts,
                "vmr": vmr_result.get("vmr", 0),
                "pattern": vmr_result.get("pattern", "unknown"),
                "total_count": sum(counts),
                "mean_count": float(np.mean(counts)),
                "max_count": max(counts),
                "min_count": min(counts),
            }

        except Exception as e:
            logger.error(f"Quadrat count failed: {e}")
            return {"error": str(e)}
