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
        from .spatial_methods import SpatialMethods

        SpatialMethods._cells(cells)
        SpatialMethods._values(cells, values)
        if weight_type not in {"queen", "rook", "distance"}:
            raise ValueError("weight_type must be queen, rook or distance")
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
        if weight_type not in {"queen", "rook", "distance"}:
            raise ValueError("weight_type must be queen, rook or distance")
        weights = np.zeros((n, n))

        backend_name = self.dispatcher._resolve_backend_name("indexing", self.backend)
        backend: Any = self.dispatcher.get_backend(backend_name)
        if backend is None:
            raise ValueError(
                f"geo_infer_space: backend '{backend_name}' resolved but not "
                f"retrievable; cannot build weight matrix"
            )

        for i, cell_i in enumerate(cells):
            neighbor_set = set(backend.get_cell_neighbors(cell_i, k=1))
            if weight_type == "distance":
                neighbor_set.update(backend.get_cell_neighbors(cell_i, k=2))

            for j, cell_j in enumerate(cells):
                if i != j and cell_j in neighbor_set:
                    weights[i, j] = (
                        1.0 / backend.get_cell_distance(cell_i, cell_j)
                        if weight_type == "distance"
                        else 1.0
                    )

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
            Dictionary with G* statistics for each cell. Undefined scores are
            None with per-cell reasons; they never enter hotspot lists.
            Invalid input and backend/topology failures propagate.
        """
        from .spatial_methods import SpatialMethods

        cells = list(SpatialMethods._cells(cells, allow_empty=False))
        SpatialMethods._values(cells, values)
        SpatialMethods._integer(distance, "distance", minimum=1)
        if len(cells) != len(values):
            raise ValueError("Cells and values must have same length")

        n = len(values)
        values_arr = np.asarray(values, dtype=float)
        scale = float(np.max(np.abs(values_arr)))
        scaled = values_arr / scale if scale else values_arr
        scaled_mean = float(np.mean(scaled))
        scaled_std = float(np.std(scaled))
        mean = scaled_mean * scale
        std = scaled_std * scale

        logger.info(f"Calculating Getis-Ord G* for {n} observations")

        g_stars = {}
        undefined = {}
        hotspots = []
        coldspots = []

        backend_name = self.dispatcher._resolve_backend_name("indexing", self.backend)
        backend: Any = self.dispatcher.get_backend(backend_name)
        cell_values = dict(zip(cells, scaled))
        cell_set = set(cells)

        for i, cell in enumerate(cells):
            if n < 2 or scaled_std == 0:
                g_stars[cell] = None
                undefined[cell] = (
                    "insufficient_observations" if n < 2 else "zero_global_variance"
                )
                continue
            neighbors = set()
            for ring in range(1, distance + 1):
                neighbors.update(backend.get_cell_neighbors(cell, k=ring))
            neighborhood = [cell] + sorted((neighbors & cell_set) - {cell})
            w = len(neighborhood)
            if w == n:
                g_stars[cell] = None
                undefined[cell] = "neighborhood_covers_entire_domain"
                continue
            numerator = sum(cell_values[c] for c in neighborhood) - scaled_mean * w
            denominator = scaled_std * np.sqrt(w * (n - w) / (n - 1))
            g_star = float(numerator / denominator)
            g_stars[cell] = g_star

            if g_star > 1.96:
                hotspots.append(
                    {
                        "cell": cell,
                        "g_star": g_star,
                        "value": float(values_arr[i]),
                        "significance": "significant" if g_star > 2.58 else "moderate",
                    }
                )
            elif g_star < -1.96:
                coldspots.append(
                    {
                        "cell": cell,
                        "g_star": g_star,
                        "value": float(values_arr[i]),
                        "significance": "significant" if g_star < -2.58 else "moderate",
                    }
                )

        return {
            "g_stars": g_stars,
            "undefined": undefined,
            "hotspots": hotspots,
            "coldspots": coldspots,
            "num_hotspots": len(hotspots),
            "num_coldspots": len(coldspots),
            "distance": distance,
            "global_mean": float(mean),
            "global_std": float(std),
        }

    def nearest_neighbor_index(
        self, cells: list[str], *, study_area_km2: float | None = None
    ) -> dict[str, Any]:
        """Describe nearest geodesic centroid distances, optionally against CSR.

        A Clark-Evans ratio, standard error and normal approximation require an
        explicit positive study area in km². Without it only distances are
        returned. H3 grid steps cannot be compared with a metric density.
        Edge effects are not corrected; the supplied study window defines the
        homogeneous complete-spatial-randomness reference assumption.
        """
        from .spatial_methods import SpatialMethods

        SpatialMethods._cells(cells)
        if study_area_km2 is not None:
            SpatialMethods._number(study_area_km2, "study_area_km2")
            if study_area_km2 <= 0:
                raise ValueError("study_area_km2 must be positive")
        n = len(cells)
        if n < 2:
            return {"error": "Need at least 2 observations"}
        backend_name = self.dispatcher._resolve_backend_name("indexing", self.backend)
        backend = self.dispatcher.get_backend(backend_name)
        if backend is None:
            raise ValueError(f"Backend {backend_name!r} is unavailable")
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        coordinates = [backend.cell_to_latlng(cell) for cell in cells]
        nearest = []
        for i, (lat, lng) in enumerate(coordinates):
            distances = [
                geod.inv(lng, lat, other_lng, other_lat)[2] / 1000
                for j, (other_lat, other_lng) in enumerate(coordinates)
                if i != j
            ]
            nearest.append(min(distances))
        observed = float(np.mean(nearest))
        result = {
            "n": n,
            "observed_mean_distance": observed,
            "distance_unit": "km",
            "reference_tested": study_area_km2 is not None,
        }
        if study_area_km2 is None:
            return result
        density = n / study_area_km2
        expected = 0.5 / np.sqrt(density)
        se = 0.26136 / np.sqrt(n * density)
        ratio = observed / expected
        z_score = (observed - expected) / se
        from scipy import stats

        result.update(
            nni=float(ratio),
            expected_mean_distance=float(expected),
            z_score=float(z_score),
            p_value=float(2 * stats.norm.sf(abs(z_score))),
            study_area_km2=study_area_km2,
            edge_corrected=False,
            pattern="clustered"
            if ratio < 1
            else "dispersed"
            if ratio > 1
            else "random",
        )
        return result

    def calculate_summary_statistics(self, values: list[float]) -> dict[str, Any]:
        """
        Calculate comprehensive summary statistics for spatial data.

        Args:
            values: Numeric values to summarize

        Returns:
            Dictionary with summary statistics
        """
        from .spatial_methods import SpatialMethods

        SpatialMethods._values(values, values)
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
        cv = (std / abs(mean) * 100) if mean != 0 else 0.0

        # Skewness and kurtosis
        if n > 2 and variance > 0:
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
            Dictionary with VMR statistics. Fewer than two observations or a
            zero mean produce None statistics and an explicit undefined reason.
            The chi-square reference assumes independent Poisson counts with
            equal exposure; noninteger intensities receive only descriptive VMR.
        """
        from .spatial_methods import SpatialMethods

        SpatialMethods._values(values, values)
        if any(value < 0 for value in values):
            raise ValueError("Count/intensity values must be non-negative")
        values_arr = np.asarray(values, dtype=float)
        n = len(values_arr)
        with np.errstate(over="raise", invalid="raise"):
            mean = float(np.mean(values_arr)) if n else None
            variance = float(np.var(values_arr, ddof=1)) if n > 1 else None
        if n < 2 or mean == 0:
            return {
                "vmr": None,
                "variance": variance,
                "mean": mean,
                "chi_square": None,
                "df": n - 1 if n else None,
                "p_value": None,
                "pattern": None,
                "n": n,
                "reference_tested": False,
                "undefined": "insufficient_observations" if n < 2 else "zero_mean",
            }

        assert mean is not None and variance is not None
        vmr = variance / mean

        # Chi-square test
        reference_tested = bool(np.all(values_arr == np.floor(values_arr)))
        chi_sq = p_value = None
        if reference_tested:
            chi_sq = (n - 1) * vmr
            from scipy import stats

            # sf avoids cancellation in the upper tail.
            p_value = min(
                1.0,
                2 * min(stats.chi2.cdf(chi_sq, n - 1), stats.chi2.sf(chi_sq, n - 1)),
            )

        if vmr < 0.5:
            pattern = "highly underdispersed (regular/uniform)"
        elif vmr < 1.0:
            pattern = "underdispersed"
        elif vmr > 2.0:
            pattern = "highly overdispersed (clustered)"
        elif vmr > 1.0:
            pattern = "overdispersed"
        else:
            pattern = "random (Poisson)" if reference_tested else "variance equals mean"

        return {
            "vmr": float(vmr),
            "variance": float(variance),
            "mean": float(mean),
            "chi_square": float(chi_sq) if chi_sq is not None else None,
            "df": n - 1,
            "p_value": float(p_value) if p_value is not None else None,
            "pattern": pattern,
            "n": n,
            "reference_tested": reference_tested,
            "reference_assumption": "independent Poisson counts with equal exposure",
            "undefined": None,
        }

    def quadrat_count(
        self,
        cells: list[str],
        values: list[float] | None = None,
        quadrat_size: int = 2,
    ) -> dict[str, Any]:
        """
        Perform quadrat count analysis.

        Groups cells into observed parent quadrats and analyzes their count
        distribution. Unobserved quadrats are not invented as zero counts.
        Input and backend/topology failures propagate without partial output.

        Args:
            cells: List of spatial cell identifiers
            values: Optional values (counts) at each cell
            quadrat_size: Integer parent-resolution steps, from zero (retain
                exact cells) through the common input resolution.

        Returns:
            Dictionary with quadrat analysis results
        """
        from .spatial_methods import SpatialMethods
        import h3

        cells = list(SpatialMethods._cells(cells, allow_empty=False))
        SpatialMethods._integer(quadrat_size, "quadrat_size")
        resolution = h3.get_resolution(cells[0])
        if quadrat_size > resolution:
            raise ValueError("quadrat_size exceeds the common cell resolution")
        n = len(cells)

        if values is None:
            values = [1] * n  # Point count
        SpatialMethods._values(cells, values)
        if any(value < 0 for value in values):
            raise ValueError("Count/intensity values must be non-negative")

        logger.info(f"Performing quadrat count analysis for {n} cells")

        backend_name = self.dispatcher._resolve_backend_name("indexing", self.backend)
        backend: Any = self.dispatcher.get_backend(backend_name)
        parent_resolution = resolution - quadrat_size
        quadrat_counts: dict[str, float] = {}
        for cell, value in zip(cells, values):
            parent = backend.get_cell_parent(cell, parent_resolution)
            if (
                not isinstance(parent, str)
                or not h3.is_valid_cell(parent)
                or h3.int_to_str(h3.str_to_int(parent)) != parent
                or h3.get_resolution(parent) != parent_resolution
                or h3.cell_to_parent(cell, parent_resolution) != parent
            ):
                raise ValueError("Backend returned an invalid H3 parent quadrat")
            quadrat_counts[parent] = quadrat_counts.get(parent, 0.0) + float(value)

        counts = list(quadrat_counts.values())
        vmr_result = self.variance_mean_ratio(counts)
        return {
            "num_quadrats": len(quadrat_counts),
            "quadrat_size": quadrat_size,
            "parent_resolution": parent_resolution,
            "quadrat_counts": quadrat_counts,
            "scope": "observed_parent_quadrats",
            "counts": counts,
            "vmr": vmr_result["vmr"],
            "pattern": vmr_result["pattern"],
            "undefined": vmr_result["undefined"],
            "total_count": sum(counts),
            "mean_count": float(np.mean(counts)),
            "max_count": max(counts),
            "min_count": min(counts),
        }
