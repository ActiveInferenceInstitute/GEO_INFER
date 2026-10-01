"""Random Field Theory inference for Statistical Parametric Mapping.

The implementation uses the Gaussian kinematic formula for one-, two-, and
three-dimensional rectangular search regions. Peak-level family-wise error
(FWE) probabilities include every Euler-characteristic (EC) density and the
corresponding lower-dimensional boundary resels. Cluster-level inference uses
the Poisson clumping approximation for the maximum topological cluster extent.

For a Gaussian field at threshold ``u``, the expected EC is

``E[chi(A_u)] = sum(R_d * rho_d(u), d=0..D)``,

where ``R_d`` are resel counts (Lipschitz--Killing curvatures in FWHM units)
and ``rho_d`` are Gaussian EC densities. At a sufficiently high threshold,
the EC is the number of connected excursion components. Conditional cluster
extent survival is approximated by ``exp(-beta * k ** (2 / D))`` and the
search-volume FWE probability follows from a Poisson maximum-cluster model.
"""

from __future__ import annotations

from itertools import combinations
from typing import Union

import numpy as np
from scipy import ndimage
from scipy.special import eval_hermitenorm, gamma
from scipy.stats import f, norm, t

from ..models.data_models import ContrastResult, SPMResult

ScalarOrArray = Union[float, np.ndarray]


class RandomFieldTheory:
    """Random Field Theory multiple-comparison correction for SPM fields.

    Parameters
    ----------
    field_shape:
        Shape of the complete rectangular search region. One-, two-, and
        three-dimensional fields are supported.
    smoothness:
        FWHM smoothness in the physical units of each field dimension.
    search_volume:
        Optional top-dimensional search volume in resels. When supplied
        directly, lower-dimensional resel counts are inferred from the field's
        aspect ratio. :meth:`compute_search_volume` calculates the complete
        resel vector from field geometry and should normally be preferred.
    df:
        Residual degrees of freedom for t fields or denominator degrees of
        freedom for F fields (whose numerator degrees of freedom are one).
    """

    def __init__(
        self,
        field_shape: tuple[int, ...],
        smoothness: np.ndarray | None = None,
        search_volume: float | None = None,
        df: int | None = None,
    ):
        self.field_shape = tuple(field_shape)
        self.ndim = len(self.field_shape)
        self.smoothness = (
            None if smoothness is None else np.asarray(smoothness, dtype=float)
        )
        self.search_volume = None if search_volume is None else float(search_volume)
        self.df = df
        self.resel_counts: np.ndarray | None = None
        self._voxel_sizes = np.ones(self.ndim, dtype=float)

        # Kept as a compatibility attribute for callers that inspected the old
        # implementation. EC calculations use ec_densities() directly.
        self._rft_constants = {
            1: np.sqrt(4 * np.log(2)),
            2: 4 * np.log(2),
            3: (4 * np.log(2)) ** (3 / 2),
        }

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate field geometry and optional inference parameters."""
        if self.ndim not in (1, 2, 3):
            raise ValueError(f"RFT supports 1D, 2D, and 3D fields, got {self.ndim}D")
        if any(
            not isinstance(size, (int, np.integer)) or int(size) <= 0
            for size in self.field_shape
        ):
            raise ValueError("Field dimensions must be positive integers")
        if self.smoothness is not None:
            if self.smoothness.shape != (self.ndim,):
                raise ValueError(f"Smoothness must have {self.ndim} dimensions")
            if not np.all(np.isfinite(self.smoothness)) or np.any(self.smoothness <= 0):
                raise ValueError("Smoothness values must be finite and positive")
        if self.search_volume is not None and (
            not np.isfinite(self.search_volume) or self.search_volume <= 0
        ):
            raise ValueError("Search volume must be finite and positive")
        if self.df is not None and self.df <= 0:
            raise ValueError("Degrees of freedom must be positive")

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        return alpha

    @staticmethod
    def _normalise_stat_type(stat_type: str) -> str:
        normalised = stat_type.upper()
        if normalised not in {"T", "F", "Z"}:
            raise ValueError(f"Unknown statistic type: {stat_type}")
        return normalised

    def estimate_smoothness(
        self, residuals: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Estimate axis-wise FWHM from finite differences of residuals.

        For a unit-variance stationary Gaussian field, the derivative variance
        is estimated by ``Var(diff) / Var(field)``. The corresponding FWHM is
        ``sqrt(4 log(2) / lambda)``. Only adjacent pairs that are finite and
        inside ``mask`` contribute to an axis estimate.
        """
        residual_field = np.asarray(residuals, dtype=float)
        if residual_field.size != int(np.prod(self.field_shape)):
            raise ValueError("Residuals must contain one value per field location")
        residual_field = residual_field.reshape(self.field_shape)

        if mask is None:
            valid = np.ones(self.field_shape, dtype=bool)
        else:
            valid = np.asarray(mask, dtype=bool)
            if valid.size != residual_field.size:
                raise ValueError("Mask must contain one value per field location")
            valid = valid.reshape(self.field_shape)
        valid &= np.isfinite(residual_field)
        if np.count_nonzero(valid) < 2:
            raise ValueError("At least two finite residuals are required")

        residual_variance = float(np.var(residual_field[valid]))
        fwhm = np.ones(self.ndim, dtype=float)
        if residual_variance <= 0:
            self.smoothness = fwhm
            self.resel_counts = None
            return fwhm

        for axis in range(self.ndim):
            lower = [slice(None)] * self.ndim
            upper = [slice(None)] * self.ndim
            lower[axis] = slice(None, -1)
            upper[axis] = slice(1, None)
            pair_mask = valid[tuple(lower)] & valid[tuple(upper)]
            differences = np.diff(residual_field, axis=axis)[pair_mask]
            if differences.size:
                derivative_variance = float(np.var(differences))
                if derivative_variance > 0:
                    fwhm[axis] = np.sqrt(
                        4 * np.log(2) * residual_variance / derivative_variance
                    )

        self.smoothness = fwhm
        self.resel_counts = None
        return fwhm

    @staticmethod
    def _box_resel_counts(resel_edges: np.ndarray) -> np.ndarray:
        """Return intrinsic volumes of a box from its resel edge lengths."""
        ndim = len(resel_edges)
        counts = np.empty(ndim + 1, dtype=float)
        counts[0] = 1.0
        for order in range(1, ndim + 1):
            counts[order] = sum(
                float(np.prod(resel_edges[list(indices)]))
                for indices in combinations(range(ndim), order)
            )
        return counts

    def compute_resel_counts(self, voxel_sizes: np.ndarray | None = None) -> np.ndarray:
        """Compute all zero- through D-dimensional resel counts.

        A rectangular field with FWHM-normalised side lengths ``q_i`` has resel
        counts equal to the elementary symmetric sums of those lengths. This
        includes ``R_0 = 1`` (the Euler characteristic), boundary resels, and
        the top-dimensional search volume ``R_D``.
        """
        if self.smoothness is None:
            raise ValueError(
                "Smoothness must be estimated before computing resel counts"
            )
        sizes = (
            np.ones(self.ndim, dtype=float)
            if voxel_sizes is None
            else np.asarray(voxel_sizes, dtype=float)
        )
        if sizes.shape != (self.ndim,):
            raise ValueError(f"Voxel sizes must have {self.ndim} dimensions")
        if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
            raise ValueError("Voxel sizes must be finite and positive")

        self._voxel_sizes = sizes
        physical_edges = np.asarray(self.field_shape, dtype=float) * sizes
        resel_edges = physical_edges / self.smoothness
        counts = self._box_resel_counts(resel_edges)
        self.resel_counts = counts
        self.search_volume = float(counts[-1])
        return counts.copy()

    def compute_search_volume(self, voxel_sizes: np.ndarray | None = None) -> float:
        """Compute and store the top-dimensional search volume in resels."""
        return float(self.compute_resel_counts(voxel_sizes)[-1])

    def _resolved_resel_counts(self) -> np.ndarray:
        """Resolve a complete resel vector while honoring an explicit volume."""
        if self.search_volume is None:
            raise ValueError("Search volume must be computed first")

        if self.smoothness is None:
            edges = np.asarray(self.field_shape, dtype=float)
        else:
            edges = (
                np.asarray(self.field_shape, dtype=float)
                * self._voxel_sizes
                / self.smoothness
            )

        raw_volume = float(np.prod(edges))
        scale = (self.search_volume / raw_volume) ** (1 / self.ndim)
        counts = self._box_resel_counts(edges * scale)
        # Avoid round-off drift in the user-visible top-dimensional count.
        counts[-1] = self.search_volume
        self.resel_counts = counts
        return counts

    def _stat_to_z(self, value: float, stat_type: str) -> float:
        """Convert a one-sided statistic height to a Gaussian Z height."""
        kind = self._normalise_stat_type(stat_type)
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Statistic thresholds must be finite")
        if kind == "Z" or (kind == "T" and self.df is None):
            return value
        if kind == "T":
            probability = float(t.sf(value, self.df))
        else:
            if value < 0:
                raise ValueError("F-statistic thresholds must be non-negative")
            probability = float(f.sf(value, 1, self.df or 100))
        probability = float(
            np.clip(probability, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        )
        return float(norm.isf(probability))

    def _z_to_stat(self, z_value: float, stat_type: str) -> float:
        """Convert a Gaussian Z height to the requested statistic scale."""
        kind = self._normalise_stat_type(stat_type)
        if kind == "Z" or (kind == "T" and self.df is None):
            return float(z_value)
        probability = float(norm.sf(z_value))
        if kind == "T":
            return float(t.isf(probability, self.df))
        return float(f.isf(probability, 1, self.df or 100))

    def ec_densities(self, threshold: float, stat_type: str = "Z") -> np.ndarray:
        """Return all Gaussian EC densities ``rho_0`` through ``rho_D``.

        Statistic fields other than Z are transformed to an equivalent
        one-sided Gaussian height before applying the Gaussian densities.
        """
        z_value = self._stat_to_z(threshold, stat_type)
        densities = np.empty(self.ndim + 1, dtype=float)
        densities[0] = norm.sf(z_value)
        exponential = np.exp(-(z_value**2) / 2)
        for order in range(1, self.ndim + 1):
            densities[order] = (
                (4 * np.log(2)) ** (order / 2)
                * eval_hermitenorm(order - 1, z_value)
                * exponential
                / (2 * np.pi) ** ((order + 1) / 2)
            )
        return densities

    def expected_euler_characteristic(
        self, threshold: float, stat_type: str = "Z"
    ) -> float:
        """Return the full expected EC of the one-sided excursion set."""
        counts = self._resolved_resel_counts()
        return float(np.dot(counts, self.ec_densities(threshold, stat_type)))

    def expected_clusters(self, threshold: float, stat_type: str = "t") -> float:
        """Approximate the expected number of high-threshold clusters.

        At high excursion thresholds the Euler characteristic equals the
        number of connected components with high probability. The full EC,
        including boundary terms, therefore replaces the former
        top-dimensional-only approximation.
        """
        if self.search_volume is None:
            raise ValueError("Search volume must be computed first")
        if self.smoothness is None:
            raise ValueError("Smoothness must be estimated first")
        return max(0.0, self.expected_euler_characteristic(threshold, stat_type))

    def peak_fwe_p_value(
        self,
        threshold: float,
        stat_type: str = "Z",
        *,
        two_sided: bool = False,
    ) -> float:
        """Approximate the peak-level FWE tail probability at ``threshold``."""
        z_value = self._stat_to_z(threshold, stat_type)
        tail_factor = 2.0 if two_sided else 1.0
        expected_ec = tail_factor * max(
            0.0, self.expected_euler_characteristic(threshold, stat_type)
        )
        # A search-volume probability cannot be below its pointwise marginal
        # tail. This also protects low thresholds where the EC heuristic is
        # outside its asymptotic regime and odd-dimensional terms may be negative.
        pointwise_tail = tail_factor * norm.sf(z_value)
        return float(np.clip(max(expected_ec, pointwise_tail), 0.0, 1.0))

    def peak_threshold(
        self,
        alpha: float = 0.05,
        stat_type: str = "t",
        *,
        two_sided: bool = True,
    ) -> float:
        """Solve the full-EC peak-level FWE threshold numerically."""
        alpha = self._validate_alpha(alpha)
        if self.search_volume is None:
            raise ValueError("Search volume must be computed first")

        lower_z = 0.0
        upper_z = 8.0
        while (
            self.peak_fwe_p_value(
                self._z_to_stat(upper_z, stat_type),
                stat_type,
                two_sided=two_sided,
            )
            > alpha
        ):
            upper_z *= 2
            if upper_z > 64:
                raise RuntimeError("Unable to bracket the RFT peak threshold")

        for _ in range(80):
            middle_z = (lower_z + upper_z) / 2
            probability = self.peak_fwe_p_value(
                self._z_to_stat(middle_z, stat_type),
                stat_type,
                two_sided=two_sided,
            )
            if probability > alpha:
                lower_z = middle_z
            else:
                upper_z = middle_z
        return self._z_to_stat((lower_z + upper_z) / 2, stat_type)

    def cluster_threshold(
        self,
        alpha: float = 0.05,
        stat_type: str = "t",
        *,
        two_sided: bool = True,
    ) -> float:
        """Return the EC-based FWE height threshold for an excursion set.

        This compatibility method historically supplied the height used to
        form clusters. Cluster-extent inference normally uses a fixed
        uncorrected forming threshold instead; pass that threshold explicitly
        to :meth:`correct_p_values`.
        """
        if self.smoothness is None:
            raise ValueError("Smoothness must be estimated first")
        return self.peak_threshold(alpha, stat_type, two_sided=two_sided)

    def expected_excursion_volume(
        self, cluster_forming_threshold: float, stat_type: str = "Z"
    ) -> float:
        """Return expected one-sided suprathreshold volume in resels."""
        if self.search_volume is None:
            raise ValueError("Search volume must be computed first")
        z_value = self._stat_to_z(cluster_forming_threshold, stat_type)
        return float(self.search_volume * norm.sf(z_value))

    def _cluster_extent_parameters(
        self, cluster_forming_threshold: float, stat_type: str
    ) -> tuple[float, float, float]:
        """Return expected clusters, expected volume, and extent rate beta."""
        z_value = self._stat_to_z(cluster_forming_threshold, stat_type)
        if z_value <= 0:
            raise ValueError("Cluster-forming threshold must have positive Z height")
        expected_clusters = max(
            0.0,
            self.expected_euler_characteristic(cluster_forming_threshold, stat_type),
        )
        expected_volume = self.expected_excursion_volume(
            cluster_forming_threshold, stat_type
        )
        if expected_clusters <= 0 or expected_volume <= 0:
            raise ValueError(
                "Cluster-extent inference requires positive expected clusters and excursion volume"
            )
        mean_extent = expected_volume / expected_clusters
        beta = (gamma(self.ndim / 2 + 1) / mean_extent) ** (2 / self.ndim)
        return expected_clusters, expected_volume, float(beta)

    def _extent_in_resels(
        self, extent: ScalarOrArray, *, extent_in_resels: bool
    ) -> np.ndarray:
        values = np.asarray(extent, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Cluster extents must be finite and non-negative")
        if extent_in_resels:
            return values
        if self.search_volume is None:
            raise ValueError("Search volume must be computed first")
        resels_per_voxel = self.search_volume / float(np.prod(self.field_shape))
        return values * resels_per_voxel

    @staticmethod
    def _return_scalar_if_scalar(
        original: ScalarOrArray, values: np.ndarray
    ) -> ScalarOrArray:
        if np.ndim(original) == 0:
            return float(values)
        return values

    def cluster_extent_probability(
        self,
        extent: ScalarOrArray,
        cluster_forming_threshold: float,
        stat_type: str = "Z",
        *,
        extent_in_resels: bool = True,
    ) -> ScalarOrArray:
        """Return the uncorrected survival probability of a cluster extent.

        ``extent`` is measured in resels by default. Set
        ``extent_in_resels=False`` to supply a count of field voxels.
        """
        resel_extent = self._extent_in_resels(extent, extent_in_resels=extent_in_resels)
        _, _, beta = self._cluster_extent_parameters(
            cluster_forming_threshold, stat_type
        )
        survival = np.exp(-beta * resel_extent ** (2 / self.ndim))
        return self._return_scalar_if_scalar(extent, survival)

    def cluster_extent_p_value(
        self,
        extent: ScalarOrArray,
        cluster_forming_threshold: float,
        stat_type: str = "Z",
        *,
        extent_in_resels: bool = True,
        two_sided: bool = False,
    ) -> ScalarOrArray:
        """Return maximum-cluster FWE p-values for observed extents."""
        expected_clusters, _, _ = self._cluster_extent_parameters(
            cluster_forming_threshold, stat_type
        )
        survival = np.asarray(
            self.cluster_extent_probability(
                extent,
                cluster_forming_threshold,
                stat_type,
                extent_in_resels=extent_in_resels,
            ),
            dtype=float,
        )
        tail_factor = 2.0 if two_sided else 1.0
        corrected = -np.expm1(-tail_factor * expected_clusters * survival)
        corrected = np.clip(corrected, 0.0, 1.0)
        return self._return_scalar_if_scalar(extent, corrected)

    def cluster_extent_threshold(
        self,
        alpha: float,
        cluster_forming_threshold: float,
        stat_type: str = "Z",
        *,
        extent_in_resels: bool = True,
        two_sided: bool = False,
    ) -> float:
        """Return the minimum continuous extent whose cluster FWE is ``alpha``."""
        alpha = self._validate_alpha(alpha)
        expected_clusters, _, beta = self._cluster_extent_parameters(
            cluster_forming_threshold, stat_type
        )
        tail_factor = 2.0 if two_sided else 1.0
        target_survival = -np.log1p(-alpha) / (tail_factor * expected_clusters)
        if target_survival >= 1:
            extent_resels = 0.0
        else:
            extent_resels = (-np.log(target_survival) / beta) ** (self.ndim / 2)
        if extent_in_resels:
            return float(extent_resels)
        assert self.search_volume is not None
        resels_per_voxel = self.search_volume / float(np.prod(self.field_shape))
        return float(extent_resels / resels_per_voxel)

    def _label_and_correct_clusters(
        self,
        excursion_mask: np.ndarray,
        threshold: float,
        stat_type: str,
        corrected: np.ndarray,
        *,
        connectivity: int,
        two_sided: bool,
    ) -> None:
        structure = ndimage.generate_binary_structure(self.ndim, connectivity)
        labels, cluster_count = ndimage.label(excursion_mask, structure=structure)
        for cluster_id in range(1, cluster_count + 1):
            cluster_mask = labels == cluster_id
            extent_voxels = int(np.count_nonzero(cluster_mask))
            probability = self.cluster_extent_p_value(
                extent_voxels,
                threshold,
                stat_type,
                extent_in_resels=False,
                two_sided=two_sided,
            )
            corrected[cluster_mask] = probability

    def correct_p_values(
        self,
        statistical_map: np.ndarray,
        stat_type: str = "t",
        method: str = "cluster",
        *,
        cluster_forming_threshold: float | None = None,
        cluster_forming_alpha: float = 0.001,
        connectivity: int = 1,
        two_sided: bool | None = None,
    ) -> np.ndarray:
        """Apply peak-height or topological cluster-extent RFT correction.

        For ``method="peak"``, each location receives its full-EC peak FWE
        probability. For ``method="cluster"``, connected excursion components
        receive a shared maximum-cluster extent p-value and locations below the
        cluster-forming threshold receive one.

        By default, t and Z maps are treated as two-sided while F maps are
        one-sided. The default cluster-forming height is the one-sided
        uncorrected Gaussian tail specified by ``cluster_forming_alpha``.
        """
        kind = self._normalise_stat_type(stat_type)
        if kind == "T" and self.df is None:
            raise ValueError("Degrees of freedom required for t-statistics")
        if self.search_volume is None:
            raise ValueError("Search volume must be computed first")
        if two_sided is None:
            two_sided = kind != "F"

        original = np.asarray(statistical_map, dtype=float)
        if original.size != int(np.prod(self.field_shape)):
            raise ValueError(
                "Statistical map must contain one value per field location"
            )
        if not np.all(np.isfinite(original)):
            raise ValueError("Statistical map must contain only finite values")
        field = original.reshape(self.field_shape)
        method_name = method.lower()

        if method_name == "peak":
            heights = np.abs(field) if kind != "F" else field
            corrected = np.fromiter(
                (
                    self.peak_fwe_p_value(float(value), kind, two_sided=bool(two_sided))
                    for value in heights.flat
                ),
                dtype=float,
                count=heights.size,
            ).reshape(self.field_shape)
            return corrected.reshape(original.shape)

        if method_name != "cluster":
            raise ValueError(f"Unknown correction method: {method}")
        if not isinstance(connectivity, (int, np.integer)) or not (
            1 <= int(connectivity) <= self.ndim
        ):
            raise ValueError(f"connectivity must be between 1 and {self.ndim}")

        if cluster_forming_threshold is None:
            cluster_forming_alpha = self._validate_alpha(cluster_forming_alpha)
            cluster_forming_threshold = self._z_to_stat(
                norm.isf(cluster_forming_alpha), kind
            )
        threshold = float(cluster_forming_threshold)
        # Validate the forming threshold and the cluster distribution before
        # labeling so an invalid inference configuration fails closed.
        self._cluster_extent_parameters(threshold, kind)

        corrected = np.ones(self.field_shape, dtype=float)
        self._label_and_correct_clusters(
            field >= threshold,
            threshold,
            kind,
            corrected,
            connectivity=int(connectivity),
            two_sided=bool(two_sided),
        )
        if two_sided and kind != "F":
            self._label_and_correct_clusters(
                field <= -threshold,
                threshold,
                kind,
                corrected,
                connectivity=int(connectivity),
                two_sided=True,
            )
        return corrected.reshape(original.shape)


def compute_spm(
    model_result: SPMResult,
    contrast: ContrastResult,
    correction: str = "RFT",
    alpha: float = 0.05,
) -> ContrastResult:
    """Compute a statistical parametric map with multiple-comparison control."""
    if correction.upper() == "RFT":
        residuals = np.asarray(model_result.residuals)
        field_shape = (residuals.shape[0],)
        residual_df = max(1, field_shape[0] - model_result.design_matrix.n_regressors)

        rft = RandomFieldTheory(field_shape, df=residual_df)
        rft.estimate_smoothness(residuals)
        rft.compute_search_volume()
        statistic = np.asarray(contrast.t_statistic, dtype=float)
        if statistic.size == 1:
            # A scalar coefficient contrast has no connected-component
            # topology. Correct its height against the residual search region.
            corrected_p = np.asarray(
                rft.peak_fwe_p_value(
                    float(np.abs(statistic).item()),
                    stat_type="t",
                    two_sided=True,
                )
            )
        else:
            corrected_p = rft.correct_p_values(
                statistic, stat_type="t", method="cluster"
            )

        contrast.corrected_p_values = corrected_p
        contrast.correction_method = "RFT"
        contrast.significance_mask = corrected_p < alpha
        contrast.threshold = alpha

    elif correction.upper() == "FDR":
        # Benjamini--Hochberg adjusted q-values.
        p_flat = contrast.p_values.flatten()
        n_tests = len(p_flat)
        if n_tests == 0:
            contrast.corrected_p_values = np.ones_like(contrast.p_values)
        else:
            order = np.argsort(p_flat, kind="stable")
            sorted_p = p_flat[order]
            q_values = np.empty(n_tests)
            running = 1.0
            for rank in range(n_tests, 0, -1):
                running = min(running, sorted_p[rank - 1] * n_tests / rank)
                q_values[rank - 1] = running
            q_values = np.clip(q_values, 0.0, 1.0)
            q_flat = np.empty(n_tests)
            q_flat[order] = q_values
            contrast.corrected_p_values = q_flat.reshape(contrast.p_values.shape)

        contrast.correction_method = "FDR"
        contrast.significance_mask = contrast.corrected_p_values < alpha

    elif correction.upper() == "BONFERRONI":
        n_tests = int(np.prod(contrast.p_values.shape))
        contrast.corrected_p_values = np.minimum(contrast.p_values * n_tests, 1.0)
        contrast.correction_method = "Bonferroni"
        contrast.significance_mask = contrast.corrected_p_values < alpha

    else:
        contrast.corrected_p_values = contrast.p_values
        contrast.correction_method = "uncorrected"
        contrast.significance_mask = contrast.p_values < alpha

    return contrast
