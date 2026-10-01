"""
Enhanced Catastrophe Models for Advanced Risk Assessment

This module provides sophisticated catastrophe modeling capabilities with:
- Integration with GEO-INFER-SPACE for spatial analysis
- Integration with GEO-INFER-TIME for temporal dynamics
- Integration with GEO-INFER-MATH for advanced statistical methods
- Climate change scenario modeling
- Advanced event generation with spatial correlation
- Real-time catastrophe monitoring
- Uncertainty quantification and ensemble modeling
- Multi-peril interaction and compound events
"""

import numpy as np
import pandas as pd
from itertools import permutations
from typing import Any
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import json
from scipy import stats, spatial

from ..utils.rng import SeedLike, resolve_rng

# GEO-INFER module imports with error handling
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface

    SPACE_AVAILABLE = True
except ImportError:
    SPACE_AVAILABLE = False
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

try:
    from geo_infer_time.core.temporal_analysis import TemporalAnalysisInterface

    TIME_AVAILABLE = True
except ImportError:
    TIME_AVAILABLE = False
    TemporalAnalysisInterface = None

try:
    from geo_infer_math.core.spatial_statistics import (  # type: ignore[import-untyped]
        MoranI,
    )

    MATH_AVAILABLE = True
except ImportError:
    MATH_AVAILABLE = False
    MoranI = None

logger = logging.getLogger(__name__)


@dataclass
class CatastropheConfig:
    """Enhanced configuration for catastrophe models."""

    # Model parameters
    simulation_years: int = 1000
    return_periods: list[int] = field(
        default_factory=lambda: [10, 25, 50, 100, 250, 500]
    )
    simulation_method: str = (
        "monte_carlo"  # monte_carlo, historical, hybrid, parametric
    )

    # Geographic parameters
    spatial_resolution: float = 0.1  # degrees
    max_distance: float = 100.0  # km
    spatial_correlation: bool = True
    spatial_correlation_range: float = 50.0  # km

    # Event parameters
    event_types: list[str] = field(
        default_factory=lambda: [
            "earthquake",
            "hurricane",
            "flood",
            "wildfire",
            "tornado",
        ]
    )
    include_secondary_perils: bool = True
    secondary_peril_probability: float = 0.1

    # Climate parameters
    include_climate_change: bool = False
    climate_scenario: str = "rcp4.5"
    climate_time_horizon: int = 2050

    # Financial parameters
    currency: str = "USD"
    inflation_rate: float = 0.02
    discount_rate: float = 0.05

    # Uncertainty parameters
    uncertainty_method: str = "parametric"  # parametric, bootstrap, bayesian
    confidence_level: float = 0.95
    num_uncertainty_samples: int = 1000

    # Performance parameters
    parallel_processing: bool = True
    batch_size: int = 1000
    cache_results: bool = True

    # Reproducibility: seed for the model's own generator. ``None`` means a
    # fresh generator seeded from OS entropy, so runs are not replayable.
    random_seed: SeedLike = None


class MultiHazardInteractionMatrix:
    """Directed cross-hazard interactions for compound exceedance analysis.

    Rows are source hazards and columns are downstream hazards.  Off-diagonal
    entries are interaction strengths in ``[-1, 1]``: zero preserves the
    downstream hazard's marginal exceedance probability, one makes downstream
    exceedance certain after its source exceeds, and minus one suppresses it.
    The matrix may be asymmetric, which represents causal chains such as
    earthquake -> fire-following -> flood rather than an ordinary symmetric
    loss-correlation matrix.

    Compound probabilities use a first-order Markov chain along the requested
    hazard sequence.  If ``p_j`` is the marginal probability of the next
    hazard and ``w_ij`` is its interaction with the preceding hazard, the
    conditional probability is

    ``p_j + w_ij * (1 - p_j)`` for non-negative interactions, and
    ``p_j * (1 + w_ij)`` for negative interactions.  Consequently an all-zero
    interaction matrix exactly recovers independent joint exceedance.

    Parameters
    ----------
    hazards : sequence of str
        Unique hazard names defining matrix row and column order.
    matrix : array-like of shape (n_hazards, n_hazards), optional
        Directed interaction strengths.  The default has zero off-diagonal
        interactions.  Diagonal entries are retained for reporting but are not
        used in compound paths.
    """

    def __init__(
        self,
        hazards: Sequence[str],
        matrix: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.hazards = self._validate_hazards(hazards)
        if matrix is None:
            self.matrix = np.eye(len(self.hazards), dtype=float)
        else:
            self.matrix = self._validate_matrix(matrix, len(self.hazards))

    @staticmethod
    def _validate_hazards(hazards: Sequence[str]) -> list[str]:
        """Return normalized unique hazard names."""
        if isinstance(hazards, (str, bytes)):
            raise TypeError("hazards must be a sequence of hazard names")
        normalized = [str(hazard).strip() for hazard in hazards]
        if not normalized or any(not hazard for hazard in normalized):
            raise ValueError("hazards must contain at least one non-empty name")
        if len(set(normalized)) != len(normalized):
            raise ValueError("hazard names must be unique")
        return normalized

    @staticmethod
    def _validate_matrix(
        matrix: Sequence[Sequence[float]], hazard_count: int
    ) -> np.ndarray:
        """Validate a finite square interaction matrix."""
        values = np.asarray(matrix, dtype=float)
        if values.shape != (hazard_count, hazard_count):
            raise ValueError(
                f"interaction matrix must have shape ({hazard_count}, {hazard_count})"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("interaction matrix must contain only finite values")
        if np.any((values < -1.0) | (values > 1.0)):
            raise ValueError("interaction strengths must be between -1 and 1")
        return np.array(values, copy=True)

    @property
    def hazard_types(self) -> list[str]:
        """Return the matrix hazard order."""
        return list(self.hazards)

    @property
    def interaction_matrix(self) -> np.ndarray:
        """Return a copy of the directed interaction matrix."""
        return np.array(self.matrix, copy=True)

    def add_hazard(self, hazard: str) -> None:
        """Add a hazard while preserving existing matrix entries."""
        name = str(hazard).strip()
        if not name:
            raise ValueError("hazard must be a non-empty name")
        if name in self.hazards:
            return
        expanded = np.zeros((len(self.hazards) + 1, len(self.hazards) + 1), dtype=float)
        expanded[:-1, :-1] = self.matrix
        expanded[-1, -1] = 1.0
        self.hazards.append(name)
        self.matrix = expanded

    def set_interaction(self, source: str, target: str, strength: float) -> None:
        """Set one directed source-to-target interaction strength."""
        if source not in self.hazards or target not in self.hazards:
            raise KeyError("source and target must both be configured hazards")
        value = float(strength)
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("interaction strength must be finite and between -1 and 1")
        self.matrix[self.hazards.index(source), self.hazards.index(target)] = value

    def get_interaction(self, source: str, target: str) -> float:
        """Return one directed interaction strength."""
        if source not in self.hazards or target not in self.hazards:
            raise KeyError("source and target must both be configured hazards")
        return float(
            self.matrix[self.hazards.index(source), self.hazards.index(target)]
        )

    @staticmethod
    def _validate_probability(probability: float, hazard: str) -> float:
        """Return one finite probability in the closed unit interval."""
        value = float(probability)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"exceedance probability for {hazard!r} must be between 0 and 1"
            )
        return value

    def conditional_exceedance_probability(
        self,
        source: str,
        target: str,
        target_probability: float,
    ) -> float:
        """Adjust a target marginal probability after its source exceeds."""
        probability = self._validate_probability(target_probability, target)
        strength = self.get_interaction(source, target)
        if strength >= 0.0:
            conditional = probability + strength * (1.0 - probability)
        else:
            conditional = probability * (1.0 + strength)
        return float(np.clip(conditional, 0.0, 1.0))

    def compound_exceedance_probability(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        hazard_sequence: Sequence[str] | None = None,
    ) -> float:
        """Calculate joint exceedance along a directed compound-hazard path.

        Parameters
        ----------
        exceedance_probabilities : mapping or sequence of float
            Marginal annual exceedance probabilities.  A sequence is aligned to
            ``hazards``; a mapping is keyed by hazard name.
        hazard_sequence : sequence of str, optional
            Ordered compound path.  The configured hazard order is used when
            omitted.

        Returns
        -------
        float
            Joint exceedance estimate in ``[0, 1]``.
        """
        if isinstance(exceedance_probabilities, Mapping):
            probabilities = {
                hazard: self._validate_probability(probability, hazard)
                for hazard, probability in exceedance_probabilities.items()
            }
        else:
            values = np.asarray(exceedance_probabilities, dtype=float)
            if values.ndim != 1 or values.size != len(self.hazards):
                raise ValueError(
                    "probability sequence must have one value per configured hazard"
                )
            probabilities = {
                hazard: self._validate_probability(values[index], hazard)
                for index, hazard in enumerate(self.hazards)
            }

        sequence = list(self.hazards if hazard_sequence is None else hazard_sequence)
        if not sequence:
            raise ValueError("hazard_sequence must contain at least one hazard")
        if len(set(sequence)) != len(sequence):
            raise ValueError("hazard_sequence must not repeat hazards")
        unknown = [hazard for hazard in sequence if hazard not in self.hazards]
        if unknown:
            raise KeyError(f"Unknown hazards in sequence: {unknown}")
        missing = [hazard for hazard in sequence if hazard not in probabilities]
        if missing:
            raise KeyError(f"Missing exceedance probabilities for: {missing}")

        compound = probabilities[sequence[0]]
        for source, target in zip(sequence, sequence[1:]):
            compound *= self.conditional_exceedance_probability(
                source, target, probabilities[target]
            )
        return float(np.clip(compound, 0.0, 1.0))

    @staticmethod
    def _validate_paths(
        hazard_sequences: Sequence[Sequence[str]], hazards: Sequence[str]
    ) -> list[tuple[str, ...]]:
        """Normalize a list of compound paths against the configured hazards."""
        if not hazard_sequences:
            raise ValueError("hazard_sequences must contain at least one path")
        paths: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        configured = set(hazards)
        for raw_path in hazard_sequences:
            sequence = tuple(str(hazard).strip() for hazard in raw_path)
            if not sequence or any(not hazard for hazard in sequence):
                raise ValueError("every compound path must contain hazard names")
            unknown = [hazard for hazard in sequence if hazard not in configured]
            if unknown:
                raise KeyError(f"Unknown hazards in sequence: {unknown}")
            if len(set(sequence)) != len(sequence):
                raise ValueError("a compound path must not repeat hazards")
            if sequence not in seen:
                seen.add(sequence)
                paths.append(sequence)
        return paths

    @staticmethod
    def _coerce_probabilities(
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        hazards: Sequence[str],
    ) -> dict[str, float]:
        """Normalize marginal probabilities keyed by hazard name."""
        if isinstance(exceedance_probabilities, Mapping):
            return {
                hazard: MultiHazardInteractionMatrix._validate_probability(
                    probability, hazard
                )
                for hazard, probability in exceedance_probabilities.items()
            }
        values = np.asarray(exceedance_probabilities, dtype=float)
        if values.ndim != 1 or values.size != len(hazards):
            raise ValueError(
                "probability sequence must have one value per configured hazard"
            )
        return {
            hazard: MultiHazardInteractionMatrix._validate_probability(
                values[index], hazard
            )
            for index, hazard in enumerate(hazards)
        }

    def _resolve_probability(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        hazard: str,
    ) -> float:
        """Return one hazard's marginal exceedance probability."""
        probabilities = self._coerce_probabilities(
            exceedance_probabilities, self.hazards
        )
        return probabilities[hazard]

    def joint_exceedance_probability(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        hazard_sequences: Sequence[Sequence[str]],
    ) -> float:
        """Return the probability that *any* of several compound paths exceeds.

        This is the multi-hazard *compound* complement of the single-path
        :meth:`compound_exceedance_probability`.  Each path is evaluated as its
        own directed Markov chain, and the paths are treated as independent
        competing compound scenarios whose exceedances may co-occur:

        ``P_union = 1 - Product_j (1 - p_path_j)``.

        Parameters
        ----------
        exceedance_probabilities : mapping or sequence of float
            Marginal annual exceedance probabilities (same contract as
            :meth:`compound_exceedance_probability`).
        hazard_sequences : sequence of sequence of str
            One or more ordered compound paths.  Empty paths and repeated
            hazards are rejected.

        Returns
        -------
        float
            Probability that at least one configured compound path is
            exceeded, in ``[0, 1]``.
        """
        paths = self._validate_paths(hazard_sequences, self.hazards)
        not_exceeded = 1.0
        for path in paths:
            p_path = self.compound_exceedance_probability(
                exceedance_probabilities, path
            )
            not_exceeded *= 1.0 - p_path
        return float(np.clip(1.0 - not_exceeded, 0.0, 1.0))

    def branch_exceedance_probabilities(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        max_path_length: int | None = None,
    ) -> dict[tuple[str, ...], float]:
        """Return every non-repeating directed compound path and its probability.

        Parameters
        ----------
        exceedance_probabilities : mapping or sequence of float
            Marginal annual exceedances (same contract as
            :meth:`compound_exceedance_probability`).
        max_path_length : int, optional
            Upper bound on path length (number of hazards).  When omitted all
            non-repeating paths over the configured hazards are enumerated.

        Returns
        -------
        dict of tuple -> float
            Mapping from each compound path (as an ordered hazard tuple) to its
            joint exceedance estimate.  Singletons are included with their
            marginal probability and record the independent baseline.
        """
        probabilities = self._coerce_probabilities(
            exceedance_probabilities, self.hazards
        )
        n_hazards = len(self.hazards)
        if max_path_length is None:
            max_path_length = n_hazards
        if max_path_length < 1:
            raise ValueError("max_path_length must be at least 1")
        branches: dict[tuple[str, ...], float] = {}
        for length in range(1, min(max_path_length, n_hazards) + 1):
            for permutation in permutations(self.hazards, length):
                branches[permutation] = self.compound_exceedance_probability(
                    probabilities, permutation
                )
        return branches

    def dominant_exceedance_path(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        max_path_length: int | None = None,
    ) -> tuple[tuple[str, ...], float]:
        """Return the compound path with the largest joint exceedance.

        Enumeration is over the same non-repeating ordered paths as
        :meth:`branch_exceedance_probabilities`.  Paths whose joint probability
        is zero are ignored so a zero-marginal hazard never wins by default.

        Returns
        -------
        tuple[tuple, float]
            The ``(path, probability)`` pair.  For a fully degenerate case (no
            positive path) the leading singleton with the largest marginal is
            returned.
        """
        branches = self.branch_exceedance_probabilities(
            exceedance_probabilities, max_path_length
        )
        if not branches:
            raise ValueError("no compound paths can be formed over empty hazards")
        ranking = max(branches.items(), key=lambda item: item[1])
        if ranking[1] > 0.0:
            return ranking
        marginals = {
            (hazard,): self._resolve_probability(exceedance_probabilities, hazard)
            for hazard in self.hazards
        }
        lead = max(marginals.items(), key=lambda item: item[1])
        return (lead[0], lead[1])

    def set_interactions(self, interactions: Mapping[tuple[str, str], float]) -> None:
        """Set several directed interactions at once from a mapping."""
        for (source, target), strength in interactions.items():
            self.set_interaction(source, target, strength)

    @classmethod
    def from_mapping(
        cls,
        hazards: Sequence[str],
        interactions: Mapping[tuple[str, str], float],
    ) -> "MultiHazardInteractionMatrix":
        """Build a directed interaction matrix from a sparse mapping.

        Parameters
        ----------
        hazards : sequence of str
            Unique hazard names defining matrix order.
        interactions : mapping of (source, target) -> strength
            Sparse directed interactions.  Unspecified off-diagonal pairs are
            zero (independent); diagonals are one.

        Returns
        -------
        MultiHazardInteractionMatrix
        """
        model = cls(hazards)
        model.set_interactions(interactions)
        return model

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable matrix representation."""
        return {
            "hazards": list(self.hazards),
            "interaction_matrix": self.matrix.tolist(),
        }


def calculate_compound_exceedance_probability(
    exceedance_probabilities: Mapping[str, float] | Sequence[float],
    interaction_matrix: Sequence[Sequence[float]],
    hazard_types: Sequence[str] | None = None,
    hazard_sequence: Sequence[str] | None = None,
) -> float:
    """Calculate compound exceedance from marginals and an interaction matrix."""
    if hazard_types is None:
        if not isinstance(exceedance_probabilities, Mapping):
            raise ValueError(
                "hazard_types is required when probabilities are provided as a sequence"
            )
        hazard_types = list(exceedance_probabilities)
    model = MultiHazardInteractionMatrix(hazard_types, interaction_matrix)
    return model.compound_exceedance_probability(
        exceedance_probabilities, hazard_sequence
    )


class EnhancedCatastropheModel:
    """
    Enhanced catastrophe model with advanced simulation and analysis capabilities.

    This class provides sophisticated catastrophe modeling with:
    - Advanced spatial and temporal analysis
    - Climate change scenario integration
    - Multi-peril correlation modeling
    - Uncertainty quantification
    - Real-time event simulation
    - Integration with external data sources
    """

    def __init__(
        self,
        config: CatastropheConfig | None = None,
        random_seed: SeedLike = None,
    ):
        """Initialize enhanced catastrophe model.

        Args:
            config: Enhanced model configuration.
            random_seed: Seed or generator for this model's stochastic
                simulation, overriding ``config.random_seed`` when given. See
                :func:`geo_infer_risk.utils.rng.resolve_rng` for accepted forms.
        """
        self.config = config or CatastropheConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize external interfaces
        self.spatial_interface = None
        self.temporal_interface = None
        self.math_interface = None

        if SPACE_AVAILABLE:
            try:
                self.spatial_interface = SpatialIndexingInterface()
                self.logger.info("Spatial interface initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize spatial interface: {e}")

        if TIME_AVAILABLE:
            try:
                self.temporal_interface = TemporalAnalysisInterface()
                self.logger.info("Temporal interface initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize temporal interface: {e}")

        if MATH_AVAILABLE:
            try:
                self.math_interface = MoranI()
                self.logger.info("Math interface initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize math interface: {e}")

        # Model state
        self.is_fitted = False
        self.historical_data: pd.DataFrame | None = None
        self.model_parameters: dict[str, Any] = {}
        self.climate_factors: dict[str, Any] = {}
        self.uncertainty_parameters: dict[str, Any] = {}

        # Event simulation state
        self.event_cache: dict[str, Any] = {}
        self.correlation_matrix: np.ndarray | None = None

        # All stochastic draws in this model come from this generator, never
        # from the process-wide numpy.random singleton. simulate_events() may
        # rebind it for a single run when passed its own seed.
        self._rng: np.random.Generator = resolve_rng(
            random_seed if random_seed is not None else self.config.random_seed
        )

        # Performance tracking
        self.simulation_metrics = {
            "total_simulations": 0,
            "average_simulation_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self.logger.info("Enhanced catastrophe model initialized")

    def fit(self, historical_data: pd.DataFrame) -> "EnhancedCatastropheModel":
        """Fit the model to historical data with enhanced analysis."""
        logger.info("Fitting enhanced catastrophe model...")

        self.historical_data = historical_data.copy()

        # Perform comprehensive data analysis
        self._analyze_historical_data()

        # Fit model parameters
        self._fit_model_parameters()

        # Initialize climate factors if needed
        if self.config.include_climate_change:
            self._initialize_climate_factors()

        # Calculate spatial correlation if available
        if self.config.spatial_correlation and self.spatial_interface:
            self._calculate_spatial_correlation()

        self.is_fitted = True
        logger.info("Enhanced catastrophe model fitted successfully")
        return self

    def _analyze_historical_data(self) -> None:
        """Perform comprehensive analysis of historical data."""
        if self.historical_data is None or self.historical_data.empty:
            return

        # Basic statistics
        self.model_parameters["event_count"] = len(self.historical_data)
        self.model_parameters["time_span_years"] = self._calculate_time_span()

        # Event frequency analysis
        self.model_parameters["annual_frequency"] = self._calculate_annual_frequency()

        # Magnitude/intensity analysis
        self._analyze_intensity_distribution()

        # Spatial analysis
        if self.spatial_interface:
            self._analyze_spatial_patterns()

        # Temporal analysis
        if self.temporal_interface:
            self._analyze_temporal_patterns()

    def _calculate_time_span(self) -> float:
        """Calculate time span of historical data."""
        assert self.historical_data is not None
        if "timestamp" not in self.historical_data.columns:
            return 50.0  # Default 50 years

        timestamps = pd.to_datetime(self.historical_data["timestamp"])
        time_span = (timestamps.max() - timestamps.min()).days / 365.25
        return float(time_span)

    def _calculate_annual_frequency(self) -> float:
        """Calculate annual event frequency."""
        time_span = self.model_parameters.get("time_span_years", 50.0)
        event_count = self.model_parameters.get("event_count", 100)
        return float(event_count / time_span)

    def _analyze_intensity_distribution(self) -> None:
        """Analyze intensity distribution of historical events."""
        assert self.historical_data is not None
        intensity_column = self._get_intensity_column()

        if intensity_column and intensity_column in self.historical_data.columns:
            intensities = self.historical_data[intensity_column].values

            # Fit distribution parameters
            try:
                # Candidate distributions, fit in name order; "weibull" is
                # fit with the location fixed at 0.
                candidate_distributions = {
                    "exponential": stats.expon,
                    "weibull": stats.weibull_min,
                    "lognormal": stats.lognorm,
                    "gumbel_r": stats.gumbel_r,
                }

                best_distribution = "exponential"
                best_aic = float("inf")
                best_params: tuple[float, ...] | None = None

                for dist_name, dist in candidate_distributions.items():
                    try:
                        if dist_name == "weibull":
                            params = dist.fit(intensities, floc=0)
                        else:
                            params = dist.fit(intensities)
                    except Exception as fit_error:
                        logger.debug(f"Fit failed for {dist_name}: {fit_error}")
                        continue

                    # Calculate AIC
                    log_likelihood = np.sum(dist.logpdf(intensities, *params))
                    n_params = len(params)
                    aic = 2 * n_params - 2 * log_likelihood

                    if aic < best_aic:
                        best_aic = aic
                        best_distribution = dist_name
                        best_params = params

                if best_params is not None:
                    self.model_parameters["intensity_distribution"] = best_distribution
                    self.model_parameters["intensity_distribution_params"] = best_params
                else:
                    logger.warning(
                        "All candidate intensity distribution fits failed; "
                        "storing no distribution parameters"
                    )
                    self.model_parameters["intensity_distribution"] = "exponential"

            except Exception as e:
                logger.warning(f"Failed to fit intensity distribution: {e}")
                self.model_parameters["intensity_distribution"] = "exponential"

    def _get_intensity_column(self) -> str | None:
        """Get the appropriate intensity column name."""
        assert self.historical_data is not None
        intensity_columns = [
            "magnitude",
            "intensity",
            "wind_speed",
            "water_depth",
            "fire_intensity",
        ]
        for col in intensity_columns:
            if col in self.historical_data.columns:
                return col
        return None

    def _analyze_spatial_patterns(self) -> None:
        """Analyze spatial patterns in historical data."""
        assert self.historical_data is not None
        if not self.spatial_interface or "latitude" not in self.historical_data.columns:
            return

        try:
            coords = self.historical_data[["longitude", "latitude"]].values

            # Calculate spatial statistics
            if len(coords) > 1:
                # Spatial autocorrelation (simplified)
                distances = spatial.distance.pdist(coords)
                self.model_parameters["spatial_correlation_length"] = (
                    np.mean(distances) * 0.3
                )

                # Hotspot analysis (simplified)
                self.model_parameters["spatial_hotspots"] = self._identify_hotspots(
                    coords
                )

        except Exception as e:
            logger.warning(f"Spatial analysis failed: {e}")

    def _analyze_temporal_patterns(self) -> None:
        """Analyze temporal patterns in historical data."""
        assert self.historical_data is not None
        if (
            not self.temporal_interface
            or "timestamp" not in self.historical_data.columns
        ):
            return

        try:
            # Extract temporal patterns
            timestamps = pd.to_datetime(self.historical_data["timestamp"])

            # Monthly patterns
            monthly_counts = timestamps.dt.month.value_counts()
            self.model_parameters["monthly_patterns"] = monthly_counts.to_dict()

            # Seasonal patterns
            seasonal_counts = timestamps.dt.quarter.value_counts()
            self.model_parameters["seasonal_patterns"] = seasonal_counts.to_dict()

            # Trend analysis (simplified)
            years = timestamps.dt.year
            yearly_counts = years.value_counts().sort_index()
            if len(yearly_counts) > 5:
                trend = self._calculate_trend(yearly_counts)
                self.model_parameters["temporal_trend"] = trend

        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")

    def _identify_hotspots(self, coords: np.ndarray) -> list[dict[str, float]]:
        """Identify spatial hotspots (simplified)."""
        # Simple hotspot identification using clustering
        try:
            from sklearn.cluster import KMeans

            if len(coords) > 10:
                # Use K-means for hotspot identification
                n_clusters = min(5, len(coords) // 10)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(coords)

                hotspots = []
                for i in range(n_clusters):
                    cluster_points = coords[clusters == i]
                    if len(cluster_points) > 0:
                        center = np.mean(cluster_points, axis=0)
                        hotspots.append(
                            {
                                "latitude": center[1],
                                "longitude": center[0],
                                "event_count": len(cluster_points),
                            }
                        )

                return hotspots
        except ImportError:
            # Fallback to simple centroid calculation
            return [
                {
                    "latitude": np.mean(coords[:, 1]),
                    "longitude": np.mean(coords[:, 0]),
                    "event_count": len(coords),
                }
            ]

        return []

    def _calculate_trend(self, yearly_counts: pd.Series) -> float:
        """Calculate temporal trend."""
        years = yearly_counts.index.values
        counts = yearly_counts.values

        if len(years) < 2:
            return 0.0

        # Simple linear trend
        slope, _, r_value, _, _ = stats.linregress(years, counts)
        return float(slope)

    def _fit_model_parameters(self) -> None:
        """Fit model parameters from historical data.

        Base implementation sets default parameters. Subclasses should override
        to provide hazard-specific parameter estimation from historical events.
        """
        self.model_parameters.setdefault("mean_intensity", 1.0)
        self.model_parameters.setdefault("std_intensity", 0.5)
        logger.info(
            "Using default model parameters; override in subclass for specific hazard"
        )

    def _initialize_climate_factors(self) -> None:
        """Initialize climate change adjustment factors."""
        if not self.config.include_climate_change:
            return

        # Climate scenario factors
        scenario_factors = {
            "rcp2.6": {"intensity": 1.05, "frequency": 1.1, "time_horizon": 2050},
            "rcp4.5": {"intensity": 1.15, "frequency": 1.2, "time_horizon": 2050},
            "rcp8.5": {"intensity": 1.3, "frequency": 1.4, "time_horizon": 2050},
            "ssp1-1.9": {"intensity": 1.02, "frequency": 1.05, "time_horizon": 2100},
            "ssp1-2.6": {"intensity": 1.08, "frequency": 1.15, "time_horizon": 2100},
            "ssp2-4.5": {"intensity": 1.18, "frequency": 1.25, "time_horizon": 2100},
            "ssp3-7.0": {"intensity": 1.25, "frequency": 1.35, "time_horizon": 2100},
            "ssp5-8.5": {"intensity": 1.4, "frequency": 1.5, "time_horizon": 2100},
        }

        self.climate_factors = scenario_factors.get(
            self.config.climate_scenario, {"intensity": 1.0, "frequency": 1.0}
        )
        logger.info(
            f"Climate factors initialized for scenario {self.config.climate_scenario}"
        )

    def _calculate_spatial_correlation(self) -> None:
        """Calculate spatial correlation matrix."""
        if not self.spatial_interface or self.historical_data is None:
            return

        try:
            coords = self.historical_data[["longitude", "latitude"]].values

            # Calculate pairwise distances
            distances = spatial.distance.pdist(coords)

            # Simple correlation model (exponential decay)
            correlation_range = self.config.spatial_correlation_range
            correlations = np.exp(-distances / correlation_range)

            # Create correlation matrix
            n = len(coords)
            self.correlation_matrix = np.eye(n)
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    self.correlation_matrix[i, j] = correlations[k]
                    self.correlation_matrix[j, i] = correlations[k]
                    k += 1

            logger.info("Spatial correlation matrix calculated")

        except Exception as e:
            logger.warning(f"Failed to calculate spatial correlation: {e}")

    def simulate_events(
        self,
        n_simulations: int,
        region: dict | None = None,
        time_period: tuple[datetime, datetime] | None = None,
        random_seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        """Simulate catastrophe events with advanced features.

        Args:
            n_simulations: Number of events to simulate.
            region: Spatial region constraints.
            time_period: Temporal constraints.
            random_seed: Seed or generator for this run, rebinding the model's
                generator. When ``None`` (default) the generator established at
                construction continues, so successive calls draw fresh events
                rather than repeating. See
                :func:`geo_infer_risk.utils.rng.resolve_rng`.

        Returns:
            List of simulated catastrophe events.
        """
        if random_seed is not None:
            self._rng = resolve_rng(random_seed)

        logger.info(f"Simulating {n_simulations} catastrophe events")

        events = []

        # Apply climate change adjustment if enabled
        climate_multiplier = (
            self.climate_factors.get("frequency", 1.0)
            if self.config.include_climate_change
            else 1.0
        )

        # Generate events in batches for efficiency
        batch_size = min(self.config.batch_size, n_simulations)
        remaining_events = n_simulations

        while remaining_events > 0:
            current_batch = min(batch_size, remaining_events)
            batch_events = self._generate_event_batch(
                current_batch, region, time_period, climate_multiplier
            )
            events.extend(batch_events)
            remaining_events -= current_batch

        # Apply spatial correlation if available and configured
        if self.config.spatial_correlation and self.correlation_matrix is not None:
            events = self._apply_spatial_correlation(events)

        # Apply temporal patterns if available
        if self.temporal_interface and time_period:
            events = self._apply_temporal_patterns(events, time_period)

        # Update metrics
        self.simulation_metrics["total_simulations"] += n_simulations

        logger.info(f"Generated {len(events)} catastrophe events")
        return events

    def _generate_event_batch(
        self,
        batch_size: int,
        region: dict | None = None,
        time_period: tuple[datetime, datetime] | None = None,
        climate_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Generate a batch of catastrophe events."""
        events = []

        for i in range(batch_size):
            event = self._generate_single_event(region, time_period, climate_multiplier)
            events.append(event)

        return events

    def _generate_single_event(
        self,
        region: dict | None = None,
        time_period: tuple[datetime, datetime] | None = None,
        climate_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """Generate a single catastrophe event.

        Base implementation generates a generic event with random location and intensity.
        Subclasses should override for hazard-specific event generation.
        """
        import random

        mean_intensity = self.model_parameters.get("mean_intensity", 1.0)
        std_intensity = self.model_parameters.get("std_intensity", 0.5)
        intensity = max(
            0.0, random.gauss(mean_intensity * climate_multiplier, std_intensity)
        )

        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        if region:
            lat = random.uniform(region.get("min_lat", -90), region.get("max_lat", 90))
            lon = random.uniform(
                region.get("min_lon", -180), region.get("max_lon", 180)
            )

        event_time = datetime.now()
        if time_period:
            start, end = time_period
            delta = (end - start).total_seconds()
            event_time = start + timedelta(seconds=random.uniform(0, delta))

        return {
            "intensity": intensity,
            "location": {"latitude": lat, "longitude": lon},
            "timestamp": event_time,
            "metadata": {"climate_multiplier": climate_multiplier},
        }

    def _apply_spatial_correlation(
        self, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply spatial correlation to generated events using distance-decay model."""
        if not self.spatial_interface or self.correlation_matrix is None:
            return events

        try:
            n_events = len(events)
            if n_events < 2:
                return events

            # Build pairwise distance matrix for the generated events
            event_coords = np.array(
                [
                    [e["location"]["longitude"], e["location"]["latitude"]]
                    for e in events
                ]
            )
            dist_matrix = spatial.distance.squareform(
                spatial.distance.pdist(event_coords)
            )

            # Compute exponential-decay correlation from config range
            corr_range = self.config.spatial_correlation_range
            C = np.exp(-dist_matrix / corr_range)

            # Ensure positive-definiteness then Cholesky-decompose
            C += 1e-6 * np.eye(n_events)
            L = np.linalg.cholesky(C)

            # Draw correlated standard-normal perturbations
            z = self._rng.standard_normal(n_events)
            correlated_z = L @ z

            # Apply perturbation to intensities (multiplicative, bounded to ±30 %)
            for i, event in enumerate(events):
                factor = 1.0 + 0.3 * np.tanh(correlated_z[i])
                event["intensity"] = event.get("intensity", 1.0) * factor
                event.setdefault("metadata", {})["spatial_correlation_factor"] = float(
                    factor
                )

            logger.info(
                "Spatial correlation applied to %d events (range=%.1f)",
                n_events,
                corr_range,
            )
            return events

        except Exception as e:
            logger.warning(f"Failed to apply spatial correlation: {e}")
            return events

    def _apply_temporal_patterns(
        self, events: list[dict[str, Any]], time_period: tuple[datetime, datetime]
    ) -> list[dict[str, Any]]:
        """Apply temporal patterns to events."""
        if not self.temporal_interface:
            return events

        try:
            # Apply seasonal and temporal patterns
            monthly_patterns = self.model_parameters.get("monthly_patterns", {})

            for event in events:
                timestamp = event["timestamp"]
                month = timestamp.month

                # Apply monthly multiplier
                multiplier = monthly_patterns.get(month, 1.0)
                event["intensity"] *= multiplier
                event["metadata"]["temporal_adjustment"] = multiplier

            logger.info("Temporal patterns applied")
            return events

        except Exception as e:
            logger.warning(f"Failed to apply temporal patterns: {e}")
            return events

    def calculate_loss(self, event: dict[str, Any], exposure: dict[str, Any]) -> float:
        """Calculate loss for a given event and exposure.

        Base implementation uses a simple intensity-based loss fraction.
        Subclasses should override for hazard-specific loss calculations.

        Args:
            event: Catastrophe event with intensity and location.
            exposure: Exposure data with total value.

        Returns:
            Estimated loss amount.
        """
        intensity = event.get("intensity", 0.0)
        total_value = exposure.get("total_value", 0.0)
        # Simple linear damage function capped at 100%
        damage_fraction = min(1.0, max(0.0, intensity / 10.0))
        return float(damage_fraction * total_value)

    def get_model_status(self) -> dict[str, Any]:
        """Get comprehensive model status information."""
        return {
            "is_fitted": self.is_fitted,
            "historical_data_available": self.historical_data is not None,
            "event_count": (
                len(self.historical_data) if self.historical_data is not None else 0
            ),
            "climate_change_enabled": self.config.include_climate_change,
            "spatial_correlation_enabled": self.config.spatial_correlation,
            "correlation_matrix_available": self.correlation_matrix is not None,
            "integration_status": {
                "spatial_interface": self.spatial_interface is not None,
                "temporal_interface": self.temporal_interface is not None,
                "math_interface": self.math_interface is not None,
            },
            "model_parameters": self.model_parameters,
            "simulation_metrics": self.simulation_metrics,
        }

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        model_state = {
            "config": self.config.__dict__,
            "model_parameters": self.model_parameters,
            "climate_factors": self.climate_factors,
            "is_fitted": self.is_fitted,
            "simulation_metrics": self.simulation_metrics,
            "metadata": {"saved_at": datetime.now().isoformat(), "version": "2.0.0"},
        }

        with open(filepath, "w") as f:
            json.dump(model_state, f, indent=2, default=str)

        logger.info(f"Catastrophe model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        with open(filepath) as f:
            model_state = json.load(f)

        self.config = CatastropheConfig(**model_state["config"])
        self.model_parameters = model_state["model_parameters"]
        self.climate_factors = model_state["climate_factors"]
        self.is_fitted = model_state["is_fitted"]
        self.simulation_metrics = model_state["simulation_metrics"]

        logger.info(f"Catastrophe model loaded from {filepath}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check on catastrophe model."""
        return {
            "status": "operational" if self.is_fitted else "not_fitted",
            "total_simulations": self.simulation_metrics["total_simulations"],
            "integration_status": {
                "spatial": SPACE_AVAILABLE,
                "temporal": TIME_AVAILABLE,
                "math": MATH_AVAILABLE,
            },
            "timestamp": datetime.now().isoformat(),
        }


# Enhanced Specific Catastrophe Models


class EnhancedEarthquakeModel(EnhancedCatastropheModel):
    """Enhanced earthquake catastrophe model with advanced seismological modeling."""

    def __init__(
        self,
        config: CatastropheConfig | None = None,
        random_seed: SeedLike = None,
    ):
        super().__init__(config, random_seed=random_seed)
        self.fault_lines: list[Any] = []
        self.seismicity_rates: dict[str, Any] = {}

    def _fit_model_parameters(self) -> None:
        """Fit earthquake-specific model parameters."""
        if self.historical_data is None:
            return

        if "magnitude" in self.historical_data.columns:
            magnitudes = self.historical_data["magnitude"].values

            # Fit Gutenberg-Richter parameters
            self.model_parameters.update(
                {
                    "mean_magnitude": np.mean(magnitudes),
                    "std_magnitude": np.std(magnitudes),
                    "min_magnitude": np.min(magnitudes),
                    "max_magnitude": np.max(magnitudes),
                    "b_value": self._estimate_b_value(magnitudes),
                    "annual_rate": self.model_parameters["annual_frequency"],
                }
            )

        # Fit depth distribution
        if "depth" in self.historical_data.columns:
            depths = self.historical_data["depth"].values
            self.model_parameters.update(
                {"mean_depth": np.mean(depths), "depth_distribution": "exponential"}
            )

    def _estimate_b_value(self, magnitudes: np.ndarray) -> float:
        """Estimate Gutenberg-Richter b-value."""
        if len(magnitudes) < 10:
            return 1.0

        # Use maximum likelihood estimation
        min_mag = np.percentile(magnitudes, 90)
        complete_mags = magnitudes[magnitudes >= min_mag]

        if len(complete_mags) < 5:
            return 1.0

        mean_mag = np.mean(complete_mags)
        b_value = 1.0 / (mean_mag - min_mag + 0.05)

        return float(max(0.5, min(2.0, b_value)))

    def _generate_single_event(
        self,
        region: dict | None = None,
        time_period: tuple[datetime, datetime] | None = None,
        climate_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """Generate a single earthquake event."""
        # Generate magnitude using Gutenberg-Richter
        magnitude = self._generate_earthquake_magnitude()

        # Generate location
        location = self._generate_earthquake_location(region)

        # Generate timestamp
        timestamp = self._generate_event_timestamp(time_period)

        # Create event
        event = {
            "event_id": f"EQ_{int(self._rng.integers(1000000))}",
            "hazard_type": "earthquake",
            "timestamp": timestamp,
            "location": location,
            "magnitude": magnitude,
            "depth": location["depth"],
            "intensity_measure": "magnitude",
            "units": "Mw",
            "metadata": {
                "climate_adjusted": self.config.include_climate_change,
                "climate_scenario": (
                    self.config.climate_scenario
                    if self.config.include_climate_change
                    else None
                ),
                "generation_method": "gutenberg_richter",
                "model_version": "2.0.0",
            },
        }

        # Add tectonic region
        event["tectonic_region"] = self._rng.choice(
            ["active_crustal", "subduction", "stable_crustal"]
        )

        return event

    def _generate_earthquake_magnitude(self) -> float:
        """Generate earthquake magnitude using fitted parameters."""
        if not self.is_fitted:
            # Default Gutenberg-Richter with b=1.0
            u = self._rng.uniform(0, 1)
            return float(4.0 + np.log10(1 / u))  # Simplified

        params = self.model_parameters
        b_value = params.get("b_value", 1.0)
        min_mag = params.get("min_magnitude", 4.0)

        # Generate using inverse transform sampling
        u = self._rng.uniform(0, 1)
        magnitude = min_mag + np.log10(1 / u) / b_value

        return float(min(8.5, magnitude))

    def _generate_earthquake_location(
        self, region: dict | None = None
    ) -> dict[str, Any]:
        """Generate earthquake location."""
        if region and "bounds" in region:
            bounds = region["bounds"]
            lat = self._rng.uniform(
                bounds.get("min_lat", -60), bounds.get("max_lat", 60)
            )
            lon = self._rng.uniform(
                bounds.get("min_lon", -180), bounds.get("max_lon", 180)
            )
        else:
            lat = self._rng.uniform(-60, 60)
            lon = self._rng.uniform(-180, 180)

        # Generate depth
        if self.is_fitted:
            depth = self._rng.exponential(self.model_parameters.get("mean_depth", 15.0))
        else:
            depth = self._rng.exponential(15.0)

        return {"latitude": lat, "longitude": lon, "depth": depth}

    def _generate_event_timestamp(
        self, time_period: tuple[datetime, datetime] | None = None
    ) -> datetime:
        """Draw a uniformly distributed event timestamp.

        Args:
            time_period: Inclusive-start, exclusive-end window to draw from.
                When omitted, a day within the next year is drawn.

        Returns:
            The event timestamp. A zero-width window returns its start.
        """
        if time_period:
            start_time, end_time = time_period
            span_seconds = int((end_time - start_time).total_seconds())
            if span_seconds <= 0:
                return start_time
            return start_time + timedelta(
                seconds=int(self._rng.integers(0, span_seconds))
            )
        return datetime.now() + timedelta(days=int(self._rng.integers(0, 365)))

    def calculate_loss(self, event: dict[str, Any], exposure: dict[str, Any]) -> float:
        """Calculate earthquake loss."""
        magnitude = event["magnitude"]
        distance = self._calculate_distance(event, exposure)
        depth = event.get("depth", 15.0)

        # Simplified GMPE stand-in via _magnitude_to_pga — see SKILL.md
        # "Honest Capability Register".
        pga = self._magnitude_to_pga(magnitude, distance, depth)

        # Fixed soil multipliers (part of the GMPE stand-in entry in the
        # SKILL.md "Honest Capability Register").
        site_factor = 1.0
        if "soil_type" in exposure:
            soil_type = exposure["soil_type"]
            if soil_type == "soft":
                site_factor = 1.3
            elif soil_type == "rock":
                site_factor = 0.8

        # Calculate damage ratio using vulnerability functions
        damage_ratio = self._calculate_earthquake_damage_ratio(
            pga * site_factor, exposure
        )

        # Calculate loss
        loss = damage_ratio * exposure.get("value", 100000)

        return float(min(loss, exposure.get("value", 100000)))  # Cap at property value

    def _magnitude_to_pga(
        self, magnitude: float, distance: float, depth: float
    ) -> float:
        """Convert magnitude to PGA using simplified GMPE."""
        # Simplified log-distance attenuation, not a calibrated GMPE — see
        # SKILL.md "Honest Capability Register".
        if distance < 1:
            return 0.5 * 10 ** (0.3 * magnitude - 2.0)  # Near-field

        # Far-field attenuation
        r = np.sqrt(distance**2 + depth**2)
        pga = 10 ** (0.3 * magnitude - 2.0 - np.log10(r) - 0.002 * r)

        return float(max(0.001, pga))  # Minimum PGA threshold

    def _calculate_earthquake_damage_ratio(
        self, pga: float, exposure: dict[str, Any]
    ) -> float:
        """Calculate earthquake damage ratio."""
        # Simplified damage calculation
        if pga < 0.05:
            return 0.0
        elif pga < 0.15:
            return 0.1
        elif pga < 0.3:
            return 0.3
        elif pga < 0.5:
            return 0.6
        else:
            return 0.9

    def _calculate_distance(
        self, event: dict[str, Any], exposure: dict[str, Any]
    ) -> float:
        """Calculate distance between event and exposure."""
        event_lat = event["location"]["latitude"]
        event_lon = event["location"]["longitude"]
        exposure_lat = exposure.get("latitude", 0)
        exposure_lon = exposure.get("longitude", 0)

        # Haversine formula
        R = 6371  # Earth's radius in km
        lat1_rad, lon1_rad = np.radians([event_lat, event_lon])
        lat2_rad, lon2_rad = np.radians([exposure_lat, exposure_lon])

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return float(R * c)


class EnhancedHurricaneModel(EnhancedCatastropheModel):
    """Enhanced hurricane model with storm track modeling."""

    def __init__(
        self,
        config: CatastropheConfig | None = None,
        random_seed: SeedLike = None,
    ):
        super().__init__(config, random_seed=random_seed)
        self.track_data: list[Any] = []
        self.intensity_data: dict[str, Any] = {}

    def _fit_model_parameters(self) -> None:
        """Fit hurricane-specific model parameters."""
        if self.historical_data is None:
            return

        if "wind_speed" in self.historical_data.columns:
            wind_speeds = self.historical_data["wind_speed"].values

            # Fit Weibull distribution
            shape, loc, scale = stats.weibull_min.fit(wind_speeds, floc=0)

            self.model_parameters.update(
                {
                    "mean_wind_speed": np.mean(wind_speeds),
                    "std_wind_speed": np.std(wind_speeds),
                    "weibull_shape": shape,
                    "weibull_scale": scale,
                    "annual_frequency": self.model_parameters["annual_frequency"],
                }
            )

    def _generate_single_event(
        self,
        region: dict | None = None,
        time_period: tuple[datetime, datetime] | None = None,
        climate_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """Generate a single hurricane event."""
        # Generate wind speed
        wind_speed = self._generate_hurricane_intensity()

        # Generate track
        track = self._generate_hurricane_track(region)

        # Generate timestamp
        timestamp = self._generate_event_timestamp(  # type: ignore[attr-defined]
            time_period
        )

        # Create event
        event = {
            "event_id": f"HUR_{int(self._rng.integers(1000000))}",
            "hazard_type": "hurricane",
            "timestamp": timestamp,
            "location": track[0] if track else {"latitude": 25.0, "longitude": -80.0},
            "wind_speed": wind_speed,
            "category": self._get_hurricane_category(wind_speed),
            "track": track,
            "intensity_measure": "wind_speed",
            "units": "m/s",
            "metadata": {
                "climate_adjusted": self.config.include_climate_change,
                "climate_scenario": (
                    self.config.climate_scenario
                    if self.config.include_climate_change
                    else None
                ),
                "generation_method": "weibull_track",
                "model_version": "2.0.0",
            },
        }

        return event

    def _generate_hurricane_intensity(self) -> float:
        """Generate hurricane wind speed."""
        if self.is_fitted:
            params = self.model_parameters
            shape = params.get("weibull_shape", 2.5)
            scale = params.get("weibull_scale", 30.0)
            wind_speed = self._rng.weibull(shape) * scale
        else:
            wind_speed = self._rng.weibull(2.5) * 40 + 30

        return float(max(25, wind_speed))  # Minimum tropical storm strength

    def _generate_hurricane_track(
        self, region: dict | None = None
    ) -> list[dict[str, Any]]:
        """Generate hurricane track."""
        track_length = int(self._rng.integers(5, 20))
        track = []

        # Start in tropical Atlantic
        if region and "bounds" in region:
            bounds = region["bounds"]
            start_lat = self._rng.uniform(
                bounds.get("min_lat", 10), bounds.get("max_lat", 30)
            )
            start_lon = self._rng.uniform(
                bounds.get("min_lon", -100), bounds.get("max_lon", -60)
            )
        else:
            start_lat = self._rng.uniform(10, 30)
            start_lon = self._rng.uniform(-100, -60)

        current_lat, current_lon = start_lat, start_lon

        for i in range(track_length):
            # Storm movement (westward then northward)
            if i < track_length // 2:
                delta_lon = self._rng.normal(-0.2, 0.1)
                delta_lat = self._rng.normal(0.1, 0.1)
            else:
                delta_lon = self._rng.normal(0.1, 0.1)
                delta_lat = self._rng.normal(0.3, 0.1)

            current_lat += delta_lat
            current_lon += delta_lon

            # Intensity decay
            decay_factor = np.exp(-i * 0.1)
            wind_speed = self._generate_hurricane_intensity() * decay_factor

            track_point = {
                "time_offset": i * 6,  # Hours
                "latitude": current_lat,
                "longitude": current_lon,
                "wind_speed": wind_speed,
                "pressure": 1013.0 - (wind_speed - 30) * 2.0,
            }

            track.append(track_point)

        return track

    def _get_hurricane_category(self, wind_speed: float) -> int:
        """Get Saffir-Simpson hurricane category."""
        if wind_speed < 33:
            return 0  # Tropical storm
        elif wind_speed < 43:
            return 1
        elif wind_speed < 50:
            return 2
        elif wind_speed < 58:
            return 3
        elif wind_speed < 70:
            return 4
        else:
            return 5

    def calculate_loss(self, event: dict[str, Any], exposure: dict[str, Any]) -> float:
        """Calculate hurricane loss."""
        wind_speed = event["wind_speed"]
        distance = self._calculate_minimum_distance(event, exposure)

        # Wind damage
        wind_factor = min(wind_speed / 100.0, 1.0)

        # Distance factor
        distance_factor = max(0.1, 1.0 - distance / 100.0)

        # Storm surge if applicable
        storm_surge = event.get("storm_surge", 0)
        surge_factor = min(storm_surge / 5.0, 1.0)

        # Vulnerability
        vulnerability = exposure.get("vulnerability", 0.6)

        total_loss = (
            exposure.get("value", 200000)
            * (wind_factor + surge_factor)
            * distance_factor
            * vulnerability
        )

        return float(min(total_loss, exposure.get("value", 200000)))

    def _calculate_minimum_distance(
        self, event: dict[str, Any], exposure: dict[str, Any]
    ) -> float:
        """Calculate minimum distance from hurricane track to exposure."""
        track = event.get("track", [])
        if not track:
            return self._calculate_distance(event, exposure)

        exposure_lat = exposure.get("latitude", 0)
        exposure_lon = exposure.get("longitude", 0)

        min_distance = float("inf")

        for track_point in track:
            track_lat = track_point["latitude"]
            track_lon = track_point["longitude"]

            distance = (
                np.sqrt(
                    (exposure_lat - track_lat) ** 2 + (exposure_lon - track_lon) ** 2
                )
                * 111
            )
            min_distance = min(min_distance, distance)

        return min_distance

    def _calculate_distance(
        self, event: dict[str, Any], exposure: dict[str, Any]
    ) -> float:
        """Calculate distance between event and exposure."""
        event_lat = event["location"]["latitude"]
        event_lon = event["location"]["longitude"]
        exposure_lat = exposure.get("latitude", 0)
        exposure_lon = exposure.get("longitude", 0)

        # Haversine formula
        R = 6371
        lat1_rad, lon1_rad = np.radians([event_lat, event_lon])
        lat2_rad, lon2_rad = np.radians([exposure_lat, exposure_lon])

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return float(R * c)


class EnhancedFloodModel(EnhancedCatastropheModel):
    """Enhanced flood model with hydrological modeling."""

    def __init__(
        self,
        config: CatastropheConfig | None = None,
        random_seed: SeedLike = None,
    ):
        super().__init__(config, random_seed=random_seed)
        self.river_data: dict[str, Any] = {}
        self.rainfall_data: dict[str, Any] = {}

    def _fit_model_parameters(self) -> None:
        """Fit flood-specific model parameters."""
        if self.historical_data is None:
            return

        if "water_depth" in self.historical_data.columns:
            depths = self.historical_data["water_depth"].values

            # Fit Gumbel distribution for extreme values
            try:
                loc, scale = stats.gumbel_r.fit(depths)
                self.model_parameters.update(
                    {
                        "mean_depth": np.mean(depths),
                        "gumbel_location": loc,
                        "gumbel_scale": scale,
                        "distribution": "gumbel",
                    }
                )
            except Exception:
                self.model_parameters.update(
                    {
                        "mean_depth": np.mean(depths),
                        "std_depth": np.std(depths),
                        "distribution": "normal",
                    }
                )

    def _generate_single_event(
        self,
        region: dict | None = None,
        time_period: tuple[datetime, datetime] | None = None,
        climate_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """Generate a single flood event."""
        # Generate water depth
        water_depth = self._generate_flood_intensity()

        # Generate location
        location = self._generate_flood_location(region)

        # Generate timestamp
        timestamp = self._generate_event_timestamp(  # type: ignore[attr-defined]
            time_period
        )

        # Create event
        event = {
            "event_id": f"FLD_{int(self._rng.integers(1000000))}",
            "hazard_type": "flood",
            "timestamp": timestamp,
            "location": location,
            "water_depth": water_depth,
            "intensity_measure": "water_depth",
            "units": "m",
            "metadata": {
                "climate_adjusted": self.config.include_climate_change,
                "climate_scenario": (
                    self.config.climate_scenario
                    if self.config.include_climate_change
                    else None
                ),
                "generation_method": "gumbel",
                "model_version": "2.0.0",
            },
        }

        # Add flood-specific properties
        event["flood_type"] = self._rng.choice(["riverine", "pluvial", "coastal"])
        event["duration"] = self._rng.exponential(72.0)  # Hours
        event["affected_area"] = self._rng.exponential(50.0)  # km²

        return event

    def _generate_flood_intensity(self) -> float:
        """Generate flood water depth."""
        if self.is_fitted:
            params = self.model_parameters
            distribution = params.get("distribution", "normal")

            if distribution == "gumbel":
                loc = params.get("gumbel_location", 2.0)
                scale = params.get("gumbel_scale", 1.0)
                depth = self._rng.gumbel(loc, scale)
            else:
                mean_depth = params.get("mean_depth", 2.0)
                std_depth = params.get("std_depth", 1.0)
                depth = self._rng.normal(mean_depth, std_depth)
        else:
            depth = self._rng.exponential(2.0)

        return float(max(0.1, depth))  # Minimum flood depth

    def _generate_flood_location(self, region: dict | None = None) -> dict[str, Any]:
        """Generate flood location."""
        if region and "bounds" in region:
            bounds = region["bounds"]
            lat = self._rng.uniform(
                bounds.get("min_lat", -90), bounds.get("max_lat", 90)
            )
            lon = self._rng.uniform(
                bounds.get("min_lon", -180), bounds.get("max_lon", 180)
            )
        else:
            # Global flood distribution
            lat = self._rng.uniform(-60, 60)
            lon = self._rng.uniform(-180, 180)

        return {
            "latitude": lat,
            "longitude": lon,
            "depth": 0.0,
        }  # Floods have no depth component

    def calculate_loss(self, event: dict[str, Any], exposure: dict[str, Any]) -> float:
        """Calculate flood loss."""
        water_depth = event["water_depth"]
        distance = self._calculate_distance(event, exposure)  # type: ignore[attr-defined]

        # Check if within affected area
        affected_area = event.get("affected_area", 50.0)
        if distance > affected_area:
            return 0.0

        # Distance factor
        distance_factor = 1.0 - (distance / affected_area)

        # Water depth factor
        water_factor = min(water_depth / 3.0, 1.0)

        # Vulnerability
        vulnerability = exposure.get("vulnerability", 0.7)

        total_loss = (
            exposure.get("value", 150000)
            * distance_factor
            * water_factor
            * vulnerability
        )

        return float(min(total_loss, exposure.get("value", 150000)))


# Factory functions
def create_enhanced_earthquake_model(
    config: CatastropheConfig | None = None,
) -> EnhancedEarthquakeModel:
    """Create an enhanced earthquake catastrophe model."""
    return EnhancedEarthquakeModel(config)


def create_enhanced_hurricane_model(
    config: CatastropheConfig | None = None,
) -> EnhancedHurricaneModel:
    """Create an enhanced hurricane catastrophe model."""
    return EnhancedHurricaneModel(config)


def create_enhanced_flood_model(
    config: CatastropheConfig | None = None,
) -> EnhancedFloodModel:
    """Create an enhanced flood catastrophe model."""
    return EnhancedFloodModel(config)


class CatastropheModelManager:
    """Manages a registry of catastrophe models for use in risk analysis.

    Provides methods to register, retrieve, and run catastrophe models
    across different hazard types.
    """

    def __init__(self, config: CatastropheConfig | None = None) -> None:
        self.config = config or CatastropheConfig()
        self._models: dict[str, EnhancedCatastropheModel] = {}
        self.hazard_interactions = MultiHazardInteractionMatrix(self.config.event_types)

    @property
    def models(self) -> dict[str, EnhancedCatastropheModel]:
        """Return the live model registry for compatibility with managers."""
        return self._models

    def register_model(self, name: str, model: EnhancedCatastropheModel) -> None:
        """Register a catastrophe model under a given name."""
        self._models[name] = model
        self.hazard_interactions.add_hazard(name)

    def get_model(self, name: str) -> EnhancedCatastropheModel | None:
        """Retrieve a registered model by name."""
        return self._models.get(name)

    def list_models(self) -> list[str]:
        """Return names of all registered models."""
        return list(self._models.keys())

    def configure_hazard_interactions(
        self,
        hazards: Sequence[str],
        interaction_matrix: Sequence[Sequence[float]],
    ) -> MultiHazardInteractionMatrix:
        """Replace the directed multi-hazard interaction matrix."""
        self.hazard_interactions = MultiHazardInteractionMatrix(
            hazards, interaction_matrix
        )
        return self.hazard_interactions

    def set_hazard_interaction(self, source: str, target: str, strength: float) -> None:
        """Set one directed cross-hazard interaction."""
        self.hazard_interactions.set_interaction(source, target, strength)

    def calculate_compound_exceedance_probability(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        hazard_sequence: Sequence[str] | None = None,
    ) -> float:
        """Calculate compound exceedance using the configured interactions."""
        return self.hazard_interactions.compound_exceedance_probability(
            exceedance_probabilities, hazard_sequence
        )

    def joint_exceedance_probability(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        hazard_sequences: Sequence[Sequence[str]],
    ) -> float:
        """Calculate the union exceedance across several compound paths."""
        return self.hazard_interactions.joint_exceedance_probability(
            exceedance_probabilities, hazard_sequences
        )

    def branch_exceedance_probabilities(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        max_path_length: int | None = None,
    ) -> dict[tuple[str, ...], float]:
        """Enumerate directed compound paths and their joint probabilities."""
        return self.hazard_interactions.branch_exceedance_probabilities(
            exceedance_probabilities, max_path_length
        )

    def dominant_exceedance_path(
        self,
        exceedance_probabilities: Mapping[str, float] | Sequence[float],
        max_path_length: int | None = None,
    ) -> tuple[tuple[str, ...], float]:
        """Return the compound path with the largest joint exceedance."""
        return self.hazard_interactions.dominant_exceedance_path(
            exceedance_probabilities, max_path_length
        )

    def run_all(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run all registered models against the given input data.

        Args:
            input_data: Input parameters for each model (keyed by model name).

        Returns:
            Results dictionary keyed by model name.
        """
        results: dict[str, Any] = {}
        for name, model in self._models.items():
            model_input = input_data.get(name, {})
            results[name] = model.run_analysis(model_input)  # type: ignore[attr-defined]
        return results
