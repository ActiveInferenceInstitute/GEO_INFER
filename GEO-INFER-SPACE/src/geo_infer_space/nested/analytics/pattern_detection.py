"""
Pattern Detection for H3 Nested Systems.

This module provides advanced pattern detection capabilities for identifying
spatial, temporal, and structural patterns in nested geospatial systems.
"""

import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
from enum import Enum
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import h3

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns that can be detected."""

    SPATIAL_CLUSTER = "spatial_cluster"
    TEMPORAL_TREND = "temporal_trend"
    PERIODIC = "periodic"
    ANOMALY = "anomaly"
    HOTSPOT = "hotspot"
    COLDSPOT = "coldspot"
    GRADIENT = "gradient"
    BOUNDARY = "boundary"
    FLOW = "flow"
    OSCILLATION = "oscillation"
    GROWTH = "growth"
    DECAY = "decay"
    CUSTOM = "custom"


class PatternScale(Enum):
    """Scale at which patterns are detected."""

    LOCAL = "local"
    REGIONAL = "regional"
    GLOBAL = "global"
    MULTI_SCALE = "multi_scale"


class DetectionMethod(Enum):
    """Methods for pattern detection."""

    STATISTICAL = "statistical"
    CLUSTERING = "clustering"
    SIGNAL_PROCESSING = "signal_processing"
    MACHINE_LEARNING = "machine_learning"
    SPATIAL_ANALYSIS = "spatial_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    CUSTOM = "custom"


@dataclass
class Pattern:
    """
    Represents a detected pattern.
    """

    pattern_id: str
    pattern_type: PatternType
    detection_method: DetectionMethod

    # Spatial properties
    affected_cells: set[str] = field(default_factory=set)
    center_cell: str | None = None
    spatial_extent: float = 0.0

    # Temporal properties
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: timedelta | None = None

    # Pattern properties
    strength: float = 0.0
    confidence: float = 0.0
    significance: float = 0.0

    # Pattern parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Statistical properties
    mean_value: float | None = None
    std_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None

    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    detection_context: dict[str, Any] = field(default_factory=dict)

    def get_pattern_summary(self) -> dict[str, Any]:
        """Get summary of pattern properties."""
        return {
            "pattern_id": self.pattern_id,
            "type": self.pattern_type.value,
            "method": self.detection_method.value,
            "cell_count": len(self.affected_cells),
            "strength": self.strength,
            "confidence": self.confidence,
            "significance": self.significance,
            "spatial_extent": self.spatial_extent,
            "duration": self.duration.total_seconds() if self.duration else None,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class PatternDetectionResult:
    """
    Result of pattern detection analysis.
    """

    analysis_id: str
    detected_patterns: list[Pattern] = field(default_factory=list)

    # Summary statistics
    pattern_counts: dict[PatternType, int] = field(default_factory=dict)
    detection_coverage: float = 0.0

    # Quality metrics
    overall_confidence: float = 0.0
    detection_reliability: float = 0.0

    # Method performance
    method_performance: dict[DetectionMethod, dict[str, float]] = field(
        default_factory=dict
    )

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    cells_analyzed: int = 0


class H3PatternDetector:
    """
    Advanced pattern detector for H3 nested systems.

    Provides comprehensive pattern detection capabilities including:
    - Spatial pattern detection (clusters, hotspots, gradients)
    - Temporal pattern detection (trends, cycles, anomalies)
    - Multi-scale pattern analysis
    - Statistical and ML-based detection methods
    """

    def __init__(self, name: str = "H3PatternDetector") -> None:
        """
        Initialize pattern detector.

        Args:
            name: Detector name for identification
        """
        self.name = name

        # Detection results
        self.detection_results: dict[str, PatternDetectionResult] = {}
        self.pattern_history: dict[str, list[Pattern]] = defaultdict(list)

        # Detection configuration
        self.detection_config: dict[str, Any] = {
            "min_pattern_strength": 0.3,
            "min_confidence": 0.5,
            "significance_threshold": 0.05,
            "spatial_neighborhood_size": 2,
            "temporal_window_size": 10,
            "clustering_eps": 0.5,
            "clustering_min_samples": 3,
        }

        # Custom detection functions
        self.custom_detectors: dict[str, Callable] = {}

        # Statistics
        self.detection_stats: dict[str, int] = defaultdict(int)

        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def register_custom_detector(self, name: str, detector_function: Callable) -> None:
        """
        Register a custom pattern detection function.

        Args:
            name: Detector name
            detector_function: Function that takes (cells, config) and returns patterns
        """
        self.custom_detectors[name] = detector_function
        self.updated_at = datetime.now()

    def detect_patterns(
        self,
        nested_grid: Any,
        system_id: str | None = None,
        pattern_types: list[PatternType] | None = None,
        methods: list[DetectionMethod] | None = None,
        **kwargs: Any,
    ) -> PatternDetectionResult:
        """
        Detect patterns in nested grid systems.

        Args:
            nested_grid: NestedH3Grid instance
            system_id: Specific system to analyze (None for all)
            pattern_types: Types of patterns to detect (None for all)
            methods: Detection methods to use (None for all)
            **kwargs: Detection parameters

        Returns:
            PatternDetectionResult instance
        """
        start_time = datetime.now()
        analysis_id = f"pattern_detection_{uuid.uuid4().hex[:8]}"

        # Get cells to analyze
        if system_id:
            system = nested_grid.get_system_by_id(system_id)
            if not system:
                raise ValueError(f"System {system_id} not found")
            cells = list(system.cells.values())
        else:
            cells = []
            for system in nested_grid.systems.values():
                cells.extend(system.cells.values())

        if not cells:
            return PatternDetectionResult(analysis_id=analysis_id, processing_time=0.0)

        # Default pattern types and methods
        if pattern_types is None:
            pattern_types = [
                PatternType.SPATIAL_CLUSTER,
                PatternType.HOTSPOT,
                PatternType.ANOMALY,
                PatternType.GRADIENT,
            ]

        if methods is None:
            methods = [DetectionMethod.STATISTICAL, DetectionMethod.CLUSTERING]

        # Detect patterns using different methods
        all_patterns: list[Pattern] = []

        for method in methods:
            try:
                if method == DetectionMethod.STATISTICAL:
                    patterns = self._detect_statistical_patterns(
                        cells, pattern_types, **kwargs
                    )
                elif method == DetectionMethod.CLUSTERING:
                    patterns = self._detect_clustering_patterns(
                        cells, pattern_types, **kwargs
                    )
                elif method == DetectionMethod.SIGNAL_PROCESSING:
                    patterns = self._detect_signal_patterns(
                        cells, pattern_types, **kwargs
                    )
                elif method == DetectionMethod.MACHINE_LEARNING:
                    patterns = self._detect_ml_patterns(cells, pattern_types, **kwargs)
                elif method == DetectionMethod.SPATIAL_ANALYSIS:
                    patterns = self._detect_spatial_patterns(
                        cells, pattern_types, **kwargs
                    )
                elif method == DetectionMethod.TEMPORAL_ANALYSIS:
                    patterns = self._detect_temporal_patterns(
                        cells, pattern_types, **kwargs
                    )
                elif method == DetectionMethod.CUSTOM:
                    patterns = self._detect_custom_patterns(
                        cells, pattern_types, **kwargs
                    )
                else:
                    continue  # type: ignore[unreachable]

                all_patterns.extend(patterns)

            except Exception as e:
                logger.warning(f"Pattern detection method {method.value} failed: {e}")
                continue

        # Filter and validate patterns
        validated_patterns = self._validate_patterns(all_patterns)

        # Calculate summary statistics
        pattern_counts: dict[PatternType, int] = defaultdict(int)
        for pattern in validated_patterns:
            pattern_counts[pattern.pattern_type] += 1

        # Calculate quality metrics
        overall_confidence = (
            sum(p.confidence for p in validated_patterns) / len(validated_patterns)
            if validated_patterns
            else 0.0
        )

        detection_coverage = len(
            set().union(*[p.affected_cells for p in validated_patterns])
        ) / len(cells)

        # Create result
        processing_time = (datetime.now() - start_time).total_seconds()

        result = PatternDetectionResult(
            analysis_id=analysis_id,
            detected_patterns=validated_patterns,
            pattern_counts=dict(pattern_counts),
            detection_coverage=detection_coverage,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            cells_analyzed=len(cells),
        )

        # Store result
        self.detection_results[analysis_id] = result

        # Update pattern history
        for pattern in validated_patterns:
            self.pattern_history[pattern.pattern_type.value].append(pattern)

        self.detection_stats["analyses_performed"] += 1
        self.updated_at = datetime.now()

        return result

    def _detect_statistical_patterns(
        self,
        cells: list[Any],
        pattern_types: list[PatternType],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect patterns using statistical methods."""
        patterns: list[Pattern] = []

        # Extract values for analysis
        values = []
        cell_indices = []

        value_field = kwargs.get("value_field", "value")

        for cell in cells:
            if value_field in cell.state_variables:
                values.append(cell.state_variables[value_field])
                cell_indices.append(cell.index)

        if not values:
            return patterns

        values_array = np.array(values)

        # Detect anomalies using statistical methods
        if PatternType.ANOMALY in pattern_types:
            anomaly_patterns = self._detect_statistical_anomalies(
                values_array, cell_indices, **kwargs
            )
            patterns.extend(anomaly_patterns)

        # Detect hotspots and coldspots
        if (
            PatternType.HOTSPOT in pattern_types
            or PatternType.COLDSPOT in pattern_types
        ):
            hotcold_patterns = self._detect_hotcold_spots(
                values_array, cell_indices, pattern_types, **kwargs
            )
            patterns.extend(hotcold_patterns)

        # Detect gradients
        if PatternType.GRADIENT in pattern_types:
            gradient_patterns = self._detect_statistical_gradients(
                values_array, cell_indices, cells, **kwargs
            )
            patterns.extend(gradient_patterns)

        return patterns

    def _detect_statistical_anomalies(
        self,
        values: np.ndarray,
        cell_indices: list[str],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect anomalies using statistical methods."""
        patterns: list[Pattern] = []

        if len(values) < 3:
            return patterns

        # Z-score based anomaly detection
        z_threshold = kwargs.get("z_threshold", 2.5)

        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return patterns

        z_scores = np.abs((values - mean_val) / std_val)
        anomaly_indices = np.where(z_scores > z_threshold)[0]

        for idx in anomaly_indices:
            pattern = Pattern(
                pattern_id=f"anomaly_{uuid.uuid4().hex[:8]}",
                pattern_type=PatternType.ANOMALY,
                detection_method=DetectionMethod.STATISTICAL,
                affected_cells={cell_indices[idx]},
                center_cell=cell_indices[idx],
                strength=float(z_scores[idx] / z_threshold),
                confidence=min(1.0, float(z_scores[idx] / z_threshold)),
                significance=float(1.0 - stats.norm.cdf(z_scores[idx])),
                parameters={"z_score": float(z_scores[idx]), "threshold": z_threshold},
            )
            patterns.append(pattern)

        return patterns

    def _detect_hotcold_spots(
        self,
        values: np.ndarray,
        cell_indices: list[str],
        pattern_types: list[PatternType],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect hotspots and coldspots."""
        patterns: list[Pattern] = []

        if len(values) < 3:
            return patterns

        percentile_threshold = kwargs.get("percentile_threshold", 90)

        # Hotspots (high values)
        if PatternType.HOTSPOT in pattern_types:
            hot_threshold = np.percentile(values, percentile_threshold)
            hot_indices = np.where(values >= hot_threshold)[0]

            if len(hot_indices) > 0:
                pattern = Pattern(
                    pattern_id=f"hotspot_{uuid.uuid4().hex[:8]}",
                    pattern_type=PatternType.HOTSPOT,
                    detection_method=DetectionMethod.STATISTICAL,
                    affected_cells=set(cell_indices[i] for i in hot_indices),
                    strength=float(np.mean(values[hot_indices]) / np.max(values)),
                    confidence=0.8,
                    parameters={
                        "threshold": float(hot_threshold),
                        "percentile": percentile_threshold,
                    },
                )
                patterns.append(pattern)

        # Coldspots (low values)
        if PatternType.COLDSPOT in pattern_types:
            cold_threshold = np.percentile(values, 100 - percentile_threshold)
            cold_indices = np.where(values <= cold_threshold)[0]

            if len(cold_indices) > 0:
                pattern = Pattern(
                    pattern_id=f"coldspot_{uuid.uuid4().hex[:8]}",
                    pattern_type=PatternType.COLDSPOT,
                    detection_method=DetectionMethod.STATISTICAL,
                    affected_cells=set(cell_indices[i] for i in cold_indices),
                    strength=float(
                        1.0 - np.mean(values[cold_indices]) / np.max(values)
                    ),
                    confidence=0.8,
                    parameters={
                        "threshold": float(cold_threshold),
                        "percentile": 100 - percentile_threshold,
                    },
                )
                patterns.append(pattern)

        return patterns

    def _detect_statistical_gradients(
        self,
        values: np.ndarray,
        cell_indices: list[str],
        cells: list[Any],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect gradients using statistical methods."""
        patterns: list[Pattern] = []

        if len(values) < 5:
            return patterns

        gradient_threshold = kwargs.get("gradient_threshold", 0.3)

        # Calculate gradients between neighboring cells
        cell_lookup = {cell.index: cell for cell in cells}
        value_lookup = {cell_indices[i]: values[i] for i in range(len(values))}

        gradient_cells = set()
        gradient_strengths = []

        for i, cell_idx in enumerate(cell_indices):
            if cell_idx not in cell_lookup:
                continue

            try:
                # Get neighbors
                neighbors = h3.grid_ring(cell_idx, 1)
                neighbor_values = []

                for neighbor_idx in neighbors:
                    if neighbor_idx in value_lookup:
                        neighbor_values.append(value_lookup[neighbor_idx])

                if len(neighbor_values) >= 3:
                    # Calculate gradient magnitude
                    gradient_magnitude = np.std(neighbor_values + [values[i]])

                    if gradient_magnitude > gradient_threshold:
                        gradient_cells.add(cell_idx)
                        gradient_strengths.append(gradient_magnitude)

            except Exception as e:
                logger.warning(f"Failed to calculate gradient for cell {cell_idx}: {e}")
                continue

        if gradient_cells:
            pattern = Pattern(
                pattern_id=f"gradient_{uuid.uuid4().hex[:8]}",
                pattern_type=PatternType.GRADIENT,
                detection_method=DetectionMethod.STATISTICAL,
                affected_cells=gradient_cells,
                strength=float(np.mean(gradient_strengths))
                if gradient_strengths
                else 0.0,
                confidence=0.7,
                parameters={"threshold": gradient_threshold},
            )
            patterns.append(pattern)

        return patterns

    def _detect_clustering_patterns(
        self,
        cells: list[Any],
        pattern_types: list[PatternType],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect patterns using clustering methods."""
        patterns: list[Pattern] = []

        if PatternType.SPATIAL_CLUSTER not in pattern_types:
            return patterns

        # Extract features for clustering
        features = []
        cell_indices = []

        feature_fields = kwargs.get("feature_fields", ["value"])

        for cell in cells:
            feature_vector = []
            for field_name in feature_fields:
                if field_name in cell.state_variables:
                    feature_vector.append(cell.state_variables[field_name])
                else:
                    feature_vector.append(0.0)

            if feature_vector:
                features.append(feature_vector)
                cell_indices.append(cell.index)

        if len(features) < 3:
            return patterns

        features_array = np.array(features)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)

        # DBSCAN clustering
        eps = kwargs.get("clustering_eps", self.detection_config["clustering_eps"])
        min_samples = kwargs.get(
            "clustering_min_samples", self.detection_config["clustering_min_samples"]
        )

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(features_scaled)

        # Create patterns for each cluster
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_cells = set(cell_indices[i] for i in cluster_indices)

            # Calculate cluster properties
            cluster_features = features_scaled[cluster_indices]
            cluster_spread = np.std(cluster_features)

            pattern = Pattern(
                pattern_id=f"cluster_{label}_{uuid.uuid4().hex[:8]}",
                pattern_type=PatternType.SPATIAL_CLUSTER,
                detection_method=DetectionMethod.CLUSTERING,
                affected_cells=cluster_cells,
                strength=1.0 / (1.0 + cluster_spread),
                confidence=len(cluster_cells) / len(cells),
                parameters={
                    "cluster_id": int(label),
                    "cluster_size": len(cluster_cells),
                    "cluster_spread": float(cluster_spread),
                },
            )
            patterns.append(pattern)

        return patterns

    def _detect_signal_patterns(
        self,
        cells: list[Any],
        pattern_types: list[PatternType],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect patterns using signal processing methods."""
        patterns: list[Pattern] = []

        # This would implement signal processing-based pattern detection
        # For now, return empty list
        return patterns

    def _detect_ml_patterns(
        self,
        cells: list[Any],
        pattern_types: list[PatternType],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect patterns using machine learning methods."""
        patterns: list[Pattern] = []

        # This would implement ML-based pattern detection
        # For now, return empty list
        return patterns

    def _detect_spatial_patterns(
        self,
        cells: list[Any],
        pattern_types: list[PatternType],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect spatial patterns."""
        patterns: list[Pattern] = []

        # This would implement spatial analysis-based pattern detection
        # For now, return empty list
        return patterns

    def _detect_temporal_patterns(
        self,
        cells: list[Any],
        pattern_types: list[PatternType],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect temporal patterns."""
        patterns: list[Pattern] = []

        # This would implement temporal analysis-based pattern detection
        # For now, return empty list
        return patterns

    def _detect_custom_patterns(
        self,
        cells: list[Any],
        pattern_types: list[PatternType],
        **kwargs: Any,
    ) -> list[Pattern]:
        """Detect patterns using custom detectors."""
        patterns: list[Pattern] = []

        for detector_name, detector_func in self.custom_detectors.items():
            try:
                custom_patterns = detector_func(cells, self.detection_config, **kwargs)
                if custom_patterns:
                    patterns.extend(custom_patterns)
            except Exception as e:
                logger.warning(f"Custom detector {detector_name} failed: {e}")
                continue

        return patterns

    def _validate_patterns(self, patterns: list[Pattern]) -> list[Pattern]:
        """Validate and filter detected patterns."""
        validated = []

        min_strength = self.detection_config["min_pattern_strength"]
        min_confidence = self.detection_config["min_confidence"]

        for pattern in patterns:
            if (
                pattern.strength >= min_strength
                and pattern.confidence >= min_confidence
                and len(pattern.affected_cells) > 0
            ):
                validated.append(pattern)

        return validated

    def get_pattern_statistics(self) -> dict[str, Any]:
        """Get pattern detector statistics."""
        total_patterns = sum(
            len(patterns) for patterns in self.pattern_history.values()
        )

        return {
            "detector_name": self.name,
            "total_analyses": self.detection_stats["analyses_performed"],
            "total_patterns_detected": total_patterns,
            "pattern_type_counts": {
                ptype: len(patterns) for ptype, patterns in self.pattern_history.items()
            },
            "custom_detectors": len(self.custom_detectors),
            "stored_results": len(self.detection_results),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
