"""
Flow Analysis for H3 Nested Systems.

This module provides comprehensive flow analysis capabilities for understanding
data and information flow patterns across boundaries and hierarchies in
nested geospatial systems.
"""

import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

try:
    import h3

    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    logger.warning("h3-py package not available")


class FlowType(Enum):
    """Types of flows in nested systems."""

    MASS = "mass"
    ENERGY = "energy"
    INFORMATION = "information"
    MATERIAL = "material"
    SIGNAL = "signal"
    RESOURCE = "resource"
    CUSTOM = "custom"


class FlowDirection(Enum):
    """Direction of flow."""

    INWARD = "inward"
    OUTWARD = "outward"
    BIDIRECTIONAL = "bidirectional"
    CIRCULAR = "circular"
    RANDOM = "random"


class FlowPattern(Enum):
    """Flow patterns in systems."""

    CONVERGENT = "convergent"
    DIVERGENT = "divergent"
    PARALLEL = "parallel"
    SPIRAL = "spiral"
    TURBULENT = "turbulent"
    LAMINAR = "laminar"
    OSCILLATORY = "oscillatory"


@dataclass
class FlowVector:
    """
    Represents a flow vector between cells.
    """

    source_cell: str
    target_cell: str
    flow_type: FlowType

    # Flow properties
    magnitude: float = 0.0
    direction: float = 0.0  # Angle in radians
    velocity: float = 0.0

    # Flow data
    flow_data: dict[str, Any] = field(default_factory=dict)

    # Temporal properties
    timestamp: datetime = field(default_factory=datetime.now)
    duration: timedelta | None = None

    # Quality metrics
    confidence: float = 1.0
    reliability: float = 1.0

    def get_flow_rate(self) -> float:
        """Calculate flow rate (magnitude/time)."""
        if self.duration and self.duration.total_seconds() > 0:
            return self.magnitude / self.duration.total_seconds()
        return self.magnitude


@dataclass
class FlowField:
    """
    Represents a flow field across multiple cells.
    """

    field_id: str
    flow_type: FlowType

    # Flow vectors
    vectors: dict[tuple[str, str], FlowVector] = field(default_factory=dict)

    # Field properties
    total_flow: float = 0.0
    average_velocity: float = 0.0
    dominant_direction: float = 0.0

    # Spatial properties
    coverage_area: float = 0.0
    cell_count: int = 0

    # Temporal properties
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Pattern analysis
    detected_patterns: list[FlowPattern] = field(default_factory=list)
    pattern_confidence: dict[FlowPattern, float] = field(default_factory=dict)

    def add_vector(self, vector: FlowVector) -> None:
        """Add a flow vector to the field."""
        key = (vector.source_cell, vector.target_cell)
        self.vectors[key] = vector
        self.last_updated = datetime.now()
        self._update_field_properties()

    def _update_field_properties(self) -> None:
        """Update field-level properties."""
        if not self.vectors:
            return

        # Calculate total flow and average velocity
        total_magnitude = sum(v.magnitude for v in self.vectors.values())
        total_velocity = sum(v.velocity for v in self.vectors.values())

        self.total_flow = total_magnitude
        self.average_velocity = total_velocity / len(self.vectors)
        self.cell_count = len(
            set(
                [v.source_cell for v in self.vectors.values()]
                + [v.target_cell for v in self.vectors.values()]
            )
        )

        # Calculate dominant direction (circular mean)
        directions = [v.direction for v in self.vectors.values()]
        sin_sum = np.sum(np.sin(directions))
        cos_sum = np.sum(np.cos(directions))
        self.dominant_direction = np.arctan2(sin_sum, cos_sum)


@dataclass
class FlowAnalysisResult:
    """
    Result of flow analysis.
    """

    analysis_id: str
    flow_fields: dict[str, FlowField] = field(default_factory=dict)

    # Summary metrics
    total_flows: int = 0
    total_flow_magnitude: float = 0.0
    average_flow_rate: float = 0.0

    # Pattern analysis
    dominant_patterns: list[FlowPattern] = field(default_factory=list)
    pattern_statistics: dict[FlowPattern, dict[str, float]] = field(
        default_factory=dict
    )

    # Network metrics
    flow_connectivity: float = 0.0
    flow_efficiency: float = 0.0
    flow_stability: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0


class H3FlowAnalyzer:
    """
    Advanced flow analyzer for H3 nested systems.

    Provides comprehensive analysis of flow patterns, including:
    - Flow vector calculation and tracking
    - Flow field analysis and visualization
    - Pattern detection and classification
    - Network flow metrics and optimization
    """

    def __init__(self, name: str = "H3FlowAnalyzer") -> None:
        """
        Initialize flow analyzer.

        Args:
            name: Analyzer name for identification
        """
        self.name = name

        # Flow data storage
        self.flow_fields: dict[str, FlowField] = {}
        self.flow_history: dict[str, list[FlowVector]] = defaultdict(list)

        # Analysis results
        self.analysis_results: dict[str, FlowAnalysisResult] = {}

        # Configuration
        self.analysis_config: dict[str, Any] = {
            "min_flow_magnitude": 0.01,
            "max_flow_age": timedelta(hours=24),
            "pattern_detection_threshold": 0.7,
            "velocity_smoothing_window": 5,
        }

        # Statistics
        self.analysis_stats: dict[str, int] = defaultdict(int)

        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def create_flow_field(self, field_id: str, flow_type: FlowType) -> FlowField:
        """
        Create a new flow field.

        Args:
            field_id: Unique field identifier
            flow_type: Type of flow

        Returns:
            Created FlowField instance
        """
        flow_field = FlowField(field_id=field_id, flow_type=flow_type)

        self.flow_fields[field_id] = flow_field
        self.updated_at = datetime.now()

        return flow_field

    def add_flow_vector(
        self,
        field_id: str,
        source_cell: str,
        target_cell: str,
        magnitude: float,
        direction: float = 0.0,
        velocity: float = 0.0,
        flow_data: dict[str, Any] | None = None,
    ) -> FlowVector:
        """
        Add a flow vector to a field.

        Args:
            field_id: Flow field ID
            source_cell: Source cell index
            target_cell: Target cell index
            magnitude: Flow magnitude
            direction: Flow direction (radians)
            velocity: Flow velocity
            flow_data: Additional flow data

        Returns:
            Created FlowVector instance
        """
        if field_id not in self.flow_fields:
            raise ValueError(f"Flow field {field_id} not found")

        flow_field = self.flow_fields[field_id]

        vector = FlowVector(
            source_cell=source_cell,
            target_cell=target_cell,
            flow_type=flow_field.flow_type,
            magnitude=magnitude,
            direction=direction,
            velocity=velocity,
            flow_data=flow_data or {},
        )

        flow_field.add_vector(vector)
        self.flow_history[field_id].append(vector)

        return vector

    def analyze_flow_patterns(self, field_id: str, **kwargs: Any) -> FlowAnalysisResult:
        """
        Analyze flow patterns in a field.

        Args:
            field_id: Flow field ID to analyze
            **kwargs: Analysis parameters

        Returns:
            FlowAnalysisResult instance
        """
        start_time = datetime.now()
        analysis_id = f"flow_analysis_{uuid.uuid4().hex[:8]}"

        if field_id not in self.flow_fields:
            raise ValueError(f"Flow field {field_id} not found")

        flow_field = self.flow_fields[field_id]

        # Analyze patterns
        patterns = self._detect_flow_patterns(flow_field, **kwargs)

        # Calculate network metrics
        connectivity = self._calculate_flow_connectivity(flow_field)
        efficiency = self._calculate_flow_efficiency(flow_field)
        stability = self._calculate_flow_stability(flow_field)

        # Calculate summary metrics
        total_flows = len(flow_field.vectors)
        total_magnitude = sum(v.magnitude for v in flow_field.vectors.values())
        avg_flow_rate = (
            sum(v.get_flow_rate() for v in flow_field.vectors.values()) / total_flows
            if total_flows > 0
            else 0.0
        )

        # Create result
        processing_time = (datetime.now() - start_time).total_seconds()

        result = FlowAnalysisResult(
            analysis_id=analysis_id,
            flow_fields={field_id: flow_field},
            total_flows=total_flows,
            total_flow_magnitude=total_magnitude,
            average_flow_rate=avg_flow_rate,
            dominant_patterns=patterns,
            flow_connectivity=connectivity,
            flow_efficiency=efficiency,
            flow_stability=stability,
            processing_time=processing_time,
        )

        # Store result
        self.analysis_results[analysis_id] = result
        self.analysis_stats["patterns_analyzed"] += 1
        self.updated_at = datetime.now()

        return result

    def _detect_flow_patterns(
        self, flow_field: FlowField, **kwargs: Any
    ) -> list[FlowPattern]:
        """Detect flow patterns in a field."""
        threshold = kwargs.get(
            "threshold", self.analysis_config["pattern_detection_threshold"]
        )

        detected_patterns: list[FlowPattern] = []
        pattern_scores = {}

        if not flow_field.vectors:
            return detected_patterns

        vectors = list(flow_field.vectors.values())

        # Analyze convergence/divergence
        convergence_score = self._analyze_convergence(vectors)
        divergence_score = self._analyze_divergence(vectors)

        if convergence_score > threshold:
            detected_patterns.append(FlowPattern.CONVERGENT)
            pattern_scores[FlowPattern.CONVERGENT] = convergence_score

        if divergence_score > threshold:
            detected_patterns.append(FlowPattern.DIVERGENT)
            pattern_scores[FlowPattern.DIVERGENT] = divergence_score

        # Analyze parallel flow
        parallel_score = self._analyze_parallel_flow(vectors)
        if parallel_score > threshold:
            detected_patterns.append(FlowPattern.PARALLEL)
            pattern_scores[FlowPattern.PARALLEL] = parallel_score

        # Analyze circular/spiral patterns
        circular_score = self._analyze_circular_flow(vectors)
        if circular_score > threshold:
            detected_patterns.append(FlowPattern.SPIRAL)
            pattern_scores[FlowPattern.SPIRAL] = circular_score

        # Analyze turbulence
        turbulence_score = self._analyze_turbulence(vectors)
        if turbulence_score > threshold:
            detected_patterns.append(FlowPattern.TURBULENT)
            pattern_scores[FlowPattern.TURBULENT] = turbulence_score

        # Store pattern confidence scores
        flow_field.pattern_confidence = pattern_scores
        flow_field.detected_patterns = detected_patterns

        return detected_patterns

    def _analyze_convergence(self, vectors: list[FlowVector]) -> float:
        """Analyze convergence patterns."""
        if not vectors:
            return 0.0

        # Count flows into each cell
        inflow_counts: dict[str, int] = defaultdict(int)
        outflow_counts: dict[str, int] = defaultdict(int)

        for vector in vectors:
            inflow_counts[vector.target_cell] += 1
            outflow_counts[vector.source_cell] += 1

        # Calculate convergence score
        total_cells = len(
            set([v.source_cell for v in vectors] + [v.target_cell for v in vectors])
        )

        if total_cells == 0:
            return 0.0

        # High inflow, low outflow indicates convergence
        convergence_cells = 0
        for cell in inflow_counts:
            inflow = inflow_counts[cell]
            outflow = outflow_counts.get(cell, 0)

            if inflow > outflow and inflow > 1:
                convergence_cells += 1

        return convergence_cells / total_cells

    def _analyze_divergence(self, vectors: list[FlowVector]) -> float:
        """Analyze divergence patterns."""
        if not vectors:
            return 0.0

        # Count flows from each cell
        inflow_counts: dict[str, int] = defaultdict(int)
        outflow_counts: dict[str, int] = defaultdict(int)

        for vector in vectors:
            inflow_counts[vector.target_cell] += 1
            outflow_counts[vector.source_cell] += 1

        # Calculate divergence score
        total_cells = len(
            set([v.source_cell for v in vectors] + [v.target_cell for v in vectors])
        )

        if total_cells == 0:
            return 0.0

        # High outflow, low inflow indicates divergence
        divergence_cells = 0
        for cell in outflow_counts:
            outflow = outflow_counts[cell]
            inflow = inflow_counts.get(cell, 0)

            if outflow > inflow and outflow > 1:
                divergence_cells += 1

        return divergence_cells / total_cells

    def _analyze_parallel_flow(self, vectors: list[FlowVector]) -> float:
        """Analyze parallel flow patterns."""
        if not vectors:
            return 0.0

        directions = [v.direction for v in vectors]

        if len(directions) < 2:
            return 0.0

        # Calculate direction variance
        directions_array = np.array(directions)

        # Handle circular nature of angles
        sin_directions = np.sin(directions_array)
        cos_directions = np.cos(directions_array)

        sin_var = np.var(sin_directions)
        cos_var = np.var(cos_directions)

        # Low variance indicates parallel flow
        total_variance = sin_var + cos_var
        parallel_score = 1.0 / (1.0 + total_variance)

        return cast(float, parallel_score)

    def _analyze_circular_flow(self, vectors: list[FlowVector]) -> float:
        """Analyze circular/spiral flow patterns."""
        if not vectors or not H3_AVAILABLE:
            return 0.0

        # This would analyze spatial arrangement of vectors for circular patterns
        # For now, return a simple heuristic
        return 0.0

    def _analyze_turbulence(self, vectors: list[FlowVector]) -> float:
        """Analyze turbulent flow patterns."""
        if not vectors:
            return 0.0

        # High variance in direction and magnitude indicates turbulence
        directions = [v.direction for v in vectors]
        magnitudes = [v.magnitude for v in vectors]

        if len(directions) < 3:
            return 0.0

        direction_var = np.var(directions)
        magnitude_var = np.var(magnitudes)

        # Normalize variances
        direction_turbulence = min(1.0, direction_var / (np.pi**2))
        magnitude_turbulence = min(1.0, magnitude_var / np.var(magnitudes))

        return (
            cast(float, direction_turbulence) + cast(float, magnitude_turbulence)
        ) / 2.0

    def _calculate_flow_connectivity(self, flow_field: FlowField) -> float:
        """Calculate flow connectivity metric."""
        if not flow_field.vectors:
            return 0.0

        # Get unique cells
        all_cells = set()
        for vector in flow_field.vectors.values():
            all_cells.add(vector.source_cell)
            all_cells.add(vector.target_cell)

        total_cells = len(all_cells)
        connected_pairs = len(flow_field.vectors)

        if total_cells < 2:
            return 0.0

        # Maximum possible connections
        max_connections = total_cells * (total_cells - 1)

        return connected_pairs / max_connections if max_connections > 0 else 0.0

    def _calculate_flow_efficiency(self, flow_field: FlowField) -> float:
        """Calculate flow efficiency metric."""
        if not flow_field.vectors:
            return 0.0

        # Efficiency based on flow magnitude vs. path length
        total_efficiency = 0.0

        for vector in flow_field.vectors.values():
            if H3_AVAILABLE:
                try:
                    # Calculate H3 distance
                    distance = h3.grid_distance(vector.source_cell, vector.target_cell)
                    if distance > 0:
                        efficiency = vector.magnitude / distance
                        total_efficiency += efficiency
                except Exception:
                    total_efficiency += vector.magnitude
            else:
                total_efficiency += vector.magnitude

        return total_efficiency / len(flow_field.vectors)

    def _calculate_flow_stability(self, flow_field: FlowField) -> float:
        """Calculate flow stability metric."""
        field_history = self.flow_history.get(flow_field.field_id, [])

        if len(field_history) < 2:
            return 1.0  # Assume stable if no history

        # Analyze temporal stability of flow patterns
        recent_vectors = field_history[-10:]  # Last 10 vectors

        if not recent_vectors:
            return 1.0

        # Calculate variance in magnitude and direction over time
        magnitudes = [v.magnitude for v in recent_vectors]
        directions = [v.direction for v in recent_vectors]

        magnitude_stability = 1.0 / (1.0 + np.var(magnitudes))
        direction_stability = 1.0 / (1.0 + np.var(directions))

        return cast(float, (magnitude_stability + direction_stability) / 2.0)

    def get_flow_statistics(self) -> dict[str, Any]:
        """Get flow analyzer statistics."""
        total_fields = len(self.flow_fields)
        total_vectors = sum(len(field.vectors) for field in self.flow_fields.values())

        return {
            "analyzer_name": self.name,
            "total_flow_fields": total_fields,
            "total_flow_vectors": total_vectors,
            "analysis_stats": dict(self.analysis_stats),
            "stored_results": len(self.analysis_results),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
