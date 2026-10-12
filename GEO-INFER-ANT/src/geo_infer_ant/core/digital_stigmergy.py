"""
Digital Stigmergy Systems for GEO-INFER-ANT

This module implements digital stigmergy systems that enable indirect coordination
through digital traces and shared information spaces. These systems are the modern
equivalent of pheromone-based communication for urban environments, IoT networks,
and digital platforms.

Key Features:
- Multi-platform digital communication (IoT, social media, public displays)
- Temporal information persistence and decay
- Spatial information indexing and querying
- Information quality and credibility assessment
- Multi-modal information types (text, sensor data, images, locations)
- Privacy and access control mechanisms
- Integration with real-time data streams
"""

import numpy as np
import logging
from typing import Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import uuid

from geo_infer_ant.utils.spatial import parse_h3_resolution, validate_bounds

# Integration imports
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
except ImportError as e:
    logging.getLogger(__name__).debug("Optional spatial integration unavailable: %s", e)
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

logger = logging.getLogger(__name__)


@dataclass
class DigitalTrace:
    """Digital information trace left by agents in the environment."""

    trace_id: str
    agent_id: str
    information_type: str
    content: dict[str, Any]
    location: np.ndarray | None = None  # [lat, lng] if spatial
    timestamp: datetime = field(default_factory=datetime.now)
    visibility_scope: str = "public"  # 'public', 'private', 'neighborhood', 'network'
    persistence_duration: float = 3600.0  # seconds
    credibility_score: float = 1.0
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate trace after initialization."""
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
        if self.credibility_score < 0 or self.credibility_score > 1:
            self.credibility_score = max(0.0, min(1.0, self.credibility_score))

    def is_expired(self) -> bool:
        """Check if trace has exceeded its persistence duration."""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.persistence_duration

    def get_credibility_weight(self) -> float:
        """Get credibility-weighted value for decision making."""
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600.0
        # Credibility decays over time
        time_decay = max(0.1, 1.0 - (age_hours / 24.0))  # 24-hour half-life
        return self.credibility_score * time_decay

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "information_type": self.information_type,
            "content": self.content,
            "location": self.location.tolist() if self.location is not None else None,
            "timestamp": self.timestamp.isoformat(),
            "visibility_scope": self.visibility_scope,
            "persistence_duration": self.persistence_duration,
            "credibility_score": self.credibility_score,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


@dataclass
class InformationQuery:
    """Query for digital stigmergic information."""

    query_id: str
    agent_id: str
    query_type: str  # 'resource_location', 'hazard_warning', 'traffic_info', etc.
    spatial_bounds: dict[str, float] | None = None
    temporal_window: str | None = "recent"  # 'recent', 'hour', 'day', 'week', 'all'
    information_types: list[str] = field(default_factory=list)
    credibility_threshold: float = 0.5
    max_results: int = 10
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert query to dictionary representation."""
        return {
            "query_id": self.query_id,
            "agent_id": self.agent_id,
            "query_type": self.query_type,
            "spatial_bounds": self.spatial_bounds,
            "temporal_window": self.temporal_window,
            "information_types": self.information_types,
            "credibility_threshold": self.credibility_threshold,
            "max_results": self.max_results,
            "timestamp": self.timestamp.isoformat(),
        }


class DigitalStigmergy:
    """
    Digital stigmergy system for modern agent coordination.

    Provides a comprehensive platform for digital indirect communication
    through shared information spaces, enabling agents to coordinate
    through digital traces, shared knowledge, and real-time information
    exchange.

    Key Features:
    - Multi-platform information sharing
    - Temporal information management
    - Spatial information indexing
    - Quality and credibility assessment
    - Privacy and access control
    - Real-time information streams
    """

    def __init__(
        self,
        communication_medium: str = "iot_network",
        information_types: list[str] | None = None,
        persistence_model: str = "temporal_decay",
        access_control: str = "public",
        spatial_backend: str = "h3",
    ):
        """
        Initialize digital stigmergy system.

        Args:
            communication_medium: Primary communication platform
            information_types: Types of information to handle
            persistence_model: How information persists over time
            access_control: Access control level
            spatial_backend: Backend for spatial operations
        """
        self.communication_medium = communication_medium
        self.information_types = information_types or [
            "sensor_data",
            "alerts",
            "coordination",
        ]
        self.persistence_model = persistence_model
        self.access_control = access_control
        self.spatial_resolution = parse_h3_resolution("h3_r8")

        # Information storage
        self.digital_traces: dict[str, DigitalTrace] = {}  # trace_id -> trace
        self.trace_index: dict[str, list[str]] = defaultdict(list)  # type -> trace_ids
        self.spatial_index: dict[str, list[str]] = defaultdict(
            list
        )  # spatial_cell -> trace_ids
        self.agent_traces: dict[str, list[str]] = defaultdict(
            list
        )  # agent_id -> trace_ids

        # Query history and analytics
        self.query_history: list[InformationQuery] = []
        self.access_patterns: dict[str, int] = defaultdict(int)

        # Integration components
        self.spatial_indexer: Any = None
        self.spatial_analytics: Any = None

        # Performance tracking
        self.performance_stats = {
            "traces_total": 0,
            "queries_total": 0,
            "avg_query_response_time": 0.0,
            "information_quality_score": 1.0,
        }

        # Initialize integrations
        self._initialize_integrations(spatial_backend)

        logger.info(f"DigitalStigmergy initialized for medium: {communication_medium}")

    def _initialize_integrations(self, backend: str) -> None:
        """Initialize spatial integration components."""
        if SpatialIndexingInterface:
            try:
                self.spatial_indexer = SpatialIndexingInterface(backend=backend)
                logger.info(f"Spatial indexer initialized with {backend} backend")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial indexer: {e}")

        if SpatialAnalyticsInterface:
            try:
                self.spatial_analytics = SpatialAnalyticsInterface(backend=backend)
                logger.info(f"Spatial analytics initialized with {backend} backend")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial analytics: {e}")

    async def contribute_information(
        self,
        agent_id: str,
        information_type: str,
        content: dict[str, Any],
        location: np.ndarray | None = None,
        visibility_scope: str = "public",
        persistence_duration: float = 3600.0,
        credibility_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Contribute information to the digital stigmergy system.

        Args:
            agent_id: ID of contributing agent
            information_type: Type of information being shared
            content: Information content
            location: Spatial location if applicable
            visibility_scope: Who can access this information
            persistence_duration: How long information should persist
            credibility_score: Credibility rating (0-1)
            metadata: Additional metadata

        Returns:
            Trace ID of the created information trace
        """
        start_time = datetime.now()

        try:
            # Validate information type
            if information_type not in self.information_types:
                logger.error(f"Unknown information type: {information_type}")
                return ""

            # Auto-calculate credibility if not provided
            if credibility_score is None:
                credibility_score = self._calculate_credibility_score(
                    agent_id, information_type, content
                )

            # Create digital trace
            trace = DigitalTrace(
                trace_id=str(uuid.uuid4()),
                agent_id=agent_id,
                information_type=information_type,
                content=content,
                location=location.copy() if location is not None else None,
                visibility_scope=visibility_scope,
                persistence_duration=persistence_duration,
                credibility_score=credibility_score,
                metadata=metadata or {},
            )

            # Store trace
            self.digital_traces[trace.trace_id] = trace

            # Update indices
            self.trace_index[information_type].append(trace.trace_id)
            self.agent_traces[agent_id].append(trace.trace_id)

            if location is not None and self.spatial_indexer:
                try:
                    cell_id = self.spatial_indexer.latlng_to_cell(
                        location[0], location[1], self.spatial_resolution
                    )
                    self.spatial_index[cell_id].append(trace.trace_id)
                except Exception as e:
                    logger.warning(f"Failed to index trace spatially: {e}")

            # Update performance stats
            self.performance_stats["traces_total"] += 1

            response_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats["avg_query_response_time"] = (
                self.performance_stats["avg_query_response_time"] + response_time
            ) / 2

            logger.debug(f"Information contributed by {agent_id}: {information_type}")
            return trace.trace_id

        except Exception as e:
            logger.error(f"Failed to contribute information: {e}")
            return ""

    def _calculate_credibility_score(
        self, agent_id: str, information_type: str, content: dict[str, Any]
    ) -> float:
        """Calculate credibility score for information contribution."""
        base_credibility = 1.0

        # Agent reputation: starts at 1.0 for new agents, improves with experience
        agent_trace_count = len(self.agent_traces.get(agent_id, []))
        reputation_factor = min(1.0, 0.8 + (agent_trace_count / 100.0))
        base_credibility *= reputation_factor

        # Information type reliability
        reliability_factors = {
            "environmental_data": 0.9,
            "resource_discovery": 0.8,
            "hazard_warning": 0.9,
            "traffic_info": 0.7,
            "social_coordination": 0.6,
            "task_status": 0.8,
            "sensor_data": 0.85,
            "alerts": 0.9,
            "coordination": 0.75,
        }
        reliability = reliability_factors.get(information_type, 0.7)
        base_credibility *= reliability

        # Content completeness factor
        if not content:
            completeness = 0.0
        else:
            present_values = sum(
                value is not None and value != "" for value in content.values()
            )
            completeness = present_values / len(content)
        base_credibility *= 0.7 + 0.3 * completeness

        return min(1.0, base_credibility)

    async def query_stigmergy(
        self,
        agent_id: str,
        query_type: str,
        spatial_bounds: dict[str, float] | None = None,
        temporal_window: str = "recent",
        information_types: list[str] | None = None,
        credibility_threshold: float = 0.5,
        max_results: int = 10,
    ) -> list[DigitalTrace]:
        """
        Query digital stigmergic information.

        Args:
            agent_id: ID of querying agent
            query_type: Type of information being sought
            spatial_bounds: Spatial bounds for search
            temporal_window: Time window for information
            information_types: Types of information to include
            credibility_threshold: Minimum credibility required
            max_results: Maximum number of results to return

        Returns:
            List of relevant digital traces
        """
        start_time = datetime.now()

        try:
            # Create query record
            query = InformationQuery(
                query_id=str(uuid.uuid4()),
                agent_id=agent_id,
                query_type=query_type,
                spatial_bounds=spatial_bounds,
                temporal_window=temporal_window,
                information_types=information_types or self.information_types,
                credibility_threshold=credibility_threshold,
                max_results=max_results,
            )

            self.query_history.append(query)

            # Filter traces based on criteria
            relevant_traces = self._filter_traces(query)

            # Sort by relevance and credibility
            relevant_traces.sort(
                key=lambda t: (t.get_credibility_weight(), t.access_count), reverse=True
            )

            # Limit results
            results = relevant_traces[:max_results]

            # Update access patterns
            for trace in results:
                trace.access_count += 1
                self.access_patterns[trace.information_type] += 1

            # Update performance stats
            response_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats["queries_total"] += 1
            self.performance_stats["avg_query_response_time"] = (
                self.performance_stats["avg_query_response_time"] + response_time
            ) / 2

            logger.debug(f"Query by {agent_id} returned {len(results)} traces")
            return results

        except Exception as e:
            logger.error(f"Failed to query stigmergy: {e}")
            return []

    def _filter_traces(self, query: InformationQuery) -> list[DigitalTrace]:
        """Filter traces based on query criteria."""
        filtered_traces = []

        # Get candidate traces by information type first.
        candidate_trace_ids = set()

        # Filter by information types
        for info_type in query.information_types:
            if info_type in self.trace_index:
                candidate_trace_ids.update(self.trace_index[info_type])

        # Filter by spatial bounds
        if query.spatial_bounds:
            bounds = validate_bounds(query.spatial_bounds)
            spatial_trace_ids = {
                trace_id
                for trace_id in candidate_trace_ids
                if self._location_in_bounds(self.digital_traces.get(trace_id), bounds)
            }
            # Cell-center polygon covers are not a complete candidate set:
            # tiny boxes can cover no center, and large covers exceed budgets.
            # The coordinate predicate above is exact and allocates no cover.
            candidate_trace_ids = spatial_trace_ids

        # Filter by temporal window
        temporal_cutoff = self._get_temporal_cutoff(query.temporal_window)

        # Apply all filters
        for trace_id in candidate_trace_ids:
            if trace_id not in self.digital_traces:
                continue

            trace = self.digital_traces[trace_id]

            # Check expiration
            if trace.is_expired():
                # Remove expired trace
                self._remove_trace(trace_id)
                continue

            # Check temporal window
            if trace.timestamp < temporal_cutoff:
                continue

            # Check credibility
            if trace.get_credibility_weight() < query.credibility_threshold:
                continue

            # Check access control
            if not self._check_access_control(
                query.agent_id, trace, query.spatial_bounds
            ):
                continue

            filtered_traces.append(trace)

        return filtered_traces

    def _get_spatial_cells(self, bounds: dict[str, float]) -> list[str]:
        """Convert spatial bounds to H3 cells."""
        if not self.spatial_indexer:
            return []

        try:
            normalized = validate_bounds(bounds)
            polygon = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [normalized["min_lng"], normalized["min_lat"]],
                        [normalized["max_lng"], normalized["min_lat"]],
                        [normalized["max_lng"], normalized["max_lat"]],
                        [normalized["min_lng"], normalized["max_lat"]],
                        [normalized["min_lng"], normalized["min_lat"]],
                    ]
                ],
            }
            if hasattr(self.spatial_indexer, "polygon_to_cells"):
                return list(
                    self.spatial_indexer.polygon_to_cells(
                        polygon, self.spatial_resolution
                    )
                )
            center_lat = (normalized["min_lat"] + normalized["max_lat"]) / 2
            center_lng = (normalized["min_lng"] + normalized["max_lng"]) / 2
            return [
                self.spatial_indexer.latlng_to_cell(
                    center_lat, center_lng, self.spatial_resolution
                )
            ]

        except Exception as e:
            logger.warning(f"Failed to get spatial cells: {e}")
            return []

    def _get_temporal_cutoff(self, temporal_window: str | None) -> datetime:
        """Get temporal cutoff based on window specification."""
        now = datetime.now()

        windows = {
            "recent": timedelta(hours=1),
            "hour": timedelta(hours=1),
            "day": timedelta(days=1),
            "week": timedelta(weeks=1),
            "all": timedelta(days=365 * 10),  # Long time in the past
        }

        if temporal_window is None:
            temporal_window = "all"
        if temporal_window not in windows:
            raise ValueError(f"Unsupported temporal window: {temporal_window}")
        window_delta = windows[temporal_window]
        return now - window_delta

    @staticmethod
    def _location_in_bounds(
        trace: DigitalTrace | None, bounds: dict[str, float]
    ) -> bool:
        if trace is None or trace.location is None:
            return False
        lat, lng = np.asarray(trace.location, dtype=float)
        return bool(
            bounds["min_lat"] <= lat <= bounds["max_lat"]
            and bounds["min_lng"] <= lng <= bounds["max_lng"]
        )

    def _check_access_control(
        self,
        agent_id: str,
        trace: DigitalTrace,
        spatial_bounds: dict[str, float] | None = None,
    ) -> bool:
        """Check if agent has access to trace based on visibility scope."""
        if trace.visibility_scope == "public":
            return True
        elif trace.visibility_scope == "neighborhood":
            return spatial_bounds is not None and self._location_in_bounds(
                trace, validate_bounds(spatial_bounds)
            )
        elif trace.visibility_scope == "private":
            # Only agent who created it can access
            return trace.agent_id == agent_id
        elif trace.visibility_scope == "network":
            allowed_agents = trace.metadata.get("allowed_agents", [])
            return agent_id in allowed_agents

        return False

    def _remove_trace(self, trace_id: str) -> None:
        """Remove a trace from all indices."""
        if trace_id not in self.digital_traces:
            return

        trace = self.digital_traces[trace_id]

        # Remove from main storage
        del self.digital_traces[trace_id]

        # Remove from type index
        if trace.information_type in self.trace_index:
            self.trace_index[trace.information_type].remove(trace_id)

        # Remove from agent traces
        if trace.agent_id in self.agent_traces:
            self.agent_traces[trace.agent_id].remove(trace_id)

        # Remove from spatial index
        if trace.location is not None and self.spatial_indexer:
            try:
                cell_id = self.spatial_indexer.latlng_to_cell(
                    trace.location[0], trace.location[1], self.spatial_resolution
                )
                if cell_id in self.spatial_index:
                    self.spatial_index[cell_id].remove(trace_id)
            except Exception as e:
                logger.warning(f"Failed to remove from spatial index: {e}")

    async def extract_patterns(
        self,
        information_contributions: list[DigitalTrace] | None = None,
        pattern_types: list[str] | None = None,
        temporal_analysis: str = "recent",
    ) -> dict[str, Any]:
        """
        Extract emergent patterns from digital stigmergy contributions.

        Args:
            information_contributions: Traces to analyze (all if None)
            pattern_types: Types of patterns to extract
            temporal_analysis: Temporal scope for analysis

        Returns:
            Extracted patterns and insights
        """
        try:
            if information_contributions is None:
                contributions = list(self.digital_traces.values())
            else:
                contributions = information_contributions

            if not contributions:
                return {"status": "no_data"}

            pattern_types = pattern_types or [
                "clusters",
                "flows",
                "anomalies",
                "trends",
            ]
            patterns: dict[str, Any] = {}

            # Spatial clustering analysis
            if "clusters" in pattern_types and self.spatial_analytics:
                try:
                    spatial_clusters = self._analyze_spatial_clusters(contributions)
                    patterns["spatial_clusters"] = spatial_clusters
                except Exception as e:
                    logger.warning(f"Spatial clustering analysis failed: {e}")

            # Information flow analysis
            if "flows" in pattern_types:
                try:
                    information_flows = self._analyze_information_flows(contributions)
                    patterns["information_flows"] = information_flows
                except Exception as e:
                    logger.warning(f"Information flow analysis failed: {e}")

            # Anomaly detection
            if "anomalies" in pattern_types:
                try:
                    anomalies = self._detect_anomalies(contributions)
                    patterns["anomalies"] = anomalies
                except Exception as e:
                    logger.warning(f"Anomaly detection failed: {e}")

            # Temporal trends
            if "trends" in pattern_types:
                try:
                    trends = self._analyze_temporal_trends(
                        contributions, temporal_analysis
                    )
                    patterns["temporal_trends"] = trends
                except Exception as e:
                    logger.warning(f"Temporal trend analysis failed: {e}")

            patterns["status"] = "success"
            logger.debug(
                f"Extracted {len(patterns)} pattern types from {len(contributions)} contributions"
            )
            return patterns

        except Exception as e:
            logger.error(f"Failed to extract patterns: {e}")
            return {"error": str(e)}

    def _analyze_spatial_clusters(
        self, contributions: list[DigitalTrace]
    ) -> dict[str, Any]:
        """Analyze spatial clustering of information contributions."""
        if not self.spatial_analytics:
            return {"status": "spatial_analytics_unavailable"}

        # Extract spatial contributions
        spatial_traces = [t for t in contributions if t.location is not None]

        if len(spatial_traces) < 3:
            return {"status": "insufficient_spatial_data"}

        # Convert to spatial data for analysis
        locations = np.array([t.location for t in spatial_traces])

        try:
            # Use spatial analytics for clustering
            clusters = self.spatial_analytics.analyze_clusters(
                data=locations,
                method="kmeans",
                n_clusters=min(5, len(locations) // 10 + 1),
            )

            # Analyze information types in each cluster
            cluster_info = {}
            for i, cluster in enumerate(clusters["clusters"]):
                cluster_traces = [spatial_traces[j] for j in cluster["indices"]]

                info_types: defaultdict[str, int] = defaultdict(int)
                for trace in cluster_traces:
                    info_types[trace.information_type] += 1

                cluster_info[f"cluster_{i}"] = {
                    "center": cluster["center"].tolist(),
                    "size": len(cluster_traces),
                    "information_types": dict(info_types),
                    "avg_credibility": np.mean(
                        [t.credibility_score for t in cluster_traces]
                    ),
                }

            return {
                "n_clusters": len(clusters["clusters"]),
                "cluster_details": cluster_info,
                "total_contributions": len(spatial_traces),
            }

        except Exception as e:
            logger.warning(f"Spatial clustering failed: {e}")
            return {"status": "clustering_failed"}

    def _analyze_information_flows(
        self, contributions: list[DigitalTrace]
    ) -> dict[str, Any]:
        """Analyze information flows and sharing patterns."""
        flows: dict[str, Any] = {
            "information_type_flows": defaultdict(list),
            "agent_contribution_patterns": defaultdict(list),
            "temporal_flows": defaultdict(list),
        }

        # Analyze flows by information type
        for trace in contributions:
            flows["information_type_flows"][trace.information_type].append(
                {
                    "timestamp": trace.timestamp,
                    "agent_id": trace.agent_id,
                    "location": (
                        trace.location.tolist() if trace.location is not None else None
                    ),
                    "credibility": trace.credibility_score,
                }
            )

        # Analyze agent contribution patterns
        for trace in contributions:
            flows["agent_contribution_patterns"][trace.agent_id].append(
                {
                    "timestamp": trace.timestamp,
                    "information_type": trace.information_type,
                    "location": (
                        trace.location.tolist() if trace.location is not None else None
                    ),
                }
            )

        # Analyze temporal flows
        for trace in contributions:
            hour = trace.timestamp.hour
            flows["temporal_flows"][f"hour_{hour}"].append(
                {"information_type": trace.information_type, "agent_id": trace.agent_id}
            )

        return dict(flows)

    def _detect_anomalies(self, contributions: list[DigitalTrace]) -> dict[str, Any]:
        """Detect anomalous patterns in information contributions."""
        anomalies: dict[str, Any] = {
            "unusual_activity_spikes": [],
            "low_credibility_clusters": [],
            "spatial_anomalies": [],
            "temporal_anomalies": [],
        }

        if len(contributions) < 10:
            return anomalies

        try:
            # Detect temporal spikes in activity
            hourly_activity: defaultdict[int, int] = defaultdict(int)
            for trace in contributions:
                hour = trace.timestamp.hour
                hourly_activity[hour] += 1

            avg_activity = np.mean(list(hourly_activity.values()))
            std_activity = np.std(list(hourly_activity.values()))

            for hour, activity in hourly_activity.items():
                if activity > avg_activity + 2 * std_activity:
                    anomalies["unusual_activity_spikes"].append(
                        {
                            "hour": hour,
                            "activity": activity,
                            "deviation": (activity - avg_activity) / std_activity,
                        }
                    )

            # Detect low credibility information clusters
            low_cred_traces = [t for t in contributions if t.credibility_score < 0.3]

            if len(low_cred_traces) > 3 and self.spatial_analytics:
                # Spatial clustering of low credibility info
                locations = np.array(
                    [t.location for t in low_cred_traces if t.location is not None]
                )
                if len(locations) > 2:
                    # Simple clustering for anomaly detection
                    from sklearn.cluster import DBSCAN

                    clustering = DBSCAN(eps=0.01, min_samples=3).fit(locations)

                    if len(set(clustering.labels_)) > 1:  # Found clusters
                        for label in set(clustering.labels_):
                            if label != -1:  # Not noise
                                cluster_traces = [
                                    low_cred_traces[i]
                                    for i in range(len(low_cred_traces))
                                    if clustering.labels_[i] == label
                                ]

                                anomalies["low_credibility_clusters"].append(
                                    {
                                        "cluster_id": label,
                                        "size": len(cluster_traces),
                                        "avg_credibility": np.mean(
                                            [
                                                t.credibility_score
                                                for t in cluster_traces
                                            ]
                                        ),
                                        "information_types": list(
                                            set(
                                                [
                                                    t.information_type
                                                    for t in cluster_traces
                                                ]
                                            )
                                        ),
                                    }
                                )

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

        return anomalies

    def _analyze_temporal_trends(
        self, contributions: list[DigitalTrace], temporal_scope: str
    ) -> dict[str, Any]:
        """Analyze temporal trends in information contributions."""
        trends: dict[str, Any] = {
            "information_type_trends": defaultdict(list),
            "activity_trends": defaultdict(int),
            "credibility_trends": defaultdict(list),
        }

        # Group by time periods
        time_groups = self._group_by_time_period(contributions, temporal_scope)

        for period, period_traces in time_groups.items():
            # Information type trends
            type_counts: defaultdict[str, int] = defaultdict(int)
            for trace in period_traces:
                type_counts[trace.information_type] += 1

            trends["information_type_trends"][period] = dict(type_counts)
            trends["activity_trends"][period] = len(period_traces)

            # Credibility trends
            if period_traces:
                avg_credibility = np.mean([t.credibility_score for t in period_traces])
                trends["credibility_trends"][period] = avg_credibility

        return dict(trends)

    def _group_by_time_period(
        self, contributions: list[DigitalTrace], scope: str
    ) -> dict[str, list[DigitalTrace]]:
        """Group contributions by time period."""
        groups = defaultdict(list)

        if scope == "hourly":
            for trace in contributions:
                period = trace.timestamp.strftime("%Y-%m-%d-%H")
                groups[period].append(trace)
        elif scope == "daily":
            for trace in contributions:
                period = trace.timestamp.strftime("%Y-%m-%d")
                groups[period].append(trace)
        elif scope == "weekly":
            for trace in contributions:
                period = trace.timestamp.strftime("%Y-W%U")
                groups[period].append(trace)

        return dict(groups)

    async def update_information_quality(self) -> float:
        """Update and return overall information quality score."""
        try:
            if not self.digital_traces:
                return 1.0

            # Calculate quality metrics
            total_traces = len(self.digital_traces)
            # Credibility score
            active = [t for t in self.digital_traces.values() if not t.is_expired()]
            if not active:
                self.performance_stats["information_quality_score"] = 0.0
                await self._cleanup_expired_traces()
                return 0.0
            avg_credibility = np.mean([t.credibility_score for t in active])

            # Diversity score (variety of information types)
            active_types = len(
                set(
                    [
                        t.information_type
                        for t in self.digital_traces.values()
                        if not t.is_expired()
                    ]
                )
            )
            max_types = len(self.information_types)
            diversity_score = active_types / max_types if max_types > 0 else 0

            # Freshness score (recent activity)
            recent_threshold = datetime.now() - timedelta(hours=1)
            recent_traces = len(
                [
                    t
                    for t in self.digital_traces.values()
                    if t.timestamp > recent_threshold
                ]
            )
            freshness_score = min(1.0, recent_traces / max(10, total_traces * 0.1))

            # Combine metrics
            quality_score = float(
                avg_credibility * 0.4 + diversity_score * 0.3 + freshness_score * 0.3
            )

            self.performance_stats["information_quality_score"] = quality_score

            # Clean up expired traces
            await self._cleanup_expired_traces()

            return quality_score

        except Exception as e:
            logger.error(f"Failed to update information quality: {e}")
            return 0.5

    async def _cleanup_expired_traces(self) -> None:
        """Remove expired traces from the system."""
        expired_traces = []

        for trace_id, trace in self.digital_traces.items():
            if trace.is_expired():
                expired_traces.append(trace_id)

        for trace_id in expired_traces:
            self._remove_trace(trace_id)

        if expired_traces:
            logger.debug(f"Cleaned up {len(expired_traces)} expired traces")

    def get_system_statistics(self) -> dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "total_traces": len(self.digital_traces),
            "active_traces": len(
                [t for t in self.digital_traces.values() if not t.is_expired()]
            ),
            "total_queries": len(self.query_history),
            "information_types": len(self.information_types),
            "performance_stats": self.performance_stats.copy(),
            "trace_index_sizes": {k: len(v) for k, v in self.trace_index.items()},
            "agent_participation": len(self.agent_traces),
            "spatial_coverage": len(self.spatial_index) if self.spatial_indexer else 0,
        }

        # Information type distribution
        type_distribution: defaultdict[str, int] = defaultdict(int)
        for trace in self.digital_traces.values():
            if not trace.is_expired():
                type_distribution[trace.information_type] += 1

        stats["information_type_distribution"] = dict(type_distribution)

        # Query pattern analysis
        if self.query_history:
            query_types: defaultdict[str, int] = defaultdict(int)
            for query in self.query_history:
                query_types[query.query_type] += 1

            stats["query_patterns"] = dict(query_types)

        return stats

    def save_digital_traces(self, filepath: str) -> bool:
        """Save digital traces to file."""
        try:
            import json

            data = {
                "communication_medium": self.communication_medium,
                "information_types": self.information_types,
                "persistence_model": self.persistence_model,
                "access_control": self.access_control,
                "traces": [trace.to_dict() for trace in self.digital_traces.values()],
                "query_history": [query.to_dict() for query in self.query_history],
                "access_patterns": dict(self.access_patterns),
                "performance_stats": self.performance_stats,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Digital traces saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save digital traces: {e}")
            return False

    def load_digital_traces(self, filepath: str) -> bool:
        """Load digital traces from file."""
        try:
            import json

            with open(filepath) as f:
                data = json.load(f)

            # Restore traces
            self.digital_traces.clear()
            for trace_data in data.get("traces", []):
                trace = DigitalTrace(
                    trace_id=trace_data["trace_id"],
                    agent_id=trace_data["agent_id"],
                    information_type=trace_data["information_type"],
                    content=trace_data["content"],
                    location=(
                        np.array(trace_data["location"])
                        if trace_data["location"]
                        else None
                    ),
                    timestamp=datetime.fromisoformat(trace_data["timestamp"]),
                    visibility_scope=trace_data["visibility_scope"],
                    persistence_duration=trace_data["persistence_duration"],
                    credibility_score=trace_data["credibility_score"],
                    access_count=trace_data["access_count"],
                    metadata=trace_data["metadata"],
                )
                self.digital_traces[trace.trace_id] = trace

            # Restore indices
            self._rebuild_indices()

            # Restore query history
            self.query_history = [
                InformationQuery(
                    query_id=q["query_id"],
                    agent_id=q["agent_id"],
                    query_type=q["query_type"],
                    spatial_bounds=q["spatial_bounds"],
                    temporal_window=q["temporal_window"],
                    information_types=q["information_types"],
                    credibility_threshold=q["credibility_threshold"],
                    max_results=q["max_results"],
                    timestamp=datetime.fromisoformat(q["timestamp"]),
                )
                for q in data.get("query_history", [])
            ]

            # Restore access patterns
            self.access_patterns = defaultdict(int, data.get("access_patterns", {}))

            # Restore performance stats
            self.performance_stats = data.get(
                "performance_stats", self.performance_stats
            )

            logger.info(f"Digital traces loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load digital traces: {e}")
            return False

    def _rebuild_indices(self) -> None:
        """Rebuild all indices after loading traces."""
        # Clear existing indices
        self.trace_index.clear()
        self.spatial_index.clear()
        self.agent_traces.clear()

        # Rebuild indices
        for trace_id, trace in self.digital_traces.items():
            self.trace_index[trace.information_type].append(trace_id)
            self.agent_traces[trace.agent_id].append(trace_id)

            if trace.location is not None and self.spatial_indexer:
                try:
                    cell_id = self.spatial_indexer.latlng_to_cell(
                        trace.location[0], trace.location[1], self.spatial_resolution
                    )
                    self.spatial_index[cell_id].append(trace_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to rebuild spatial index for trace {trace_id}: {e}"
                    )
