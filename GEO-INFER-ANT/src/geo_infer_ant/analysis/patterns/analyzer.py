"""
Emergent Behavior Pattern Analysis for GEO-INFER-ANT

This module provides comprehensive tools for analyzing emergent patterns in swarm
behavior, including spatial patterns, temporal dynamics, interaction networks,
and complex system metrics. It integrates advanced statistical and machine learning
techniques to identify and characterize collective intelligence phenomena.

Key Features:
- Spatial pattern recognition (clustering, flocking, migration)
- Temporal pattern analysis (synchronization, periodicity)
- Interaction network analysis (communication, influence)
- Information theory measures (mutual information, transfer entropy)
- Complexity measures (fractal dimension, Lyapunov exponents)
- Visualization and interpretation tools
"""

import numpy as np
import logging
from typing import Any, cast
from datetime import datetime
from collections import defaultdict

# Integration imports
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
    from geo_infer_space.core.statistics import SpatialStatistics
except ImportError as e:
    logging.getLogger(__name__).debug("Optional spatial integration unavailable: %s", e)
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None
    SpatialStatistics = None

from ._config import AnalysisConfiguration

logger = logging.getLogger(__name__)


class SwarmPatternAnalyzer:
    """
    Comprehensive pattern analyzer for swarm behavior and emergent phenomena.

    This class provides sophisticated analysis tools for understanding collective
    behavior in swarm intelligence systems, including pattern recognition,
    statistical analysis, information theory measures, and complexity analysis.

    Integration Points:
    - GEO-INFER-SPACE: Spatial analysis and indexing for pattern recognition
    - GEO-INFER-MATH: Statistical and mathematical analysis tools
    - GEO-INFER-TIME: Temporal pattern analysis and dynamics
    """

    def __init__(
        self,
        analysis_types: list[str] | None = None,
        statistical_methods: list[str] | None = None,
        visualization_tools: list[str] | None = None,
        spatial_backend: str = "h3",
        **kwargs: Any,
    ):
        """
        Initialize pattern analyzer.

        Args:
            analysis_types: Types of analysis to perform
            statistical_methods: Statistical methods to use
            visualization_tools: Visualization tools for results
            spatial_backend: Backend for spatial operations
            **kwargs: Additional configuration parameters
        """
        self.config = AnalysisConfiguration(
            analysis_types=analysis_types
            or ["spatial_patterns", "temporal_patterns", "interaction_networks"],
            statistical_methods=statistical_methods
            or ["cluster_analysis", "network_analysis", "information_theory"],
            visualization_tools=visualization_tools
            or ["trajectory_plots", "interaction_graphs", "phase_diagrams"],
        )

        # Analysis state
        self.analysis_history: list[dict[str, Any]] = []
        self.pattern_cache: dict[str, Any] = {}

        # Integration components
        self.spatial_indexer: Any | None = None
        self.spatial_analytics: Any | None = None
        self.spatial_statistics: Any | None = None

        # Initialize integrations
        self._initialize_integrations(spatial_backend)

        logger.info(
            f"SwarmPatternAnalyzer initialized with {len(self.config.analysis_types)} analysis types"
        )

    def _initialize_integrations(self, backend: str) -> None:
        """Initialize integration with other GEO-INFER modules."""
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

        if SpatialStatistics:
            try:
                self.spatial_statistics = SpatialStatistics()
                logger.info("Spatial statistics initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial statistics: {e}")

    def analyze_spatial_patterns(
        self,
        agent_trajectories: list[np.ndarray] | np.ndarray,
        pattern_types: list[str] | None = None,
        spatial_scale: float | None = None,
        temporal_window: float | None = None,
    ) -> dict[str, Any]:
        """
        Analyze spatial patterns in agent trajectories.

        Args:
            agent_trajectories: Agent position data over time
            pattern_types: Types of spatial patterns to detect
            spatial_scale: Scale for spatial analysis (meters)
            temporal_window: Time window for analysis (seconds)

        Returns:
            Analysis results for spatial patterns
        """
        logger.info(
            f"Analyzing spatial patterns in {len(agent_trajectories)} trajectories"
        )

        analysis_results: dict[str, Any] = {
            "analysis_type": "spatial_patterns",
            "analysis_time": datetime.now(),
            "patterns_detected": {},
            "statistical_measures": {},
            "interpretation": {},
        }

        try:
            # Process trajectory data
            if isinstance(agent_trajectories, list):
                trajectories = np.array(agent_trajectories)
            else:
                trajectories = agent_trajectories

            if len(trajectories) == 0:
                return analysis_results

            # Detect different types of spatial patterns
            pattern_types = pattern_types or [
                "clustering",
                "flocking",
                "swarming",
                "migration",
                "dispersion",
            ]

            for pattern_type in pattern_types:
                if pattern_type == "clustering":
                    pattern_result = self._analyze_clustering_patterns(trajectories)
                elif pattern_type == "flocking":
                    pattern_result = self._analyze_flocking_patterns(trajectories)
                elif pattern_type == "migration":
                    pattern_result = self._analyze_migration_patterns(trajectories)
                elif pattern_type == "dispersion":
                    pattern_result = self._analyze_dispersion_patterns(trajectories)
                else:
                    pattern_result = {"status": "insufficient_data"}

                analysis_results["patterns_detected"][pattern_type] = pattern_result

            # Calculate statistical measures
            statistical_measures = self._calculate_spatial_statistics(trajectories)
            analysis_results["statistical_measures"] = statistical_measures

            # Generate interpretation
            interpretation = self._interpret_spatial_patterns(analysis_results)
            analysis_results["interpretation"] = interpretation

        except Exception as e:
            logger.error(f"Spatial pattern analysis failed: {e}")
            analysis_results["error"] = str(e)

        # Cache results
        self.pattern_cache["spatial_patterns"] = analysis_results
        self.analysis_history.append(analysis_results)

        logger.info(
            f"Spatial pattern analysis completed: {len(analysis_results['patterns_detected'])} pattern types analyzed"
        )
        return analysis_results

    def _analyze_clustering_patterns(self, trajectories: np.ndarray) -> dict[str, Any]:
        """Analyze spatial clustering patterns in trajectories."""
        try:
            if self.spatial_analytics:
                # Use spatial analytics for clustering
                clusters = self.spatial_analytics.analyze_clusters(
                    data=trajectories.reshape(-1, trajectories.shape[-1]),
                    method="dbscan",
                    eps=0.01,  # 1km clustering radius
                    min_samples=3,
                )

                return {
                    "method": "spatial_analytics",
                    "clusters": clusters,
                    "n_clusters": len(clusters.get("clusters", [])),
                    "cluster_sizes": [
                        len(c.get("indices", [])) for c in clusters.get("clusters", [])
                    ],
                }
            else:
                return self._distance_clustering_analysis(trajectories)

        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
            return self._distance_clustering_analysis(trajectories)

    def _distance_clustering_analysis(self, trajectories: np.ndarray) -> dict[str, Any]:
        """Cluster trajectories using pairwise distance when indexing is unavailable."""
        # Simple distance-based clustering
        n_agents, n_steps, n_dims = trajectories.shape

        # Calculate average positions
        avg_positions = np.mean(trajectories, axis=1)

        # Calculate pairwise distances
        distances = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                distances[i, j] = np.linalg.norm(avg_positions[i] - avg_positions[j])

        # Find clusters based on distance threshold
        distance_threshold = 0.02  # degrees (roughly 2km)
        clusters = []

        visited = set()
        for i in range(n_agents):
            if i not in visited:
                cluster = [i]
                visited.add(i)

                for j in range(i + 1, n_agents):
                    if distances[i, j] < distance_threshold:
                        cluster.append(j)
                        visited.add(j)

                clusters.append(cluster)

        return {
            "method": "distance_based",
            "n_clusters": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "distance_threshold": distance_threshold,
        }

    def _analyze_flocking_patterns(self, trajectories: np.ndarray) -> dict[str, Any]:
        """Analyze flocking behavior patterns.

        Separation is a nearest-neighbour distance, so fewer than two agents
        is a degenerate input: the result is ``{"status":
        "flocking_analysis_failed", "error": ...}`` naming the agent count.
        """
        try:
            if trajectories.shape[0] < 2:
                raise ValueError(
                    "flocking analysis requires at least 2 agents; "
                    f"got {trajectories.shape[0]}"
                )
            # Calculate velocity vectors
            velocities = np.diff(trajectories, axis=1)  # Shape: (agents, steps-1, dims)

            # Calculate alignment (average velocity correlation)
            alignment_scores = []
            for step in range(velocities.shape[1]):
                step_velocities = velocities[:, step, :]
                if np.any(step_velocities):
                    # Average velocity direction
                    avg_velocity = np.mean(step_velocities, axis=0)
                    if np.linalg.norm(avg_velocity) > 0:
                        # Correlation with average direction
                        correlations = []
                        for agent_vel in step_velocities:
                            if np.linalg.norm(agent_vel) > 0:
                                correlation = np.dot(agent_vel, avg_velocity) / (
                                    np.linalg.norm(agent_vel)
                                    * np.linalg.norm(avg_velocity)
                                )
                                correlations.append(correlation)

                        alignment_scores.append(np.mean(correlations))

            # Calculate cohesion (distance to center)
            cohesion_scores = []
            for step in range(trajectories.shape[1]):
                positions = trajectories[:, step, :]
                center = np.mean(positions, axis=0)
                distances_to_center = [
                    np.linalg.norm(pos - center) for pos in positions
                ]
                cohesion_scores.append(np.mean(distances_to_center))

            # Calculate separation (minimum distances between agents)
            separation_scores = []
            for step in range(trajectories.shape[1]):
                positions = trajectories[:, step, :]
                pair_dists = np.linalg.norm(
                    positions[:, None, :] - positions[None, :, :], axis=-1
                )
                np.fill_diagonal(pair_dists, np.inf)
                separation_scores.append(np.mean(pair_dists.min(axis=1)))

            return {
                "flocking_measures": {
                    "alignment": np.mean(alignment_scores) if alignment_scores else 0.0,
                    "cohesion": np.mean(cohesion_scores) if cohesion_scores else 0.0,
                    "separation": (
                        np.mean(separation_scores) if separation_scores else 0.0
                    ),
                },
                "flocking_detected": self._detect_flocking_behavior(
                    alignment_scores, cohesion_scores, separation_scores
                ),
            }

        except Exception as e:
            logger.warning(f"Flocking analysis failed: {e}")
            return {"status": "flocking_analysis_failed", "error": str(e)}

    def _detect_flocking_behavior(
        self, alignment: list[float], cohesion: list[float], separation: list[float]
    ) -> bool:
        """Detect if flocking behavior is present."""
        if not alignment or not cohesion or not separation:
            return False

        # Simple heuristic: balanced alignment, cohesion, and separation
        avg_alignment = float(np.mean(alignment))
        avg_cohesion = float(np.mean(cohesion))
        avg_separation = float(np.mean(separation))

        # Flocking criteria (tuned thresholds)
        alignment_ok = avg_alignment > 0.3
        cohesion_ok = 0.001 < avg_cohesion < 0.1  # Not too dispersed, not too clustered
        separation_ok = avg_separation > 0.005  # Maintain some separation

        return bool(alignment_ok and cohesion_ok and separation_ok)

    def _analyze_migration_patterns(self, trajectories: np.ndarray) -> dict[str, Any]:
        """Analyze migration and movement patterns."""
        try:
            # Calculate displacement over time
            displacements = []
            for agent in range(trajectories.shape[0]):
                start_pos = trajectories[agent, 0, :]
                end_pos = trajectories[agent, -1, :]
                displacement = np.linalg.norm(end_pos - start_pos)
                displacements.append(displacement)

            avg_displacement = np.mean(displacements)

            # Calculate movement directionality
            directionality_scores = []
            for agent in range(trajectories.shape[0]):
                positions = trajectories[agent, :, :]
                if len(positions) > 1:
                    # Calculate net displacement vector
                    displacement_vector = positions[-1] - positions[0]
                    total_distance = np.sum(
                        [
                            np.linalg.norm(positions[i + 1] - positions[i])
                            for i in range(len(positions) - 1)
                        ]
                    )

                    if total_distance > 0:
                        directionality = (
                            np.linalg.norm(displacement_vector) / total_distance
                        )
                        directionality_scores.append(directionality)

            avg_directionality = (
                np.mean(directionality_scores) if directionality_scores else 0.0
            )

            # Detect migration vs random movement
            migration_detected = bool(
                avg_directionality > 0.3 and avg_displacement > 0.01
            )

            return {
                "migration_measures": {
                    "avg_displacement": avg_displacement,
                    "avg_directionality": avg_directionality,
                    "max_displacement": np.max(displacements),
                    "displacement_variance": np.var(displacements),
                },
                "migration_detected": migration_detected,
                "movement_type": (
                    "migration" if migration_detected else "random_movement"
                ),
            }

        except Exception as e:
            logger.warning(f"Migration analysis failed: {e}")
            return {"status": "migration_analysis_failed", "error": str(e)}

    def _analyze_dispersion_patterns(self, trajectories: np.ndarray) -> dict[str, Any]:
        """Analyze dispersion and spreading patterns."""
        try:
            # Calculate spatial spread over time
            spread_measures = []

            for step in range(trajectories.shape[1]):
                positions = trajectories[:, step, :]
                if len(positions) > 1:
                    # Calculate bounding box area
                    min_coords = np.min(positions, axis=0)
                    max_coords = np.max(positions, axis=0)
                    bounding_area = np.prod(max_coords - min_coords)

                    # Calculate standard deviation of positions
                    position_std = np.std(positions, axis=0)

                    spread_measures.append(
                        {
                            "bounding_area": bounding_area,
                            "position_std": position_std,
                            "spatial_extent": np.max(max_coords - min_coords),
                        }
                    )

            # Analyze dispersion trend
            if len(spread_measures) > 1:
                initial_spread = spread_measures[0]["bounding_area"]
                final_spread = spread_measures[-1]["bounding_area"]

                if final_spread > initial_spread * 2:
                    dispersion_trend = "increasing"
                elif final_spread < initial_spread * 0.5:
                    dispersion_trend = "decreasing"
                else:
                    dispersion_trend = "stable"

                dispersion_rate = (final_spread - initial_spread) / len(spread_measures)
            else:
                dispersion_trend = "insufficient_data"
                dispersion_rate = 0.0

            return {
                "dispersion_measures": {
                    "initial_spread": (
                        spread_measures[0]["bounding_area"] if spread_measures else 0.0
                    ),
                    "final_spread": (
                        spread_measures[-1]["bounding_area"] if spread_measures else 0.0
                    ),
                    "max_spread": (
                        max([s["bounding_area"] for s in spread_measures])
                        if spread_measures
                        else 0.0
                    ),
                    "dispersion_trend": dispersion_trend,
                    "dispersion_rate": dispersion_rate,
                },
                "dispersion_detected": dispersion_trend == "increasing",
            }

        except Exception as e:
            logger.warning(f"Dispersion analysis failed: {e}")
            return {"status": "dispersion_analysis_failed", "error": str(e)}

    def _calculate_spatial_statistics(self, trajectories: np.ndarray) -> dict[str, Any]:
        """Calculate comprehensive spatial statistics."""
        stats: dict[str, Any] = {}

        try:
            # Basic spatial statistics
            all_positions = trajectories.reshape(-1, trajectories.shape[-1])
            stats["overall"] = {
                "total_positions": len(all_positions),
                "spatial_bounds": {
                    "min_lat": np.min(all_positions[:, 0]),
                    "max_lat": np.max(all_positions[:, 0]),
                    "min_lng": np.min(all_positions[:, 1]),
                    "max_lng": np.max(all_positions[:, 1]),
                },
                "center_of_mass": np.mean(all_positions, axis=0).tolist(),
                "spatial_extent": np.max(all_positions, axis=0)
                - np.min(all_positions, axis=0),
            }

            # Agent-specific statistics
            stats["per_agent"] = []
            for agent in range(trajectories.shape[0]):
                agent_positions = trajectories[agent, :, :]
                agent_stats = {
                    "agent_id": agent,
                    "total_distance": np.sum(
                        [
                            np.linalg.norm(agent_positions[i + 1] - agent_positions[i])
                            for i in range(len(agent_positions) - 1)
                        ]
                    ),
                    "net_displacement": np.linalg.norm(
                        agent_positions[-1] - agent_positions[0]
                    ),
                    "activity_radius": np.max(
                        [
                            np.linalg.norm(pos - agent_positions[0])
                            for pos in agent_positions
                        ]
                    ),
                    "position_variance": np.var(agent_positions, axis=0).tolist(),
                }
                stats["per_agent"].append(agent_stats)

            # Temporal evolution statistics
            stats["temporal_evolution"] = []
            for step in range(trajectories.shape[1]):
                step_positions = trajectories[:, step, :]
                step_stats = {
                    "step": step,
                    "center_of_mass": np.mean(step_positions, axis=0).tolist(),
                    "spatial_spread": np.std(step_positions, axis=0).tolist(),
                    "bounding_area": np.prod(
                        np.max(step_positions, axis=0) - np.min(step_positions, axis=0)
                    ),
                }
                stats["temporal_evolution"].append(step_stats)

        except Exception as e:
            logger.warning(f"Spatial statistics calculation failed: {e}")
            stats["error"] = str(e)

        return stats

    def _interpret_spatial_patterns(
        self, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate interpretation of spatial patterns."""
        interpretation = {
            "pattern_summary": "",
            "behavior_classification": "",
            "key_insights": [],
            "recommendations": [],
        }

        try:
            patterns = analysis_results.get("patterns_detected", {})

            # Generate pattern summary
            detected_patterns = [
                p
                for p, result in patterns.items()
                if result.get("status") != "pattern_type_not_implemented"
            ]

            if detected_patterns:
                interpretation["pattern_summary"] = (
                    f"Detected {len(detected_patterns)} spatial pattern types: {', '.join(detected_patterns)}"
                )
            else:
                interpretation["pattern_summary"] = "No clear spatial patterns detected"

            # Classify overall behavior
            behavior_scores = {}

            if "clustering" in patterns:
                clustering_result = patterns["clustering"]
                if clustering_result.get("n_clusters", 0) > 1:
                    behavior_scores["clustering"] = (
                        clustering_result.get("n_clusters", 0) / 10
                    )  # Normalize

            if "flocking" in patterns:
                flocking_result = patterns["flocking"]
                if flocking_result.get("flocking_detected", False):
                    flocking_measures = flocking_result.get("flocking_measures", {})
                    behavior_scores["flocking"] = np.mean(
                        list(flocking_measures.values())
                    )

            if "migration" in patterns:
                migration_result = patterns["migration"]
                if migration_result.get("migration_detected", False):
                    behavior_scores["migration"] = migration_result.get(
                        "migration_measures", {}
                    ).get("avg_directionality", 0)

            if behavior_scores:
                dominant_behavior = max(
                    behavior_scores, key=lambda k: behavior_scores[k]
                )
                interpretation["behavior_classification"] = (
                    f"Dominant behavior: {dominant_behavior}"
                )
                interpretation["behavior_confidence"] = behavior_scores[
                    dominant_behavior
                ]

            # Generate insights
            insights = []

            for pattern_name, pattern_result in patterns.items():
                if pattern_result.get("status") == "success":
                    if pattern_name == "clustering":
                        n_clusters = pattern_result.get("n_clusters", 0)
                        if n_clusters > 1:
                            insights.append(
                                f"Population shows {n_clusters} distinct spatial clusters"
                            )
                        elif n_clusters == 1:
                            insights.append(
                                "Population is spatially cohesive with minimal clustering"
                            )

                    elif pattern_name == "flocking":
                        if pattern_result.get("flocking_detected", False):
                            measures = pattern_result.get("flocking_measures", {})
                            insights.append(
                                f"Flocking behavior detected with alignment score: {measures.get('alignment', 0):.3f}"
                            )

                    elif pattern_name == "migration":
                        if pattern_result.get("migration_detected", False):
                            displacement = pattern_result.get(
                                "migration_measures", {}
                            ).get("avg_displacement", 0)
                            insights.append(
                                f"Migration pattern detected with average displacement: {displacement:.4f} degrees"
                            )

            interpretation["key_insights"] = insights

        except Exception as e:
            logger.warning(f"Pattern interpretation failed: {e}")
            interpretation["error"] = str(e)

        return interpretation

    def analyze_interactions(
        self,
        communication_data: list[dict[str, Any]] | None = None,
        proximity_data: np.ndarray | None = None,
        influence_measures: dict[str, Any] | None = None,
        network_metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze interaction networks and social dynamics.

        Args:
            communication_data: Records of agent communications
            proximity_data: Matrix of agent proximity over time
            influence_measures: Measures of agent influence
            network_metrics: Network analysis metrics to compute

        Returns:
            Interaction network analysis results
        """
        logger.info("Analyzing interaction networks")

        analysis_results = {
            "analysis_type": "interaction_networks",
            "analysis_time": datetime.now(),
            "network_structure": {},
            "influence_analysis": {},
            "communication_patterns": {},
            "social_dynamics": {},
        }

        try:
            # Analyze communication patterns
            if communication_data:
                comm_analysis = self._analyze_communication_patterns(communication_data)
                analysis_results["communication_patterns"] = comm_analysis

            # Analyze proximity networks
            if proximity_data is not None:
                proximity_analysis = self._analyze_proximity_networks(proximity_data)
                analysis_results["network_structure"] = proximity_analysis

            # Analyze influence and leadership
            if influence_measures:
                influence_analysis = self._analyze_influence_dynamics(
                    influence_measures
                )
                analysis_results["influence_analysis"] = influence_analysis

            # Compute network metrics
            network_metrics = network_metrics or [
                "centrality",
                "clustering",
                "modularity",
                "density",
            ]
            metrics_analysis = self._compute_network_metrics(
                analysis_results, network_metrics
            )
            analysis_results["network_metrics"] = metrics_analysis

            # Analyze social dynamics
            social_dynamics = self._analyze_social_dynamics(analysis_results)
            analysis_results["social_dynamics"] = social_dynamics

        except Exception as e:
            logger.error(f"Interaction analysis failed: {e}")
            analysis_results["error"] = str(e)

        # Cache and record analysis
        self.pattern_cache["interaction_networks"] = analysis_results
        self.analysis_history.append(analysis_results)

        logger.info("Interaction network analysis completed")
        return analysis_results

    def _analyze_communication_patterns(
        self, communication_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze patterns in agent communication."""
        patterns: dict[str, Any] = {
            "communication_frequency": defaultdict(int),
            "communication_types": defaultdict(int),
            "temporal_patterns": defaultdict(list),
            "spatial_patterns": defaultdict(list),
        }

        try:
            for comm in communication_data:
                sender = comm.get("from", "unknown")
                receiver = comm.get("to", "unknown")
                comm_type = comm.get("type", "unknown")
                timestamp = comm.get("timestamp", datetime.now())
                location = comm.get("location", None)

                # Communication frequency
                patterns["communication_frequency"][f"{sender}->{receiver}"] += 1
                patterns["communication_types"][comm_type] += 1

                # Temporal patterns
                hour = timestamp.hour if hasattr(timestamp, "hour") else 0
                patterns["temporal_patterns"][hour].append(comm)

                # Spatial patterns
                if location is not None:
                    patterns["spatial_patterns"][tuple(location)].append(comm)

            # Calculate communication statistics
            total_communications = len(communication_data)
            unique_pairs = len(patterns["communication_frequency"])
            communication_efficiency = (
                unique_pairs / total_communications if total_communications > 0 else 0
            )

            patterns["statistics"] = {
                "total_communications": total_communications,
                "unique_communication_pairs": unique_pairs,
                "communication_efficiency": communication_efficiency,
                "most_common_type": (
                    max(patterns["communication_types"].items(), key=lambda x: x[1])[0]
                    if patterns["communication_types"]
                    else None
                ),
            }

        except Exception as e:
            logger.warning(f"Communication pattern analysis failed: {e}")
            patterns["error"] = str(e)

        return patterns

    def _analyze_proximity_networks(self, proximity_data: np.ndarray) -> dict[str, Any]:
        """Analyze networks based on agent proximity."""
        try:
            # proximity_data shape: (time_steps, agents, agents) or (agents, agents)
            if proximity_data.ndim == 3:
                # Average over time
                avg_proximity = np.mean(proximity_data, axis=0)
            else:
                avg_proximity = proximity_data

            if (
                avg_proximity.ndim != 2
                or avg_proximity.shape[0] != avg_proximity.shape[1]
            ):
                raise ValueError("proximity_data must produce a square matrix")
            n_agents = avg_proximity.shape[0]
            if n_agents < 2:
                return {
                    "adjacency_matrix": avg_proximity.astype(int).tolist(),
                    "network_properties": {
                        "n_nodes": n_agents,
                        "n_edges": 0,
                        "density": 0.0,
                    },
                    "degree_distribution": [0] * n_agents,
                    "proximity_threshold": 0.0,
                }
            positive_values = avg_proximity[avg_proximity > 0]
            threshold = (
                float(np.median(positive_values)) if positive_values.size else 0.0
            )
            adjacency_matrix = (avg_proximity < threshold).astype(int)
            np.fill_diagonal(adjacency_matrix, 0)

            # Calculate network properties
            # Node degrees
            degrees = np.sum(adjacency_matrix, axis=1)

            # Network density
            density = np.sum(adjacency_matrix) / (n_agents * (n_agents - 1))

            import networkx as nx

            graph = nx.from_numpy_array(adjacency_matrix)
            clustering = float(nx.average_clustering(graph))

            return {
                "adjacency_matrix": adjacency_matrix.tolist(),
                "network_properties": {
                    "n_nodes": n_agents,
                    "n_edges": int(graph.number_of_edges()),
                    "density": density,
                    "avg_degree": np.mean(degrees),
                    "clustering_coefficient": clustering,
                    "max_degree": np.max(degrees),
                    "min_degree": np.min(degrees),
                },
                "degree_distribution": degrees.tolist(),
                "proximity_threshold": threshold,
            }

        except Exception as e:
            logger.warning(f"Proximity network analysis failed: {e}")
            return {"status": "proximity_analysis_failed", "error": str(e)}

    def _analyze_influence_dynamics(
        self, influence_measures: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze influence and leadership dynamics."""
        try:
            analysis: dict[str, Any] = {
                "influence_ranking": [],
                "leadership_structure": {},
                "influence_network": {},
            }

            # Rank agents by influence
            if "agent_influence" in influence_measures:
                agent_influence = influence_measures["agent_influence"]
                sorted_influence = sorted(
                    agent_influence.items(), key=lambda x: x[1], reverse=True
                )
                analysis["influence_ranking"] = sorted_influence

                # Identify leaders (top 10%)
                n_leaders = max(1, len(sorted_influence) // 10)
                analysis["leadership_structure"] = {
                    "leaders": [agent for agent, _ in sorted_influence[:n_leaders]],
                    "followers": [agent for agent, _ in sorted_influence[n_leaders:]],
                    "influence_gap": (
                        sorted_influence[0][1] - sorted_influence[n_leaders][1]
                        if n_leaders < len(sorted_influence)
                        else 0
                    ),
                }

            return analysis

        except Exception as e:
            logger.warning(f"Influence analysis failed: {e}")
            return {"status": "influence_analysis_failed", "error": str(e)}

    def _compute_network_metrics(
        self, analysis_results: dict[str, Any], metrics: list[str]
    ) -> dict[str, Any]:
        """Compute specified network metrics."""
        metrics_results: dict[str, Any] = {}

        try:
            network_structure = analysis_results.get("network_structure", {})

            for metric in metrics:
                if metric == "centrality":
                    centrality_results = self._compute_centrality_measures(
                        network_structure
                    )
                    metrics_results["centrality"] = centrality_results

                elif metric == "clustering":
                    clustering_results = self._compute_clustering_measures(
                        network_structure
                    )
                    metrics_results["clustering"] = clustering_results

                elif metric == "modularity":
                    modularity_results = self._compute_modularity(network_structure)
                    metrics_results["modularity"] = modularity_results

                elif metric == "density":
                    density = network_structure.get("network_properties", {}).get(
                        "density", 0.0
                    )
                    metrics_results["density"] = density

        except Exception as e:
            logger.warning(f"Network metrics computation failed: {e}")
            metrics_results["error"] = str(e)

        return metrics_results

    def _compute_centrality_measures(
        self, network_structure: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute centrality measures for the network."""
        try:
            adjacency = np.array(network_structure.get("adjacency_matrix", []))

            if len(adjacency) == 0:
                return {"status": "no_adjacency_matrix"}

            if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
                raise ValueError("adjacency_matrix must be square")
            import networkx as nx

            graph = nx.from_numpy_array(adjacency)
            degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
            betweenness_centrality = np.array(
                list(nx.betweenness_centrality(graph, normalized=True).values())
            )
            closeness_centrality = np.array(
                list(nx.closeness_centrality(graph).values())
            )

            return {
                "degree_centrality": degree_centrality.tolist(),
                "betweenness_centrality": betweenness_centrality.tolist(),
                "closeness_centrality": closeness_centrality.tolist(),
                "max_degree_centrality": np.max(degree_centrality),
                "centralization_index": np.max(degree_centrality)
                - np.mean(degree_centrality),
            }

        except Exception as e:
            logger.warning(f"Centrality computation failed: {e}")
            return {"status": "centrality_computation_failed", "error": str(e)}

    def _compute_clustering_measures(
        self, network_structure: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute clustering measures for the network."""
        adjacency = np.asarray(
            network_structure.get("adjacency_matrix", []), dtype=float
        )
        if adjacency.size == 0:
            return {"status": "no_network"}
        import networkx as nx

        graph = nx.from_numpy_array(adjacency)
        coefficients = nx.clustering(graph)
        clustering_coeff = (
            float(np.mean(list(coefficients.values()))) if coefficients else 0.0
        )

        return {
            "global_clustering_coefficient": clustering_coeff,
            "node_clustering": coefficients,
            "method": "networkx.average_clustering",
        }

    def _compute_modularity(self, network_structure: dict[str, Any]) -> dict[str, Any]:
        """Compute network modularity."""
        adjacency = np.asarray(
            network_structure.get("adjacency_matrix", []), dtype=float
        )
        n_nodes = adjacency.shape[0]

        if n_nodes == 0:
            return {"status": "no_network"}

        import networkx as nx

        graph = nx.from_numpy_array(adjacency)
        if graph.number_of_edges() == 0:
            return {"modularity": 0.0, "communities": 0, "community_sizes": []}
        from networkx.algorithms import community  # type: ignore[import-untyped]

        try:
            communities = list(community.louvain_communities(graph, seed=0))
            method = "louvain"
        except (AttributeError, ImportError):
            communities = list(community.greedy_modularity_communities(graph))
            method = "greedy_modularity"
        modularity = float(community.modularity(graph, communities))

        return {
            "modularity": modularity,
            "communities": len(communities),
            "community_sizes": [len(nodes) for nodes in communities],
            "method": method,
        }

    def _analyze_social_dynamics(
        self, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze social dynamics and group behavior."""
        dynamics: dict[str, Any] = {
            "group_formation": {},
            "leadership_emergence": {},
            "information_flow": {},
            "coordination_quality": 0.0,
        }

        try:
            # Analyze group formation based on network structure
            network_structure = analysis_results.get("network_structure", {})
            network_props = network_structure.get("network_properties", {})

            if network_props:
                density = network_props.get("density", 0.0)
                clustering = network_props.get("clustering_coefficient", 0.0)
                avg_degree = network_props.get("avg_degree", 0.0)

                dynamics["group_formation"] = {
                    "network_density": density,
                    "clustering_coefficient": clustering,
                    "average_connections": avg_degree,
                    "group_formation_strength": min(
                        1.0, density * clustering * avg_degree
                    ),
                }

            # Analyze leadership emergence
            influence_analysis = analysis_results.get("influence_analysis", {})
            leadership = influence_analysis.get("leadership_structure", {})

            if leadership:
                n_leaders = len(leadership.get("leaders", []))
                influence_gap = leadership.get("influence_gap", 0.0)

                dynamics["leadership_emergence"] = {
                    "number_of_leaders": n_leaders,
                    "influence_gap": influence_gap,
                    "leadership_clarity": min(1.0, influence_gap),
                }

            # Calculate overall coordination quality
            coordination_factors = []
            if "group_formation" in dynamics:
                coordination_factors.append(
                    dynamics["group_formation"].get("group_formation_strength", 0.0)
                )
            if "leadership_emergence" in dynamics:
                coordination_factors.append(
                    dynamics["leadership_emergence"].get("leadership_clarity", 0.0)
                )

            dynamics["coordination_quality"] = (
                np.mean(coordination_factors) if coordination_factors else 0.0
            )

        except Exception as e:
            logger.warning(f"Social dynamics analysis failed: {e}")
            dynamics["error"] = str(e)

        return dynamics

    def detect_emergence(
        self,
        individual_behaviors: list[dict[str, Any]],
        collective_outcomes: dict[str, Any],
        information_measures: list[str] | None = None,
        complexity_measures: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Detect emergent phenomena in swarm behavior.

        Args:
            individual_behaviors: Individual agent behavior records
            collective_outcomes: Collective system behavior outcomes
            information_measures: Information theory measures to compute
            complexity_measures: Complexity measures to compute

        Returns:
            Emergent phenomenon detection results
        """
        logger.info("Detecting emergent phenomena")

        emergence_results: dict[str, Any] = {
            "analysis_type": "emergent_phenomena",
            "analysis_time": datetime.now(),
            "emergence_detected": False,
            "emergence_measures": {},
            "information_theory": {},
            "complexity_analysis": {},
            "emergence_interpretation": {},
        }

        try:
            # Calculate information theory measures
            information_measures = information_measures or [
                "mutual_information",
                "transfer_entropy",
            ]
            for measure in information_measures:
                if measure == "mutual_information":
                    mi_result = self._calculate_mutual_information(
                        individual_behaviors, collective_outcomes
                    )
                    emergence_results["information_theory"]["mutual_information"] = (
                        mi_result
                    )

                elif measure == "transfer_entropy":
                    te_result = self._calculate_transfer_entropy(
                        individual_behaviors, collective_outcomes
                    )
                    emergence_results["information_theory"]["transfer_entropy"] = (
                        te_result
                    )

            # Calculate complexity measures
            complexity_measures = complexity_measures or [
                "fractal_dimension",
                "lyapunov_exponents",
            ]
            for measure in complexity_measures:
                if measure == "fractal_dimension":
                    fd_result = self._calculate_fractal_dimension(individual_behaviors)
                    emergence_results["complexity_analysis"]["fractal_dimension"] = (
                        fd_result
                    )

                elif measure == "lyapunov_exponents":
                    le_result = self._calculate_lyapunov_exponents(individual_behaviors)
                    emergence_results["complexity_analysis"]["lyapunov_exponents"] = (
                        le_result
                    )

            # Detect emergence based on measures
            emergence_detected = self._assess_emergence(emergence_results)
            emergence_results["emergence_detected"] = emergence_detected

            # Generate interpretation
            interpretation = self._interpret_emergence(emergence_results)
            emergence_results["emergence_interpretation"] = interpretation

        except Exception as e:
            logger.error(f"Emergence detection failed: {e}")
            emergence_results["error"] = str(e)

        # Cache and record analysis
        self.pattern_cache["emergent_phenomena"] = emergence_results
        self.analysis_history.append(emergence_results)

        logger.info(
            f"Emergence detection completed: emergence {'detected' if emergence_results['emergence_detected'] else 'not detected'}"
        )
        return emergence_results

    def _calculate_mutual_information(
        self,
        individual_behaviors: list[dict[str, Any]],
        collective_outcomes: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate mutual information between individual and collective behaviors."""
        try:
            # Extract behavior patterns as numerical values
            individual_values: list[float] = []
            for behavior in individual_behaviors:
                # Extract numerical features from behavior
                if "action_type" in behavior:
                    # Convert action type to numerical value
                    action_type = behavior["action_type"]
                    action_map = {
                        "forage": 0,
                        "explore": 1,
                        "communicate": 2,
                        "rest": 3,
                        "defend": 4,
                    }
                    individual_values.append(action_map.get(action_type, 0))
                elif "value" in behavior:
                    individual_values.append(float(behavior["value"]))
                else:
                    individual_values.append(0.0)

            # Extract collective outcome values
            collective_values = []
            for outcome_data in collective_outcomes.values():
                if isinstance(outcome_data, dict):
                    # Extract numerical value from outcome
                    value = outcome_data.get("value", outcome_data.get("score", 0.0))
                    collective_values.append(float(cast(Any, value)))
                elif isinstance(outcome_data, (int, float)):
                    collective_values.append(float(outcome_data))
                else:
                    collective_values.append(0.0)

            if len(individual_values) < 2 or len(collective_values) < 2:
                return {
                    "mutual_information_score": 0.0,
                    "interpretation": "insufficient_data",
                }

            # Align lengths (take minimum)
            min_len = min(len(individual_values), len(collective_values))
            individual_array = np.array(individual_values[:min_len])
            collective_array = np.array(collective_values[:min_len])

            # Discretize for mutual information calculation
            # Use quantile-based binning
            n_bins = min(10, int(np.sqrt(len(individual_array))))
            if n_bins < 2:
                n_bins = 2

            individual_binned = np.digitize(
                individual_array,
                np.linspace(np.min(individual_array), np.max(individual_array), n_bins),
            )
            collective_binned = np.digitize(
                collective_array,
                np.linspace(np.min(collective_array), np.max(collective_array), n_bins),
            )

            # Calculate mutual information using scikit-learn
            try:
                from sklearn.metrics import mutual_info_score

                mi_score = mutual_info_score(individual_binned, collective_binned)
                # Normalize by minimum entropy (normalized mutual information)
                entropy_individual = self._calculate_entropy(individual_binned)
                entropy_collective = self._calculate_entropy(collective_binned)
                min_entropy = min(entropy_individual, entropy_collective)
                normalized_mi = mi_score / min_entropy if min_entropy > 0 else 0.0
                mi_score = min(1.0, normalized_mi)  # Normalize to [0, 1]
            except ImportError:
                # Fallback: calculate MI manually
                mi_score = self._calculate_mutual_information_manual(
                    individual_binned, collective_binned
                )

            return {
                "mutual_information_score": float(mi_score),
                "interpretation": (
                    "high" if mi_score > 0.7 else "medium" if mi_score > 0.3 else "low"
                ),
                "entropy_individual": (
                    float(entropy_individual)
                    if "entropy_individual" in locals()
                    else 0.0
                ),
                "entropy_collective": (
                    float(entropy_collective)
                    if "entropy_collective" in locals()
                    else 0.0
                ),
            }

        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
            return {"status": "calculation_failed", "error": str(e)}

    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate Shannon entropy of discrete values."""
        if len(values) == 0:
            return 0.0

        # Count frequencies
        unique, counts = np.unique(values, return_counts=True)
        probabilities = counts / len(values)

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)

    def _calculate_mutual_information_manual(
        self, x: np.ndarray, y: np.ndarray
    ) -> float:
        """Calculate mutual information manually."""
        # Create joint distribution
        unique_x, counts_x = np.unique(x, return_counts=True)
        unique_y, counts_y = np.unique(y, return_counts=True)

        # Joint probability
        joint_counts = np.zeros((len(unique_x), len(unique_y)))
        for i, val_x in enumerate(unique_x):
            for j, val_y in enumerate(unique_y):
                joint_counts[i, j] = np.sum((x == val_x) & (y == val_y))

        joint_prob = joint_counts / len(x)
        prob_x = counts_x / len(x)
        prob_y = counts_y / len(y)

        # Calculate MI
        mi = 0.0
        for i in range(len(unique_x)):
            for j in range(len(unique_y)):
                if joint_prob[i, j] > 0 and prob_x[i] > 0 and prob_y[j] > 0:
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (prob_x[i] * prob_y[j])
                    )

        return max(0.0, mi)

    def _calculate_transfer_entropy(
        self,
        individual_behaviors: list[dict[str, Any]],
        collective_outcomes: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Calculate transfer entropy between behaviors and outcomes.

        Transfer entropy measures the directed information flow from individual behaviors
        to collective outcomes, accounting for the history of both processes.
        """
        try:
            # Extract time series from behaviors
            behavior_series = []
            for behavior in individual_behaviors:
                if "value" in behavior:
                    behavior_series.append(float(behavior["value"]))
                elif "action_type" in behavior:
                    action_map = {
                        "forage": 0,
                        "explore": 1,
                        "communicate": 2,
                        "rest": 3,
                        "defend": 4,
                    }
                    behavior_series.append(
                        float(action_map.get(behavior["action_type"], 0))
                    )
                else:
                    behavior_series.append(0.0)

            # Extract time series from outcomes
            outcome_series = []
            for outcome_data in collective_outcomes.values():
                if isinstance(outcome_data, dict):
                    outcome_series.append(
                        float(
                            cast(
                                Any,
                                outcome_data.get(
                                    "value", outcome_data.get("score", 0.0)
                                ),
                            )
                        )
                    )
                elif isinstance(outcome_data, (int, float)):
                    outcome_series.append(float(outcome_data))
                else:
                    outcome_series.append(0.0)

            if len(behavior_series) < 5 or len(outcome_series) < 5:
                return {
                    "transfer_entropy_score": 0.0,
                    "interpretation": "insufficient_data",
                }

            # Align series
            min_len = min(len(behavior_series), len(outcome_series))
            behavior_array = np.array(behavior_series[:min_len])
            outcome_array = np.array(outcome_series[:min_len])

            # Discretize for transfer entropy
            n_bins = min(5, int(np.sqrt(len(behavior_array))))
            if n_bins < 2:
                n_bins = 2

            behavior_binned = np.digitize(
                behavior_array,
                np.linspace(np.min(behavior_array), np.max(behavior_array), n_bins),
            )
            outcome_binned = np.digitize(
                outcome_array,
                np.linspace(np.min(outcome_array), np.max(outcome_array), n_bins),
            )

            # Calculate transfer entropy with history length k=1
            k = 1  # History length
            te_score = self._compute_transfer_entropy(
                behavior_binned, outcome_binned, k
            )

            # Normalize by outcome entropy
            outcome_entropy = self._calculate_entropy(outcome_binned)
            normalized_te = te_score / outcome_entropy if outcome_entropy > 0 else 0.0
            normalized_te = min(1.0, normalized_te)

            return {
                "transfer_entropy_score": float(normalized_te),
                "raw_transfer_entropy": float(te_score),
                "interpretation": (
                    "high"
                    if normalized_te > 0.5
                    else "medium"
                    if normalized_te > 0.2
                    else "low"
                ),
            }

        except Exception as e:
            logger.warning(f"Transfer entropy calculation failed: {e}")
            return {"status": "calculation_failed", "error": str(e)}

    def _compute_transfer_entropy(
        self, x: np.ndarray, y: np.ndarray, k: int = 1
    ) -> float:
        """
        Compute transfer entropy from X to Y.

        TE(X->Y) = H(Y_t | Y_{t-k}) - H(Y_t | Y_{t-k}, X_{t-k})
        """
        if len(x) < k + 1 or len(y) < k + 1:
            return 0.0

        # Create sequences with history
        y_future = y[k:]
        y_past = (
            y[:-k] if k == 1 else np.array([y[i : i + k] for i in range(len(y) - k)])
        )
        x_past = (
            x[:-k] if k == 1 else np.array([x[i : i + k] for i in range(len(x) - k)])
        )

        # Calculate conditional entropies
        # H(Y_t | Y_{t-k})
        h_y_given_y_past = self._conditional_entropy(y_future, y_past)

        # H(Y_t | Y_{t-k}, X_{t-k})
        # Combine past states
        if k == 1:
            combined_past = np.column_stack([y_past, x_past])
        else:
            combined_past = np.column_stack([y_past.flatten(), x_past.flatten()])

        h_y_given_yx_past = self._conditional_entropy(y_future, combined_past)

        # Transfer entropy
        te = h_y_given_y_past - h_y_given_yx_past
        return max(0.0, te)

    def _conditional_entropy(self, y: np.ndarray, x: np.ndarray) -> float:
        """Calculate conditional entropy H(Y|X)."""
        if len(y) != len(x):
            return 0.0

        # Create joint distribution
        if x.ndim == 1:
            x_reshaped = x.reshape(-1, 1)
        else:
            x_reshaped = x

        # Discretize for discrete entropy calculation
        # Use unique combinations
        unique_combinations: dict[Any, list[Any]] = {}
        for i in range(len(y)):
            x_key = tuple(x_reshaped[i]) if x_reshaped.ndim > 1 else (x_reshaped[i],)
            if x_key not in unique_combinations:
                unique_combinations[x_key] = []
            unique_combinations[x_key].append(y[i])

        # Calculate weighted conditional entropy
        conditional_entropy = 0.0
        for y_values in unique_combinations.values():
            p_x = len(y_values) / len(y)
            h_y_given_x = self._calculate_entropy(np.array(y_values))
            conditional_entropy += p_x * h_y_given_x

        return conditional_entropy

    def _calculate_fractal_dimension(
        self, individual_behaviors: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate fractal dimension of behavior patterns using box-counting method.

        Implements the box-counting algorithm for estimating fractal dimension
        of spatial patterns in swarm behavior.
        """
        try:
            # Extract spatial positions from behaviors
            positions = []
            for behavior in individual_behaviors:
                if "position" in behavior:
                    pos = behavior["position"]
                    if isinstance(pos, (list, np.ndarray)) and len(pos) >= 2:
                        positions.append([float(pos[0]), float(pos[1])])

            if len(positions) < 3:
                return {"status": "insufficient_data"}

            positions_array = np.array(positions)

            # Box-counting method for fractal dimension
            # Use multiple box sizes
            min_pos = np.min(positions_array, axis=0)
            max_pos = np.max(positions_array, axis=0)
            range_size = max(max_pos - min_pos)

            if range_size == 0:
                return {"fractal_dimension": 0.0, "interpretation": "degenerate"}

            # Box sizes (powers of 2)
            box_sizes: list[float] = []
            box_counts: list[int] = []

            max_box_size = range_size
            min_box_size = range_size / 100.0  # At least 100 boxes

            current_size = max_box_size
            while current_size >= min_box_size and len(box_sizes) < 10:
                # Count boxes containing points
                n_boxes_x = int(np.ceil((max_pos[0] - min_pos[0]) / current_size))
                n_boxes_y = int(np.ceil((max_pos[1] - min_pos[1]) / current_size))

                if n_boxes_x == 0 or n_boxes_y == 0:
                    break

                # Create grid and count occupied boxes
                occupied_boxes = set()
                for pos in positions_array:
                    box_x = int((pos[0] - min_pos[0]) / current_size)
                    box_y = int((pos[1] - min_pos[1]) / current_size)
                    occupied_boxes.add((box_x, box_y))

                box_sizes.append(current_size)
                box_counts.append(len(occupied_boxes))

                current_size /= 2.0

            if len(box_sizes) < 2:
                # Fallback: simple estimate
                fractal_dim = 1.0
            else:
                # Fit log-log relationship: log(N) = -D * log(ε) + C
                log_sizes = np.log(box_sizes)
                log_counts = np.log(box_counts)

                # Linear regression
                if len(log_sizes) > 1:
                    # Use numpy polyfit
                    coeffs = np.polyfit(log_sizes, log_counts, 1)
                    fractal_dim = -coeffs[0]  # Negative slope
                else:
                    fractal_dim = 1.0

            # Clamp to reasonable range [0, 2] for 2D patterns
            fractal_dim = max(0.0, min(2.0, fractal_dim))

            return {
                "fractal_dimension": float(fractal_dim),
                "interpretation": (
                    "complex"
                    if fractal_dim > 1.5
                    else "moderate"
                    if fractal_dim > 1.0
                    else "simple"
                ),
                "box_sizes": box_sizes,
                "box_counts": box_counts,
            }

        except Exception as e:
            logger.warning(f"Fractal dimension calculation failed: {e}")
            return {"status": "calculation_failed", "error": str(e)}

    def _calculate_lyapunov_exponents(
        self, individual_behaviors: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate Lyapunov exponents for chaos analysis.

        Estimates the maximum Lyapunov exponent from behavior time series data
        using delay embedding and nearest-neighbor divergence.
        """
        try:
            # Extract temporal behavior sequences as numerical values
            behavior_values = []
            timestamps = []

            for behavior in individual_behaviors:
                if "value" in behavior:
                    behavior_values.append(float(behavior["value"]))
                    timestamps.append(behavior.get("timestamp", datetime.now()))
                elif "action_type" in behavior:
                    action_map = {
                        "forage": 0,
                        "explore": 1,
                        "communicate": 2,
                        "rest": 3,
                        "defend": 4,
                    }
                    behavior_values.append(
                        float(action_map.get(behavior["action_type"], 0))
                    )
                    timestamps.append(behavior.get("timestamp", datetime.now()))

            if len(behavior_values) < 10:
                return {"status": "insufficient_data"}

            behavior_array = np.array(behavior_values)

            # Track divergence of nearby trajectories after phase-space
            # reconstruction with delay embedding.

            # Create delay embedding (embedding dimension = 3, delay = 1)
            embedding_dim = 3
            delay = 1

            if len(behavior_array) < embedding_dim + delay:
                return {"status": "insufficient_data"}

            # Create phase space vectors
            phase_space_list = []
            for i in range(len(behavior_array) - (embedding_dim - 1) * delay):
                vector = [behavior_array[i + j * delay] for j in range(embedding_dim)]
                phase_space_list.append(vector)

            phase_space = np.array(phase_space_list)

            if len(phase_space) < 5:
                return {"status": "insufficient_data"}

            # Find nearest neighbors and track divergence
            divergences = []
            max_iterations = min(20, len(phase_space) - 1)

            for i in range(min(10, len(phase_space) - max_iterations)):
                # Find nearest neighbor
                distances = np.linalg.norm(phase_space - phase_space[i], axis=1)
                distances[i] = np.inf  # Exclude self
                nearest_idx = int(np.argmin(distances))

                if nearest_idx >= len(phase_space) - 1:
                    continue

                # Track divergence over time
                initial_distance = distances[nearest_idx]
                if initial_distance < 1e-10:
                    continue

                for j in range(
                    1, min(max_iterations, len(phase_space) - max(nearest_idx, i))
                ):
                    if i + j >= len(phase_space) or nearest_idx + j >= len(phase_space):
                        break

                    current_distance = np.linalg.norm(
                        phase_space[i + j] - phase_space[nearest_idx + j]
                    )

                    if current_distance > 0 and initial_distance > 0:
                        divergence_rate = (
                            np.log(current_distance / initial_distance) / j
                        )
                        if np.isfinite(divergence_rate):
                            divergences.append(divergence_rate)

            if len(divergences) == 0:
                # Fallback: estimate from variance growth
                if len(behavior_array) > 5:
                    first_half_var = np.var(behavior_array[: len(behavior_array) // 2])
                    second_half_var = np.var(behavior_array[len(behavior_array) // 2 :])
                    if first_half_var > 0:
                        growth_rate = np.log(second_half_var / first_half_var) / (
                            len(behavior_array) / 2
                        )
                        lyapunov_exp = max(0.0, growth_rate / 2.0)
                    else:
                        lyapunov_exp = 0.0
                else:
                    lyapunov_exp = 0.0
            else:
                # Average divergence rate
                lyapunov_exp = np.mean(divergences)
                lyapunov_exp = max(0.0, lyapunov_exp)

            return {
                "max_lyapunov_exponent": float(lyapunov_exp),
                "chaos_detected": lyapunov_exp > 0.1,  # Threshold for chaos
                "predictability": (
                    "low"
                    if lyapunov_exp > 0.1
                    else "medium"
                    if lyapunov_exp > 0.05
                    else "high"
                ),
                "divergence_samples": len(divergences),
            }

        except Exception as e:
            logger.warning(f"Lyapunov exponent calculation failed: {e}")
            return {"status": "calculation_failed", "error": str(e)}

    def _assess_emergence(self, emergence_results: dict[str, Any]) -> bool:
        """Assess whether emergence has been detected."""
        try:
            # Check information theory measures
            info_measures = emergence_results.get("information_theory", {})

            mi_score = info_measures.get("mutual_information", {}).get(
                "mutual_information_score", 0.0
            )
            te_score = info_measures.get("transfer_entropy", {}).get(
                "transfer_entropy_score", 0.0
            )

            # Check complexity measures
            complexity_measures = emergence_results.get("complexity_analysis", {})

            fractal_dim = complexity_measures.get("fractal_dimension", {}).get(
                "fractal_dimension", 0.0
            )
            lyapunov_exp = complexity_measures.get("lyapunov_exponents", {}).get(
                "max_lyapunov_exponent", 0.0
            )

            # Emergence criteria
            information_criterion = (mi_score > 0.3) or (te_score > 0.2)
            complexity_criterion = (fractal_dim > 1.2) or (lyapunov_exp > 0.05)

            return bool(information_criterion and complexity_criterion)

        except Exception as e:
            logger.warning(f"Emergence assessment failed: {e}")
            return False

    def _interpret_emergence(self, emergence_results: dict[str, Any]) -> dict[str, Any]:
        """Generate interpretation of emergence detection results."""
        interpretation: dict[str, Any] = {
            "emergence_level": "none",
            "key_characteristics": [],
            "system_complexity": "unknown",
            "predictability": "unknown",
            "recommendations": [],
        }

        try:
            emergence_detected = emergence_results.get("emergence_detected", False)

            if emergence_detected:
                interpretation["emergence_level"] = "detected"

                # Analyze characteristics
                info_measures = emergence_results.get("information_theory", {})
                complexity_measures = emergence_results.get("complexity_analysis", {})

                mi_score = info_measures.get("mutual_information", {}).get(
                    "mutual_information_score", 0.0
                )
                if mi_score > 0.5:
                    interpretation["key_characteristics"].append(
                        "Strong information coupling between individuals and collective"
                    )

                fractal_dim = complexity_measures.get("fractal_dimension", {}).get(
                    "fractal_dimension", 0.0
                )
                if fractal_dim > 1.5:
                    interpretation["key_characteristics"].append(
                        "Complex spatial organization patterns"
                    )

                lyapunov_exp = complexity_measures.get("lyapunov_exponents", {}).get(
                    "max_lyapunov_exponent", 0.0
                )
                if lyapunov_exp > 0.1:
                    interpretation["key_characteristics"].append(
                        "Chaotic dynamics with sensitive dependence on initial conditions"
                    )
                    interpretation["predictability"] = "low"
                else:
                    interpretation["predictability"] = "moderate"

                interpretation["system_complexity"] = (
                    "high" if fractal_dim > 1.5 else "medium"
                )

                # Generate recommendations
                if lyapunov_exp > 0.1:
                    interpretation["recommendations"].append(
                        "Consider chaos control strategies to improve predictability"
                    )

                if mi_score > 0.7:
                    interpretation["recommendations"].append(
                        "Leverage strong information coupling for enhanced coordination"
                    )

            else:
                interpretation["emergence_level"] = "not_detected"
                interpretation["key_characteristics"].append(
                    "Limited collective organization observed"
                )
                interpretation["system_complexity"] = "low"
                interpretation["predictability"] = "high"

        except Exception as e:
            logger.warning(f"Emergence interpretation failed: {e}")
            interpretation["error"] = str(e)

        return interpretation

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary of all analysis results."""
        summary = {
            "total_analyses": len(self.analysis_history),
            "analysis_types_performed": list(
                set([a.get("analysis_type", "unknown") for a in self.analysis_history])
            ),
            "pattern_cache_size": len(self.pattern_cache),
            "last_analysis_time": (
                self.analysis_history[-1]["analysis_time"].isoformat()
                if self.analysis_history
                else None
            ),
        }

        # Summarize cached results
        if self.pattern_cache:
            summary["cached_analyses"] = list(self.pattern_cache.keys())

        # Performance statistics
        if self.analysis_history:
            analysis_times = [
                (datetime.now() - a["analysis_time"]).total_seconds()
                for a in self.analysis_history
            ]
            summary["performance"] = {
                "avg_analysis_time": np.mean(analysis_times),
                "total_analysis_time": sum(analysis_times),
                "analyses_per_hour": len(self.analysis_history)
                / max(1, sum(analysis_times) / 3600),
            }

        return summary
