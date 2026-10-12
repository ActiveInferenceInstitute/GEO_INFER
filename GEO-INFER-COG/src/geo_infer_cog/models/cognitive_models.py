"""
Cognitive Models for GEO-INFER-COG

This module defines data models and schemas for cognitive representations
of spatial knowledge, including cognitive maps, spatial knowledge graphs,
and other cognitive structures used in geospatial reasoning.

Key Components:
- Cognitive Map representations for mental spatial models
- Spatial Knowledge Graph for structured spatial knowledge
- Cognitive Profile models for user spatial cognition
- Spatial Concept models for abstract spatial understanding

Mathematical Foundations:
- Graph theory for spatial knowledge representation
- Cognitive map models (Kuipers, 1978)
- Spatial knowledge representation frameworks
- Mental model theories (Johnson-Laird, 1983)
"""

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import networkx as nx
import numpy as np

from ..models.user_profiles import UserCognitiveProfile
from ..utils.rng import resolve_rng

# Module-level monotonic counter backing fallback entity IDs.
_ENTITY_ID_SEQUENCE: "itertools.count[int]" = itertools.count(1)

logger = logging.getLogger(__name__)


@dataclass
class SpatialNode:
    """Represents a node in a spatial knowledge graph."""

    node_id: str
    node_type: str  # 'location', 'region', 'landmark', 'concept', 'relation'
    properties: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] | None = None
    saliency: float = 0.5
    accessibility: float = 1.0
    uncertainty: float = 0.0

    def calculate_cognitive_weight(
        self, user_profile: UserCognitiveProfile | None = None
    ) -> float:
        """Calculate cognitive weight for this node."""
        base_weight = self.saliency * self.accessibility

        if user_profile:
            # Adjust based on user expertise and preferences
            expertise_bonus = user_profile.spatial_expertise * 0.2
            preference_multiplier = 1.0

            if (
                user_profile.cognitive_load_preference == "low"
                and self.uncertainty > 0.5
            ):
                preference_multiplier = 0.8
            elif (
                user_profile.cognitive_load_preference == "high" and self.saliency > 0.7
            ):
                preference_multiplier = 1.2

            return min(1.0, base_weight + expertise_bonus * preference_multiplier)

        return base_weight


@dataclass
class SpatialEdge:
    """Represents an edge (relationship) in a spatial knowledge graph."""

    edge_id: str
    source_node: str
    target_node: str
    relation_type: (
        str  # 'topological', 'directional', 'distance', 'functional', 'conceptual'
    )
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    strength: float = 1.0
    directionality: str = (
        "bidirectional"  # 'unidirectional', 'bidirectional', 'asymmetric'
    )

    def get_effective_strength(
        self, user_profile: UserCognitiveProfile | None = None
    ) -> float:
        """Get effective relationship strength considering user factors."""
        base_strength = self.strength * self.confidence

        if user_profile:
            # Adjust based on user cognitive style
            if user_profile.spatial_reasoning_style == "qualitative":
                # Prefer topological and functional relations
                if self.relation_type in ["topological", "functional"]:
                    return min(1.0, base_strength * 1.1)
            elif user_profile.spatial_reasoning_style == "quantitative":
                # Prefer distance and directional relations
                if self.relation_type in ["distance", "directional"]:
                    return min(1.0, base_strength * 1.1)

        return base_strength


class CognitiveMap:
    """
    Cognitive map representation for mental spatial models.

    This class implements cognitive map structures that model how humans
    organize and represent spatial knowledge, including landmark-based
    navigation, route knowledge, and survey knowledge.

    The cognitive map includes:
    - Landmark nodes with saliency and accessibility
    - Route segments between landmarks
    - Regional organization and hierarchies
    - Distortion effects (cognitive biases in spatial representation)
    - Integration with long-term spatial memory
    """

    def __init__(
        self,
        map_id: str,
        spatial_bounds: dict[str, Any],
        cognitive_framework: str = "landmark_based",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize cognitive map.

        Args:
            map_id: Unique identifier for this cognitive map
            spatial_bounds: Geographic bounds of the mapped area
            cognitive_framework: Cognitive mapping approach ('landmark_based', 'route_based', 'survey')
            config: Additional configuration parameters
        """
        self.map_id = map_id
        self.spatial_bounds = spatial_bounds
        self.cognitive_framework = cognitive_framework
        self.config = config or {}

        # Map components
        self.landmarks: dict[str, SpatialNode] = {}  # Landmark nodes
        self.routes: dict[str, Any] = {}  # Route segments
        self.regions: dict[str, Any] = {}  # Regional organization
        self.connections: dict[str, list[dict[str, Any]]] = {}  # Interconnections

        # Cognitive properties
        self.distortion_factors = {
            "distance_distortion": 0.1,
            "direction_distortion": 0.05,
            "landmark_bias": 0.2,
        }

        # Performance tracking
        self.map_metrics = {
            "landmarks_count": 0,
            "routes_count": 0,
            "regions_count": 0,
            "navigation_accuracy": 0.0,
            "cognitive_load": 0.0,
        }

        # Integration with spatial memory
        self.memory_integration = True
        self.last_memory_sync = None

        logger.info(
            f"Cognitive Map {map_id} initialized with framework: {cognitive_framework}"
        )

    def add_landmark(
        self,
        landmark_id: str,
        geometry: dict[str, Any],
        properties: dict[str, Any],
        saliency: float = 0.5,
    ) -> None:
        """
        Add a landmark to the cognitive map.

        Args:
            landmark_id: Unique identifier for the landmark
            geometry: Geographic location and extent
            properties: Landmark properties (name, type, significance)
            saliency: Visual/cognitive saliency (0-1)
        """
        landmark = SpatialNode(
            node_id=landmark_id,
            node_type="landmark",
            geometry=geometry,
            properties=properties,
            saliency=saliency,
        )

        self.landmarks[landmark_id] = landmark
        self.map_metrics["landmarks_count"] += 1

        logger.info(f"Landmark {landmark_id} added to cognitive map")

    def add_route(
        self,
        route_id: str,
        start_landmark: str,
        end_landmark: str,
        segments: list[dict[str, Any]],
        properties: dict[str, Any],
    ) -> None:
        """
        Add a route between landmarks.

        Args:
            route_id: Unique identifier for the route
            start_landmark: Starting landmark ID
            end_landmark: Ending landmark ID
            segments: Route geometry segments
            properties: Route properties (length, difficulty, mode)
        """
        if start_landmark not in self.landmarks or end_landmark not in self.landmarks:
            raise ValueError("Route landmarks must be added to map first")

        route = {
            "route_id": route_id,
            "start_landmark": start_landmark,
            "end_landmark": end_landmark,
            "segments": segments,
            "properties": properties,
            "cognitive_complexity": self._calculate_route_complexity(
                segments, properties
            ),
        }

        self.routes[route_id] = route
        self.map_metrics["routes_count"] += 1

        # Add bidirectional connections
        if start_landmark not in self.connections:
            self.connections[start_landmark] = []
        if end_landmark not in self.connections:
            self.connections[end_landmark] = []

        self.connections[start_landmark].append(
            {"target": end_landmark, "route_id": route_id, "connection_type": "route"}
        )

        self.connections[end_landmark].append(
            {"target": start_landmark, "route_id": route_id, "connection_type": "route"}
        )

        logger.info(
            f"Route {route_id} added between {start_landmark} and {end_landmark}"
        )

    def add_region(
        self,
        region_id: str,
        boundary: list[tuple[float, float]],
        properties: dict[str, Any],
        landmark_composition: list[str],
    ) -> None:
        """
        Add a region to the cognitive map.

        Args:
            region_id: Unique identifier for the region
            boundary: Region boundary coordinates
            properties: Region properties (name, type, function)
            landmark_composition: Landmarks within this region
        """
        region = SpatialNode(
            node_id=region_id,
            node_type="region",
            geometry={"type": "Polygon", "coordinates": [boundary]},
            properties=properties,
            saliency=self._calculate_region_saliency(landmark_composition),
        )

        self.regions[region_id] = region
        self.map_metrics["regions_count"] += 1

        # Update landmark-region relationships
        for landmark_id in landmark_composition:
            if landmark_id in self.landmarks:
                self.landmarks[landmark_id].properties["region"] = region_id

        logger.info(f"Region {region_id} added to cognitive map")

    def _calculate_route_complexity(
        self, segments: list[dict[str, Any]], properties: dict[str, Any]
    ) -> float:
        """Calculate cognitive complexity of a route."""
        # Base complexity from length and turns
        length = properties.get("length", 0)
        turns = len(segments) - 1 if segments else 0

        complexity = min(1.0, (length / 1000.0) + (turns / 10.0))

        # Adjust for route mode
        mode_multipliers = {
            "walking": 1.0,
            "driving": 0.8,
            "public_transport": 1.2,
            "cycling": 1.1,
        }

        mode_multiplier = mode_multipliers.get(properties.get("mode", "walking"), 1.0)
        complexity *= mode_multiplier

        return cast(float, complexity)

    def _calculate_region_saliency(self, landmark_composition: list[str]) -> float:
        """Calculate saliency of a region based on its landmarks."""
        if not landmark_composition:
            return 0.3

        # Average saliency of contained landmarks
        landmark_saliencies = [
            self.landmarks.get(lid, SpatialNode(lid, "landmark")).saliency
            for lid in landmark_composition
        ]

        return float(np.mean(landmark_saliencies))

    def get_navigation_path(
        self,
        start_landmark: str,
        end_landmark: str,
        user_profile: UserCognitiveProfile | None = None,
    ) -> list[str]:
        """
        Generate navigation path between landmarks.

        Args:
            start_landmark: Starting landmark ID
            end_landmark: Target landmark ID
            user_profile: User cognitive profile for personalized routing

        Returns:
            List of landmark IDs representing the navigation path
        """
        if start_landmark not in self.landmarks or end_landmark not in self.landmarks:
            return []

        # Simple shortest path in landmark graph
        try:
            # Create graph from landmark connections
            graph = nx.Graph()

            # Add landmark nodes
            for landmark_id in self.landmarks:
                graph.add_node(landmark_id)

            # Add edges based on routes
            for route in self.routes.values():
                start = route["start_landmark"]
                end = route["end_landmark"]
                complexity = route["cognitive_complexity"]

                # Use inverse complexity as weight (simpler routes preferred)
                weight = 1.0 / (1.0 + complexity)
                graph.add_edge(start, end, weight=weight)

            # Find shortest path
            if nx.has_path(graph, start_landmark, end_landmark):
                path = nx.shortest_path(
                    graph, start_landmark, end_landmark, weight="weight"
                )

                # Apply cognitive distortions
                distorted_path = self._apply_cognitive_distortions(path, user_profile)

                logger.info(f"Navigation path generated: {' -> '.join(path)}")
                return distorted_path

        except Exception as e:
            logger.error(f"Error generating navigation path: {str(e)}")

        return []

    def _apply_cognitive_distortions(
        self, path: list[str], user_profile: UserCognitiveProfile | None = None
    ) -> list[str]:
        """Apply cognitive distortions to navigation path."""
        if len(path) <= 2:
            return path

        distorted_path = [path[0]]

        for i in range(1, len(path) - 1):
            current = path[i]
            prev = path[i - 1]
            next_ = path[i + 1]

            # Get route complexity between current and neighbors
            prev_route = self._find_route_between(prev, current)
            next_route = self._find_route_between(current, next_)

            if prev_route and next_route:
                prev_complexity = prev_route["cognitive_complexity"]
                next_complexity = next_route["cognitive_complexity"]

                # Apply distortion based on relative complexity
                if next_complexity > prev_complexity * 1.5:
                    # Skip intermediate landmark if next route is much more complex
                    # (cognitive shortcut)
                    if user_profile and user_profile.cognitive_load_preference == "low":
                        continue

            distorted_path.append(current)

        distorted_path.append(path[-1])
        return distorted_path

    def _find_route_between(
        self, landmark1: str, landmark2: str
    ) -> dict[str, Any] | None:
        """Find route between two landmarks."""
        for route in self.routes.values():
            if (
                route["start_landmark"] == landmark1
                and route["end_landmark"] == landmark2
            ) or (
                route["start_landmark"] == landmark2
                and route["end_landmark"] == landmark1
            ):
                return cast(dict[str, Any], route)
        return None

    def calculate_cognitive_load(
        self, user_profile: UserCognitiveProfile | None = None
    ) -> float:
        """Calculate cognitive load for using this map."""
        # Base load from map complexity
        landmark_count = len(self.landmarks)
        route_count = len(self.routes)
        region_count = len(self.regions)

        complexity_score = landmark_count * 0.1 + route_count * 0.2 + region_count * 0.3

        # Adjust for user profile
        if user_profile:
            if user_profile.spatial_expertise < 0.5:
                # Novice users experience higher load
                complexity_score *= 1.3
            elif user_profile.spatial_expertise > 0.8:
                # Expert users experience lower load
                complexity_score *= 0.8

            # Adjust for cognitive load preference
            if user_profile.cognitive_load_preference == "low":
                complexity_score *= 1.2
            elif user_profile.cognitive_load_preference == "high":
                complexity_score *= 0.9

        self.map_metrics["cognitive_load"] = min(1.0, complexity_score)
        return self.map_metrics["cognitive_load"]

    def get_map_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the cognitive map."""
        stats = {
            "map_id": self.map_id,
            "framework": self.cognitive_framework,
            "spatial_bounds": self.spatial_bounds,
            "components": {
                "landmarks": len(self.landmarks),
                "routes": len(self.routes),
                "regions": len(self.regions),
            },
            "connectivity": self._analyze_connectivity(),
            "cognitive_properties": {
                "average_saliency": self._calculate_average_saliency(),
                "complexity_distribution": self._analyze_complexity_distribution(),
                "distortion_factors": self.distortion_factors,
            },
            "performance": self.map_metrics.copy(),
        }

        return stats

    def _analyze_connectivity(self) -> dict[str, Any]:
        """Analyze connectivity of the landmark network."""
        if not self.landmarks:
            return {"connected_components": 0, "average_degree": 0}

        # Create network graph
        graph = nx.Graph()
        for landmark_id in self.landmarks:
            graph.add_node(landmark_id)

        for connection_list in self.connections.values():
            for connection in connection_list:
                graph.add_edge(landmark_id, connection["target"])

        # Calculate connectivity metrics
        connected_components = nx.number_connected_components(graph)
        average_degree = (
            sum(dict(graph.degree()).values()) / len(graph.nodes())
            if graph.nodes()
            else 0
        )

        return {
            "connected_components": connected_components,
            "average_degree": average_degree,
            "is_connected": connected_components == 1,
            "network_density": nx.density(graph),
        }

    def _calculate_average_saliency(self) -> float:
        """Calculate average saliency of all landmarks."""
        if not self.landmarks:
            return 0.0

        saliencies = [landmark.saliency for landmark in self.landmarks.values()]
        return float(np.mean(saliencies))

    def _analyze_complexity_distribution(self) -> dict[str, float]:
        """Analyze distribution of route complexities."""
        if not self.routes:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        complexities = [route["cognitive_complexity"] for route in self.routes.values()]

        return {
            "mean": float(np.mean(complexities)),
            "std": float(np.std(complexities)),
            "min": float(np.min(complexities)),
            "max": float(np.max(complexities)),
        }

    def export_to_geojson(self) -> dict[str, Any]:
        """Export cognitive map as GeoJSON for visualization."""
        geojson: dict[str, Any] = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "map_id": self.map_id,
                "framework": self.cognitive_framework,
                "export_time": datetime.now().isoformat(),
            },
        }

        # Add landmark features
        for landmark_id, landmark in self.landmarks.items():
            feature = {
                "type": "Feature",
                "geometry": landmark.geometry,
                "properties": {
                    "id": landmark_id,
                    "type": "landmark",
                    "name": landmark.properties.get("name", landmark_id),
                    "saliency": landmark.saliency,
                    "accessibility": landmark.accessibility,
                },
            }
            geojson["features"].append(feature)

        # Add region features
        for region_id, region in self.regions.items():
            feature = {
                "type": "Feature",
                "geometry": region.geometry,
                "properties": {
                    "id": region_id,
                    "type": "region",
                    "name": region.properties.get("name", region_id),
                    "saliency": region.saliency,
                },
            }
            geojson["features"].append(feature)

        return geojson


class SpatialKnowledgeGraph:
    """
    Graph-based representation of spatial knowledge and relationships.

    This class implements a knowledge graph structure for representing
    spatial concepts, relationships, and semantic associations in a
    machine-readable format suitable for reasoning and inference.

    The knowledge graph includes:
    - Spatial entities (locations, regions, landmarks)
    - Spatial relationships (topological, directional, functional)
    - Conceptual associations and semantic links
    - Uncertainty and confidence measures
    - Temporal aspects of spatial knowledge
    """

    def __init__(
        self,
        graph_id: str,
        domain: str = "general",
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize spatial knowledge graph.

        Args:
            graph_id: Unique identifier for this knowledge graph
            domain: Knowledge domain ('urban', 'environmental', 'transportation', 'general')
            config: Additional configuration parameters
            rng: Optional random generator for fallback entity-ID suffixes.
                When omitted, a fixed-seed generator is used so ID generation
                is deterministic by default.
        """
        self.graph_id = graph_id
        self.domain = domain
        self.config = config or {}

        # Resolved via the repo-wide resolve_rng pattern; None resolves to a
        # fixed seed so ID generation is reproducible by default.
        self._rng = resolve_rng(rng)

        # Graph structure using NetworkX
        self.graph = nx.DiGraph()
        self.graph.graph["domain"] = domain
        self.graph.graph["created"] = datetime.now().isoformat()

        # Node and edge indexes for efficient querying
        self.node_index: dict[str, Any] = {}  # node_id -> node_data
        self.edge_index: dict[
            tuple[str, str], Any
        ] = {}  # (source, target) -> edge_data

        # Knowledge organization

        self.ontologies: dict[str, Any] = {}  # Domain ontologies
        self.taxonomies: dict[str, Any] = {}  # Hierarchical classifications

        # Performance tracking
        self.graph_metrics = {
            "nodes_count": 0,
            "edges_count": 0,
            "average_degree": 0.0,
            "connected_components": 0,
            "knowledge_density": 0.0,
        }

        logger.info(
            f"Spatial Knowledge Graph {graph_id} initialized for domain: {domain}"
        )

    def add_spatial_entity(
        self,
        entity_id: str,
        entity_type: str,
        geometry: dict[str, Any] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a spatial entity to the knowledge graph.

        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of spatial entity ('location', 'region', 'landmark', 'feature')
            geometry: Geographic representation
            properties: Entity properties and attributes
        """
        properties = properties or {}

        # Create node data
        node_data = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "geometry": geometry,
            "properties": properties,
            "created": datetime.now().isoformat(),
            "confidence": properties.get("confidence", 1.0),
            "saliency": properties.get("saliency", 0.5),
        }

        # Add to NetworkX graph
        self.graph.add_node(entity_id, **node_data)

        # Update indexes
        self.node_index[entity_id] = node_data

        # Update metrics
        self.graph_metrics["nodes_count"] = self.graph.number_of_nodes()
        self.graph_metrics["average_degree"] = self._calculate_average_degree()

        logger.info(f"Spatial entity {entity_id} added to knowledge graph")

    def add_spatial_relationship(
        self,
        source_entity: str,
        target_entity: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a spatial relationship between entities.

        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID
            relation_type: Type of relationship ('contains', 'adjacent', 'north_of', 'connected_to')
            properties: Relationship properties
        """
        properties = properties or {}

        if source_entity not in self.graph or target_entity not in self.graph:
            raise ValueError(
                "Both entities must exist in graph before adding relationship"
            )

        # Create edge data
        edge_data = {
            "relation_type": relation_type,
            "properties": properties,
            "confidence": properties.get("confidence", 1.0),
            "strength": properties.get("strength", 1.0),
            "created": datetime.now().isoformat(),
        }

        # Add to NetworkX graph
        self.graph.add_edge(source_entity, target_entity, **edge_data)

        # Update indexes
        edge_key = (source_entity, target_entity)
        self.edge_index[edge_key] = edge_data

        # Update metrics
        self.graph_metrics["edges_count"] = self.graph.number_of_edges()
        self.graph_metrics["average_degree"] = self._calculate_average_degree()

        logger.info(
            f"Spatial relationship {relation_type} added: {source_entity} -> {target_entity}"
        )

    def query_spatial_relationships(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
        max_depth: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Query spatial relationships for an entity.

        Args:
            entity_id: Entity to query relationships for
            relation_types: Filter by relationship types (optional)
            max_depth: Maximum depth for relationship traversal

        Returns:
            List of relationship dictionaries
        """
        if entity_id not in self.graph:
            return []

        relationships = []

        # Get immediate relationships
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            if edge_data:
                relation_type = edge_data.get("relation_type", "unknown")

                if relation_types is None or relation_type in relation_types:
                    relationships.append(
                        {
                            "source": entity_id,
                            "target": neighbor,
                            "relation_type": relation_type,
                            "confidence": edge_data.get("confidence", 1.0),
                            "strength": edge_data.get("strength", 1.0),
                            "properties": edge_data.get("properties", {}),
                        }
                    )

        # Multi-hop traversal for deeper queries
        if max_depth > 1:
            visited = {entity_id}
            frontier = [r["target"] for r in relationships]
            for depth in range(2, max_depth + 1):
                next_frontier: list[str] = []
                for node in frontier:
                    if node in visited or node not in self.graph:
                        continue
                    visited.add(node)
                    for neighbor in self.graph.neighbors(node):
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        if edge_data:
                            rel_type = edge_data.get("relation_type", "unknown")
                            if relation_types is None or rel_type in relation_types:
                                relationships.append(
                                    {
                                        "source": node,
                                        "target": neighbor,
                                        "relation_type": rel_type,
                                        "confidence": edge_data.get("confidence", 1.0),
                                        "strength": edge_data.get("strength", 1.0),
                                        "properties": edge_data.get("properties", {}),
                                        "depth": depth,
                                    }
                                )
                                next_frontier.append(neighbor)
                frontier = next_frontier

        return relationships

    def find_spatial_patterns(
        self, pattern_type: str = "clusters"
    ) -> list[dict[str, Any]]:
        """
        Find spatial patterns in the knowledge graph.

        Args:
            pattern_type: Type of pattern to find ('clusters', 'hierarchies', 'cycles')

        Returns:
            List of discovered patterns
        """
        patterns = []

        if pattern_type == "clusters":
            # Find densely connected components
            clusters = list(nx.connected_components(self.graph.to_undirected()))

            for i, cluster in enumerate(clusters):
                if len(cluster) > 2:  # Only meaningful clusters
                    patterns.append(
                        {
                            "pattern_id": f"cluster_{i}",
                            "pattern_type": "spatial_cluster",
                            "entities": list(cluster),
                            "size": len(cluster),
                            "density": self._calculate_cluster_density(cluster),
                        }
                    )

        elif pattern_type == "hierarchies":
            # Find hierarchical structures
            hierarchies = self._find_hierarchical_structures()

            for hierarchy in hierarchies:
                patterns.append(
                    {
                        "pattern_id": hierarchy["hierarchy_id"],
                        "pattern_type": "spatial_hierarchy",
                        "levels": hierarchy["levels"],
                        "entities": hierarchy["entities"],
                    }
                )

        return patterns

    def _calculate_cluster_density(self, cluster: set[str]) -> float:
        """Calculate density of a spatial cluster."""
        if len(cluster) < 2:
            return 0.0

        # Create subgraph for the cluster
        subgraph = self.graph.subgraph(cluster)

        # Calculate density (edges / possible edges)
        actual_edges = subgraph.number_of_edges()
        possible_edges = len(cluster) * (len(cluster) - 1) / 2

        return actual_edges / possible_edges if possible_edges > 0 else 0.0

    def _find_hierarchical_structures(self) -> list[dict[str, Any]]:
        """Find hierarchical structures in the spatial knowledge graph."""
        hierarchies = []

        # Simple hierarchy detection based on containment relationships
        containment_relations = [
            edge
            for edge in self.graph.edges(data=True)
            if edge[2].get("relation_type") == "contains"
        ]

        for relation in containment_relations:
            source, target, edge_data = relation

            hierarchy = {
                "hierarchy_id": f"hierarchy_{source}_{target}",
                "root": source,
                "levels": [
                    {"level": 0, "entities": [source]},
                    {"level": 1, "entities": [target]},
                ],
                "entities": [source, target],
            }

            hierarchies.append(hierarchy)

        return hierarchies

    def _calculate_average_degree(self) -> float:
        """Calculate average degree of the graph."""
        if self.graph.number_of_nodes() == 0:
            return 0.0

        return cast(
            float,
            sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
        )

    def get_graph_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "density": 0.0}

        # Basic metrics
        stats = {
            "graph_id": self.graph_id,
            "domain": self.domain,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "average_degree": self._calculate_average_degree(),
            "connected_components": nx.number_weakly_connected_components(self.graph),
        }

        # Entity type distribution
        entity_types: dict[str, int] = {}
        for _node, node_data in self.graph.nodes(data=True):
            entity_type = node_data.get("entity_type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        stats["entity_types"] = entity_types

        # Relationship type distribution
        relation_types: dict[str, int] = {}
        for edge_data in self.graph.edges.values():
            relation_type = edge_data.get("relation_type", "unknown")
            relation_types[relation_type] = relation_types.get(relation_type, 0) + 1

        stats["relation_types"] = relation_types

        return stats

    def export_to_jsonld(self) -> dict[str, Any]:
        """Export knowledge graph as JSON-LD for semantic web compatibility."""
        jsonld: dict[str, Any] = {
            "@context": {
                "spatial": "https://schema.org/spatial#",
                "geo": "https://schema.org/geo#",
                "entity_type": "spatial:entityType",
                "relation_type": "spatial:relationType",
                "confidence": "spatial:confidence",
                "geometry": "geo:geometry",
            },
            "@graph": [],
        }

        # Add entities as nodes
        for node_id, node_data in self.graph.nodes(data=True):
            entity = {
                "@id": f"entity:{node_id}",
                "@type": f"spatial:{node_data.get('entity_type', 'Entity')}",
                "entity_id": node_id,
                "confidence": node_data.get("confidence", 1.0),
                "saliency": node_data.get("saliency", 0.5),
            }

            if node_data.get("geometry"):
                entity["geometry"] = node_data["geometry"]

            if node_data.get("properties"):
                entity.update(node_data["properties"])

            jsonld["@graph"].append(entity)

        # Add relationships as edges
        for source, target, edge_data in self.graph.edges(data=True):
            relationship = {
                "@id": f"relation:{source}_{target}",
                "@type": "spatial:Relationship",
                "source_entity": f"entity:{source}",
                "target_entity": f"entity:{target}",
                "relation_type": edge_data.get("relation_type", "unknown"),
                "confidence": edge_data.get("confidence", 1.0),
                "strength": edge_data.get("strength", 1.0),
            }

            if edge_data.get("properties"):
                relationship.update(edge_data["properties"])

            jsonld["@graph"].append(relationship)

        return jsonld

    def import_from_geojson(self, geojson_data: dict[str, Any]) -> None:
        """Import spatial entities from GeoJSON format."""
        features = geojson_data.get("features", [])

        for feature in features:
            geometry = feature.get("geometry", {})
            properties = feature.get("properties", {})

            # Generate a deterministic fallback ID from the module-level
            # monotonic counter plus the graph's generator when supplied.
            entity_id = properties.get(
                "id",
                f"entity_{next(_ENTITY_ID_SEQUENCE)}_{int(self._rng.integers(0, 10000))}",
            )
            entity_type = properties.get("type", "location")

            # Add entity to graph
            self.add_spatial_entity(
                entity_id=entity_id,
                entity_type=entity_type,
                geometry=geometry,
                properties=properties,
            )

        logger.info(f"Imported {len(features)} spatial entities from GeoJSON")
