"""
Tests for the graph_theory module.
"""

import numpy as np
from geo_infer_math.core.graph_theory import (
    GraphNode,
    GraphEdge,
    SpatialGraph,
    NetworkFlow,
)


class TestGraphNode:
    """Test GraphNode class."""

    def test_node_creation(self):
        """Test creating graph nodes."""
        node = GraphNode(
            id="node1",
            coordinates=np.array([10.0, 20.0]),
            attributes={"type": "residential", "population": 1000},
        )

        assert node.id == "node1"
        np.testing.assert_array_equal(node.coordinates, np.array([10.0, 20.0]))
        assert node.attributes["type"] == "residential"
        assert node.attributes["population"] == 1000

    def test_node_without_coordinates(self):
        """Test creating node without coordinates."""
        node = GraphNode(id="node1")
        assert node.coordinates is None
        assert node.attributes == {}


class TestGraphEdge:
    """Test GraphEdge class."""

    def test_edge_creation(self):
        """Test creating graph edges."""
        edge = GraphEdge(
            source="node1",
            target="node2",
            weight=5.0,
            attributes={"road_type": "highway", "length": 100},
        )

        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.weight == 5.0
        assert edge.attributes["road_type"] == "highway"
        assert edge.attributes["length"] == 100


class TestSpatialGraph:
    """Test SpatialGraph class."""

    def setup_method(self):
        """Set up test graph."""
        self.graph = SpatialGraph(directed=False)

        # Add nodes
        self.graph.add_node("A", np.array([0, 0]), type="origin")
        self.graph.add_node("B", np.array([1, 0]), type="intermediate")
        self.graph.add_node("C", np.array([1, 1]), type="intermediate")
        self.graph.add_node("D", np.array([0, 1]), type="destination")

        # Add edges
        self.graph.add_edge("A", "B", 1.0, road_type="street")
        self.graph.add_edge("B", "C", 1.0, road_type="street")
        self.graph.add_edge("C", "D", 1.0, road_type="street")
        self.graph.add_edge("D", "A", 1.0, road_type="street")
        self.graph.add_edge("A", "C", 1.4, road_type="diagonal")  # Direct diagonal

    def test_graph_creation(self):
        """Test graph creation and basic properties."""
        assert len(self.graph.nodes) == 4
        assert len(self.graph.edges) == 10  # Undirected, so each edge is stored twice
        assert not self.graph.directed

    def test_node_operations(self):
        """Test node operations."""
        # Check neighbors
        neighbors_a = self.graph.get_neighbors("A")
        assert set(neighbors_a) == {"B", "D", "C"}

        # Check edge weights
        weight_ab = self.graph.get_edge_weight("A", "B")
        assert weight_ab == 1.0

        weight_ac = self.graph.get_edge_weight("A", "C")
        assert weight_ac == 1.4

    def test_shortest_path_dijkstra(self):
        """Test Dijkstra's shortest path algorithm."""
        path, distance = self.graph.shortest_path("A", "C", algorithm="dijkstra")

        assert path == ["A", "B", "C"] or path == ["A", "C"]
        assert distance > 0

        # Direct path should be shorter than A->B->C
        _, direct_distance = self.graph.shortest_path("A", "C")
        assert direct_distance <= 2.0  # Should use direct edge

    def test_shortest_path_bellman_ford(self):
        """Test Bellman-Ford shortest path algorithm."""
        path, distance = self.graph.shortest_path("A", "D", algorithm="bellman_ford")

        assert isinstance(path, list)
        assert isinstance(distance, float)
        assert distance > 0

    def test_minimum_spanning_tree(self):
        """Test minimum spanning tree calculation."""
        mst = self.graph.minimum_spanning_tree()

        # MST should have n-1 edges for n nodes
        assert len(mst.edges) == 6  # 4 nodes - 1 = 3 edges, but undirected so 6 entries

        # Check that all nodes are connected
        components = mst.connected_components()
        assert len(components) == 1
        assert len(components[0]) == 4

    def test_connected_components(self):
        """Test connected components analysis."""
        # Current graph should be connected
        components = self.graph.connected_components()
        assert len(components) == 1
        assert set(components[0]) == {"A", "B", "C", "D"}

        # Create disconnected graph
        disconnected_graph = SpatialGraph()
        disconnected_graph.add_node("X", np.array([0, 0]))
        disconnected_graph.add_node("Y", np.array([10, 10]))
        disconnected_graph.add_edge("X", "X", 1.0)  # Self-loop doesn't connect

        components = disconnected_graph.connected_components()
        assert len(components) == 2

    def test_centrality_measures(self):
        """Test centrality measure calculations."""
        centrality = self.graph.centrality_measures()

        assert "degree" in centrality
        assert "betweenness" in centrality
        assert "closeness" in centrality

        # All nodes should have centrality measures
        for measure_values in centrality.values():
            assert len(measure_values) == 4
            assert all(v >= 0 for v in measure_values.values())

    def test_spatial_network_analysis(self):
        """Test comprehensive spatial network analysis."""
        analysis = self.graph.spatial_network_analysis()

        assert "n_nodes" in analysis
        assert "n_edges" in analysis
        assert "density" in analysis
        assert "n_components" in analysis
        assert "centrality" in analysis

        assert analysis["n_nodes"] == 4
        assert analysis["n_components"] == 1

    def test_directed_graph(self):
        """Test directed graph functionality."""
        directed_graph = SpatialGraph(directed=True)

        directed_graph.add_node("A")
        directed_graph.add_node("B")
        directed_graph.add_edge("A", "B", 1.0)  # Only A -> B

        assert directed_graph.directed
        assert "B" in directed_graph.get_neighbors("A")
        assert "A" not in directed_graph.get_neighbors("B")  # No reverse edge

    def test_node_removal(self):
        """Test node removal."""
        original_nodes = len(self.graph.nodes)
        original_edges = len(self.graph.edges)

        self.graph.remove_node("C")

        assert len(self.graph.nodes) == original_nodes - 1
        assert len(self.graph.edges) < original_edges  # Should have fewer edges

        # Removed node should not be in neighbors
        for node_id in self.graph.nodes:
            assert "C" not in self.graph.get_neighbors(node_id)


class TestNetworkFlow:
    """Test network flow algorithms."""

    def setup_method(self):
        """Set up test network."""
        self.graph = SpatialGraph(directed=True)

        # Create a simple flow network
        nodes = ["S", "A", "B", "C", "T"]  # S=source, T=sink
        for node in nodes:
            self.graph.add_node(node)

        # Add edges with capacities
        self.graph.add_edge("S", "A", 10)
        self.graph.add_edge("S", "B", 5)
        self.graph.add_edge("A", "B", 15)
        self.graph.add_edge("A", "C", 10)
        self.graph.add_edge("B", "C", 10)
        self.graph.add_edge("B", "T", 10)
        self.graph.add_edge("C", "T", 10)

    def test_max_flow(self):
        """Test maximum flow calculation."""
        max_flow, flow_dict = NetworkFlow.max_flow(self.graph, "S", "T")

        assert isinstance(max_flow, (int, float))
        assert max_flow >= 0
        assert isinstance(flow_dict, dict)

        # Check that flow doesn't exceed capacities
        for (u, v), flow in flow_dict.items():
            if flow > 0:
                capacity = self.graph.get_edge_weight(u, v)
                assert flow <= capacity

    def test_residual_graph_creation(self):
        """Test residual graph creation."""
        residual = NetworkFlow._create_residual_graph(self.graph)

        assert isinstance(residual, dict)

        # Check that all original edges are in residual graph
        for (u, v), edge in self.graph.edges.items():
            assert (u, v) in residual
            assert residual[(u, v)] == edge.weight

    def test_augmenting_path(self):
        """Test augmenting path finding."""
        residual = NetworkFlow._create_residual_graph(self.graph)
        path = NetworkFlow._find_augmenting_path(residual, "S", "T")

        if path is not None:
            assert path[0] == "S"
            assert path[-1] == "T"
            assert len(path) >= 2

            # Check that all consecutive pairs have residual capacity
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                assert residual.get((u, v), 0) > 0


class TestComplexGraph:
    """Test with more complex graph structures."""

    def test_large_graph_centrality(self):
        """Test centrality measures on larger graph."""
        # Create a grid graph
        graph = SpatialGraph(directed=False)

        size = 5
        for i in range(size):
            for j in range(size):
                node_id = f"{i}_{j}"
                graph.add_node(node_id, np.array([i, j]))

                # Connect to neighbors
                if i > 0:
                    graph.add_edge(node_id, f"{i - 1}_{j}", 1.0)
                if j > 0:
                    graph.add_edge(node_id, f"{i}_{j - 1}", 1.0)

        centrality = graph.centrality_measures()

        # Corner nodes should have lower centrality than center nodes
        corner_centrality = centrality["degree"]["0_0"]
        center_centrality = centrality["degree"]["2_2"]

        assert corner_centrality < center_centrality

    def test_graph_with_isolated_nodes(self):
        """Test graph with isolated nodes."""
        graph = SpatialGraph()

        # Add connected component
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B", 1.0)

        # Add isolated node
        graph.add_node("C")

        components = graph.connected_components()
        assert len(components) == 2

        # One component with 2 nodes, one with 1 node
        component_sizes = [len(comp) for comp in components]
        assert set(component_sizes) == {1, 2}
