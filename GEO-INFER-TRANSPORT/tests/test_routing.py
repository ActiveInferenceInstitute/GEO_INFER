"""Tests for transport routing module."""

from geo_infer_transport.core.routing import (
    RoutingEngine,
    RoutingAlgorithm,
    OptimizationCriteria,
)


class TestRoutingDataclasses:
    """Tests for routing dataclass and enum creation."""

    def test_routing_algorithm_values(self) -> None:
        assert RoutingAlgorithm.DIJKSTRA.value == "dijkstra"
        assert RoutingAlgorithm.A_STAR.value == "a_star"

    def test_optimization_criteria_values(self) -> None:
        assert OptimizationCriteria.TIME.value == "time"
        assert OptimizationCriteria.DISTANCE.value == "distance"
        assert OptimizationCriteria.COST.value == "cost"


class TestRoutingEngineInit:
    """Tests for RoutingEngine initialization."""

    def test_import(self) -> None:
        assert RoutingEngine is not None

    def test_initialization(self) -> None:
        engine = RoutingEngine()
        assert engine is not None


class TestRoutingObjectives:
    """Route selection and reported totals agree for real graph inputs."""

    def test_objectives_choose_distinct_paths(self):
        """Each declared criterion minimizes its own edge attribute."""
        from types import SimpleNamespace
        import networkx as nx

        graph = nx.DiGraph()
        objectives = {
            "time": "travel_time",
            "distance": "length",
            "cost": "cost",
            "emissions": "emissions",
        }
        for objective, attribute in objectives.items():
            values = dict.fromkeys(objectives.values(), 100.0)
            values[attribute] = 1.0
            graph.add_edge("start", objective, **values)
            graph.add_edge(objective, "end", **values)
        for algorithm in RoutingAlgorithm:
            engine = RoutingEngine(
                SimpleNamespace(graph=graph), algorithm=algorithm.value
            )
            for objective in objectives:
                route = engine.route(
                    {"id": "start"}, {"id": "end"}, optimization=objective
                )
                assert route.path == ["start", objective, "end"]
                assert route.total_distance_m == (
                    2.0 if objective == "distance" else 200.0
                )
                assert route.total_time_s == (2.0 if objective == "time" else 200.0)

    def test_zero_node_identifier_is_preserved(self):
        """A zero identifier selects its graph node rather than a fallback."""
        from types import SimpleNamespace
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_edge(0, 1, length=12.0, travel_time=3.0)
        route = RoutingEngine(SimpleNamespace(graph=graph)).route(
            {"node_id": 0}, {"id": 1}
        )
        assert route.path == [0, 1]
        assert route.total_distance_m == 12.0
        assert route.total_time_s == 3.0

    def test_parallel_edge_totals_follow_selected_objective(self):
        """Parallel-road totals come from the road used by the shortest path."""
        from types import SimpleNamespace
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.add_edge("a", "b", length=100.0, travel_time=2.0)
        graph.add_edge("a", "b", length=10.0, travel_time=20.0)
        engine = RoutingEngine(SimpleNamespace(graph=graph))
        fastest = engine.route({"id": "a"}, {"id": "b"})
        shortest = engine.route({"id": "a"}, {"id": "b"}, optimization="distance")
        assert (fastest.total_distance_m, fastest.total_time_s) == (100.0, 2.0)
        assert (shortest.total_distance_m, shortest.total_time_s) == (10.0, 20.0)

    def test_unknown_objective_is_rejected(self):
        """Misspelled objectives cannot silently select distance routing."""
        import pytest

        with pytest.raises(ValueError):
            RoutingEngine().route({}, {}, optimization="unknown")
