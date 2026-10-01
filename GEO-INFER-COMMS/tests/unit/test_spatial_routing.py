"""Unit tests for advanced spatial routing: routers, balancers, queues, optimizers."""

from datetime import datetime, UTC

import pytest

from geo_infer_comms.core.spatial_routing import (
    AdaptiveRoutingEngine,
    AdvancedSpatialRouter,
    GeospatialLoadBalancer,
    GeospatialMessageQueue,
    SpatialClusteringRouter,
    SpatialCluster,
    SpatialRoutingMetrics,
    SpatialRoutingOptimizer,
)
from geo_infer_comms.models.message import MessagePriority, MessageResponse
from geo_infer_comms.models.spatial import (
    GeospatialMetadata,
    GeospatialPoint,
    SpatialFilter,
    SpatialIndex,
)

BAY_AREA_BOUNDS = {
    "min_longitude": -122.6,
    "min_latitude": 37.6,
    "max_longitude": -122.2,
    "max_latitude": 37.9,
}


def _point(lon: float = -122.4, lat: float = 37.8) -> GeospatialPoint:
    return GeospatialPoint(longitude=lon, latitude=lat)


def _message(
    lon: float = -122.4,
    lat: float = 37.8,
    geospatial: bool = True,
) -> MessageResponse:
    return MessageResponse(
        content="route me",
        sender_id="sender-1",
        recipients=["user-1"],
        message_type="text",
        priority="normal",
        geospatial_data=GeospatialMetadata(location=_point(lon, lat))
        if geospatial
        else None,
    )


def _router(strategy: str = "proximity") -> AdvancedSpatialRouter:
    router = AdvancedSpatialRouter(SpatialIndex(), routing_strategy=strategy)
    router.update_network_topology(
        {
            "hub-a": {"relay-1": 1.0, "edge-2": 10.0},
            "relay-1": {"edge-2": 1.0},
            "edge-2": {},
        }
    )
    router.register_node_location("hub-a", _point(-122.5, 37.7))
    router.register_node_location("relay-1", _point(-122.45, 37.75))
    router.register_node_location("edge-2", _point(-122.3, 37.85))
    return router


class TestAdvancedSpatialRouter:
    def test_simple_routing_without_geospatial_data(self) -> None:
        router = _router()

        routes = router.route_message(_message(geospatial=False), ["hub-a", "edge-2"])

        assert routes == {"hub-a": ["hub-a"], "edge-2": ["edge-2"]}

    def test_proximity_routing_returns_route_per_node(self) -> None:
        router = _router()

        routes = router.route_message(_message(), ["relay-1", "edge-2"])

        assert set(routes) == {"relay-1", "edge-2"}
        # Message sits closest to relay-1, so it becomes the Dijkstra source.
        assert routes["edge-2"] == ["relay-1", "edge-2"]
        assert routes["relay-1"] == ["relay-1"]

    def test_shortest_path_prefers_multi_hop_over_long_direct_edge(self) -> None:
        router = _router()
        # Move the message close to hub-a; the direct hub-a -> edge-2 edge
        # costs 10.0 while the hub-a -> relay-1 -> edge-2 path costs 2.0.
        routes = router.route_message(_message(-122.5, 37.7), ["edge-2"])

        assert routes["edge-2"] == ["hub-a", "relay-1", "edge-2"]

    def test_unknown_strategy_falls_back_to_proximity(self) -> None:
        router = _router()
        router.routing_strategy = "exotic"

        routes = router.route_message(_message(), ["edge-2"])
        assert routes["edge-2"] == ["relay-1", "edge-2"]

    def test_network_aware_strategy_uses_topology(self) -> None:
        router = _router("network_aware")

        routes = router.route_message(_message(), ["edge-2"])
        assert routes["edge-2"] == ["relay-1", "edge-2"]

    def test_load_balanced_strategy_orders_by_node_load(self) -> None:
        router = _router("load_balanced")
        router.update_node_loads({"relay-1": 0.9, "edge-2": 0.1})

        routes = router.route_message(_message(), ["relay-1", "edge-2"])

        assert list(routes) == ["edge-2", "relay-1"]

    def test_adaptive_strategy_avoids_congested_interior_node(self) -> None:
        router = _router("adaptive")
        router.update_node_loads({"relay-1": 1.0})

        routes = router.route_message(
            _message(-122.5, 37.7),
            ["edge-2"],
            routing_context={
                "urgency": "high",
                "network_conditions": {"congestion": True},
            },
        )

        # The alternative route must avoid the overloaded relay-1 even at
        # the cost of the expensive direct edge.
        assert routes["edge-2"] == ["hub-a", "edge-2"]

    def test_adaptive_strategy_defaults_to_proximity(self) -> None:
        router = _router("adaptive")

        routes = router.route_message(_message(), ["edge-2"], routing_context={})
        assert routes["edge-2"] == ["relay-1", "edge-2"]

    def test_unreachable_target_returns_direct_route(self) -> None:
        router = AdvancedSpatialRouter(SpatialIndex())
        router.update_network_topology({"island-1": {"island-2": 1.0}})
        router.register_node_location("island-1", _point(-122.5, 37.7))

        routes = router.route_message(_message(), ["remote-9"])
        assert routes["remote-9"] == ["remote-9"]

    def test_record_routing_result_updates_analytics(self) -> None:
        router = _router()
        router.record_routing_result("m1", ["hub-a"], True, latency=120.0)
        router.record_routing_result("m2", ["hub-a", "edge-2"], False)

        analytics = router.get_routing_analytics()
        assert analytics["total_routes"] == 2
        assert analytics["success_rate"] == pytest.approx(50.0)
        assert analytics["average_latency"] == pytest.approx(120.0)
        assert analytics["metrics"]["failed_routes"] == 1

    def test_routing_history_is_capped(self) -> None:
        router = _router()
        for index in range(10005):
            router.record_routing_result(f"m{index}", ["hub-a"], True)

        assert len(router.routing_history) == 10000


class TestSpatialRoutingMetrics:
    def test_to_dict_and_reset(self) -> None:
        metrics = SpatialRoutingMetrics(
            successful_routes=3, failed_routes=1, total_latency=400.0, route_count=2
        )
        report = metrics.to_dict()
        assert report["success_rate"] == pytest.approx(75.0)
        assert report["average_latency"] == pytest.approx(200.0)

        metrics.reset()
        assert metrics.to_dict()["successful_routes"] == 0
        assert metrics.total_latency == 0.0


class TestGeospatialLoadBalancer:
    def _balancer(self) -> GeospatialLoadBalancer:
        balancer = GeospatialLoadBalancer(["near", "far"], load_threshold=0.8)
        balancer.register_node_location("near", _point(-122.4, 37.8))
        balancer.register_node_location("far", _point(-74.0, 40.7))
        return balancer

    def test_select_nearest_unloaded_node(self) -> None:
        balancer = self._balancer()
        assert balancer.select_optimal_node(_point()) == "near"

    def test_exclude_and_overload_rules(self) -> None:
        balancer = self._balancer()
        balancer.update_node_load("near", 0.9)  # overloaded

        assert balancer.select_optimal_node(_point(), exclude_nodes=["far"]) is None
        assert balancer.select_optimal_node(_point()) == "far"

    def test_unknown_node_load_update_is_ignored(self) -> None:
        balancer = self._balancer()
        balancer.update_node_load("ghost", 0.5)

        assert "ghost" not in balancer.node_loads

    def test_nodes_without_locations_are_not_selectable(self) -> None:
        balancer = GeospatialLoadBalancer(["bare"])
        assert balancer.select_optimal_node(_point()) is None

    def test_load_distribution(self) -> None:
        balancer = self._balancer()
        balancer.update_node_load("near", 0.9)

        distribution = balancer.get_load_distribution()
        assert distribution["average_load"] == pytest.approx(0.45)
        assert distribution["max_load"] == pytest.approx(0.9)
        assert distribution["min_load"] == 0.0
        assert distribution["overloaded_nodes"] == ["near"]

    def test_load_history_is_capped(self) -> None:
        balancer = self._balancer()
        for index in range(101):
            balancer.update_node_load("near", float(index))

        assert len(balancer.load_history["near"]) == 100
        assert balancer.load_history["near"][-1] == 100.0


class TestSpatialClusteringRouter:
    def test_nodes_join_cluster_within_radius(self) -> None:
        router = SpatialClusteringRouter(cluster_radius_km=50)

        first = router.add_node_to_cluster("n1", _point(-122.4, 37.8))
        second = router.add_node_to_cluster("n2", _point(-122.41, 37.81))
        assert first == second == "cluster_0"
        assert router.node_clusters == {"n1": "cluster_0", "n2": "cluster_0"}

    def test_distant_nodes_form_new_cluster(self) -> None:
        router = SpatialClusteringRouter(cluster_radius_km=50)
        router.add_node_to_cluster("n1", _point(-122.4, 37.8))

        remote = router.add_node_to_cluster("n2", _point(-74.0, 40.7))

        assert remote == "cluster_1"
        assert len(router.clusters) == 2

    def test_cluster_center_tracks_members(self) -> None:
        router = SpatialClusteringRouter(cluster_radius_km=500)
        cluster_id = router.add_node_to_cluster("n1", _point(-122.0, 37.0))
        router.add_node_to_cluster("n2", _point(-122.0, 38.0))

        cluster = router.clusters[cluster_id]
        assert cluster.center.latitude == pytest.approx(37.5)

        cluster.remove_node("n2")
        assert cluster.nodes == {"n1": router.clusters[cluster_id].nodes["n1"]}
        assert cluster.center.latitude == pytest.approx(37.0)

    def test_route_to_cluster_strategies(self) -> None:
        router = SpatialClusteringRouter(cluster_radius_km=50)
        router.add_node_to_cluster("n1", _point(-122.4, 37.8))
        router.add_node_to_cluster("n2", _point(-74.0, 40.7))

        assert router.route_to_cluster(_point(-122.39, 37.79)) == "cluster_0"
        assert router.route_to_cluster(_point(-122.39, 37.79), "load_balanced") in {
            "cluster_0",
            "cluster_1",
        }
        assert router.route_to_cluster(_point(-122.39, 37.79), "unknown") == "cluster_0"
        assert SpatialClusteringRouter().route_to_cluster(_point(-122.4, 37.8)) is None

    def test_get_cluster_info(self) -> None:
        router = SpatialClusteringRouter(cluster_radius_km=50)
        cluster_id = router.add_node_to_cluster("n1", _point(-122.4, 37.8))

        info = router.get_cluster_info(cluster_id)
        assert info is not None
        assert info["node_count"] == 1
        assert info["nodes"] == ["n1"]
        assert info["radius_km"] == 50.0
        assert router.get_cluster_info("missing") is None


class TestSpatialCluster:
    def test_contains_location_uses_radius(self) -> None:
        cluster = SpatialCluster("c1", _point(-122.4, 37.8))

        assert cluster.contains_location(_point(-122.41, 37.81), radius_km=5) is True
        assert cluster.contains_location(_point(-74.0, 40.7), radius_km=5) is False

    def test_recalculate_center_skips_empty_cluster(self) -> None:
        cluster = SpatialCluster("c1", _point(-122.4, 37.8))
        cluster.remove_node("ghost")

        assert cluster.center == _point(-122.4, 37.8)


class TestAdaptiveRoutingEngine:
    def test_learn_from_routing_result(self) -> None:
        engine = AdaptiveRoutingEngine()
        message = _message()

        engine.learn_from_routing_result(
            message, ["a", "b"], {"success": True, "latency": 100.0}
        )
        engine.learn_from_routing_result(message, ["a", "b"], {"success": False})

        key = next(iter(engine.route_performance))
        performance = engine.route_performance[key]
        assert performance["sample_count"] == 2
        # EMA after (success, failure) with alpha=0.01:
        # 0.01 * (1 - 0.01) + 0.0 = 0.0099.
        assert performance["success_rate"] == pytest.approx(0.0099)
        # Weights start at 0.2, +0.01 on success then -0.01 on failure.
        assert engine.routing_weights["reliability"] == pytest.approx(0.2)

    def test_learn_updates_weights_on_success(self) -> None:
        engine = AdaptiveRoutingEngine()
        engine.learn_from_routing_result(_message(), ["a", "b"], {"success": True})

        assert engine.routing_weights["reliability"] == pytest.approx(0.21)

    def test_predict_optimal_route(self) -> None:
        engine = AdaptiveRoutingEngine()
        message = _message()
        available = {"fast": ["a"], "slow": ["a", "b", "c"]}

        # No history yet: default scoring prefers shorter, unlearned routes.
        assert engine.predict_optimal_route(message, available) == "fast"
        assert engine.predict_optimal_route(message, {}) is None

    def test_predict_prefers_learned_successful_route(self) -> None:
        engine = AdaptiveRoutingEngine()
        reliable = MessageResponse(
            content="route me",
            sender_id="sender-1",
            recipients=["user-1"],
            message_type="text",
            priority="normal",
            geospatial_data=GeospatialMetadata(location=_point()),
        )

        # Teach one route-key a strong success history.
        for _ in range(50):
            engine.learn_from_routing_result(
                reliable, ["a", "b"], {"success": True, "latency": 10.0}
            )

        # Same key shape (type, priority, length, first two nodes), but with
        # learned performance it must now beat the default-scored longer route.
        learned_key = engine._generate_route_key(reliable, ["a", "b"])
        learned_route = {learned_key: ["a", "b"], "other": ["a", "b", "c"]}
        assert engine.predict_optimal_route(reliable, learned_route) == learned_key

    def test_get_routing_insights(self) -> None:
        engine = AdaptiveRoutingEngine()
        engine.learn_from_routing_result(
            _message(), ["a", "b"], {"success": True, "latency": 5.0}
        )

        insights = engine.get_routing_insights()
        assert insights["total_routes_tracked"] == 1
        assert insights["high_performance_routes"] == 0  # single sample keeps rate low
        assert insights["routing_weights"] == engine.routing_weights


class TestGeospatialMessageQueue:
    def _queued(
        self,
    ) -> tuple[GeospatialMessageQueue, MessageResponse, MessageResponse]:
        queue = GeospatialMessageQueue()
        urgent = _message(-122.4, 37.8)
        urgent.priority = MessagePriority.URGENT.value
        low = _message(-74.0, 40.7)
        low.priority = MessagePriority.LOW.value

        queue.enqueue_message(low)
        queue.enqueue_message(urgent)
        return queue, urgent, low

    def test_dequeue_respects_priority(self) -> None:
        queue, urgent, _ = self._queued()

        assert queue.dequeue_message() is urgent
        assert queue.dequeue_message().priority == MessagePriority.LOW.value
        assert queue.dequeue_message() is None

    def test_spatial_index_and_filtered_dequeue(self) -> None:
        queue, _, low = self._queued()
        assert set(queue.spatial_index) == {"-122.400,37.800", "-74.000,40.700"}

        bay_filter_key = "-74.000,40.700"

        nyc_filter = SpatialFilter(
            filter_type="bounds",
            parameters={
                "bounds": {
                    "min_longitude": -75.0,
                    "min_latitude": 40.0,
                    "max_longitude": -73.0,
                    "max_latitude": 41.0,
                }
            },
        )

        assert queue.dequeue_message(spatial_filter=nyc_filter) is low
        # The dequeued message is gone from the spatial index.
        assert bay_filter_key not in queue.spatial_index

    def test_get_messages_by_location(self) -> None:
        queue, urgent, low = self._queued()

        nearby = queue.get_messages_by_location(_point(-122.4, 37.8), radius_km=5)
        assert [m.message_id for m in nearby] == [urgent.message_id]

        far_only = queue.get_messages_by_location(_point(-74.0, 40.7), radius_km=1)
        assert [m.message_id for m in far_only] == [low.message_id]

    def test_queue_stats_and_max_size_eviction(self) -> None:
        queue = GeospatialMessageQueue(max_size=1)
        first = _message()
        second = _message(-122.41, 37.81)

        queue.enqueue_message(first)
        queue.enqueue_message(second)

        stats = queue.get_queue_stats()
        assert stats["queue_size"] == 1
        assert stats["max_size"] == 1
        assert stats["message_store_size"] == 2

    def test_priority_score_ages_messages_forward(self) -> None:
        queue = GeospatialMessageQueue()
        message = _message()
        message.timestamp = datetime.now(UTC).replace(year=2020, month=1, day=1)

        score = queue._calculate_priority_score(message)
        base = queue._calculate_priority_score(_message())
        assert score == pytest.approx(base - 0.5)  # capped age adjustment


class TestSpatialRoutingOptimizer:
    def test_analyze_empty_history(self) -> None:
        optimizer = SpatialRoutingOptimizer(_router())

        assert optimizer.analyze_routing_performance() == {
            "message": "No routing history available"
        }

    def test_analyze_suggests_optimizations(self) -> None:
        router = _router()
        router.record_routing_result("m1", ["a"], False)
        router.record_routing_result("m2", ["a"], True, latency=2000.0)
        optimizer = SpatialRoutingOptimizer(router)

        report = optimizer.analyze_routing_performance()
        assert report["total_routes_analyzed"] == 2
        assert report["success_rate"] == pytest.approx(50.0)
        assert report["average_latency"] == pytest.approx(1000.0)
        assert (
            "Consider increasing retry attempts for failed routes"
            in report["optimization_suggestions"]
        )
        assert (
            "Consider optimizing network paths for high-latency routes"
            in report["optimization_suggestions"]
        )

    def test_clean_history_yields_no_suggestions(self) -> None:
        router = _router()
        router.record_routing_result("m1", ["a"], True, latency=50.0)
        optimizer = SpatialRoutingOptimizer(router)

        assert optimizer.analyze_routing_performance()["optimization_suggestions"] == []
