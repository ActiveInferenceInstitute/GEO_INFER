"""Unit tests for the public exports SensorCalibration, NetworkTopology, SensorAPI.

Covers SensorCalibration linear/polynomial fits and scipy drift significance,
NetworkTopology construction from Sensor models, and SensorAPI route
registration.
"""

import h3
import pytest
from fastapi.testclient import TestClient

from geo_infer_iot.api.sensor_api import SensorAPI
from geo_infer_iot.models.network import (
    NetworkLink,
    NetworkNode,
    NetworkTopology,
    NetworkTopologyType,
)
from geo_infer_iot.models.sensor import Location, Sensor
from geo_infer_iot.utils.calibration import SensorCalibration


def _reference_data() -> list[dict[str, float]]:
    """Reference pairs for the relation reference = 2 * sensor + 1."""
    return [
        {"sensor_value": value, "reference_value": 2.0 * value + 1.0}
        for value in (0.0, 5.0, 10.0, 15.0, 20.0)
    ]


def _drift_data() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    baseline = [{"value": value} for value in (20.0, 21.0, 19.0, 22.0, 20.5)]
    drifted = [{"value": value} for value in (27.0, 28.0, 25.5, 28.5, 26.5)]
    return baseline, drifted


def _sensor(sensor_id: str, latitude: float, longitude: float) -> Sensor:
    return Sensor(
        sensor_id=sensor_id,
        network_id="net-1",
        sensor_type="temperature",
        location=Location(latitude=latitude, longitude=longitude),
    )


class TestSensorCalibration:
    def test_linear_round_trip(self):
        calibrator = SensorCalibration()

        result = calibrator.calibrate_sensor("s1", _reference_data(), "linear")
        assert result.success is True
        assert result.calibration_parameters["slope"] == pytest.approx(2.0, abs=1e-9)
        assert result.calibration_parameters["offset"] == pytest.approx(1.0, abs=1e-9)
        assert result.calibration_error < 1e-6
        assert len(calibrator.calibration_history) == 1

        calibrated = calibrator._apply_calibration(
            [4.0], result.calibration_parameters, "linear"
        )
        assert calibrated[0] == pytest.approx(9.0, abs=1e-9)

    def test_polynomial_calibration_recovers_quadratic(self):
        calibrator = SensorCalibration()
        reference = [
            {"sensor_value": value, "reference_value": 0.5 * value**2 + 1.0}
            for value in (0.0, 1.0, 2.0, 3.0, 4.0)
        ]

        result = calibrator.calibrate_sensor("s1", reference, "polynomial")
        assert result.success is True
        assert result.calibration_error < 1e-6

    def test_drift_detection_reports_significance(self):
        calibrator = SensorCalibration()
        baseline, drifted = _drift_data()

        report = calibrator.detect_drift("s1", drifted, baseline)
        assert report["drift_detected"] is True
        assert report["drift_score"] > report["threshold"]
        assert report["confidence"] > 0.9

    def test_drift_detection_stable_sensor(self):
        calibrator = SensorCalibration()
        baseline, _ = _drift_data()

        report = calibrator.detect_drift("s1", [dict(m) for m in baseline], baseline)
        assert report["drift_detected"] is False

    def test_drift_detection_insufficient_data(self):
        calibrator = SensorCalibration()

        report = calibrator.detect_drift(
            "s1", [{"value": 1.0}] * 3, [{"value": 1.0}] * 5
        )
        assert report["drift_detected"] is False
        assert report["drift_score"] == 0.0


class TestNetworkTopology:
    def test_construction_from_sensor_models(self):
        sensors = [
            _sensor("s1", 40.0, -74.0),
            _sensor("s2", 40.01, -74.01),
            _sensor("s3", 40.02, -74.02),
        ]
        nodes = {
            sensor.sensor_id: NetworkNode(
                node_id=sensor.sensor_id,
                node_type="sensor",
                sensor_id=sensor.sensor_id,
                latitude=sensor.location.latitude,
                longitude=sensor.location.longitude,
            )
            for sensor in sensors
        }
        link = NetworkLink(
            link_id="l1", source_node="s1", target_node="s2", latency_ms=12.0
        )

        topology = NetworkTopology(
            topology_id="topo-1",
            name="Lab sensor mesh",
            topology_type=NetworkTopologyType.MESH,
            nodes=nodes,
            links={"l1": link},
        )

        assert topology.total_nodes == 3
        assert topology.active_nodes == 3
        assert topology.total_links == 1
        assert topology.active_links == 1
        assert topology.average_latency_ms == pytest.approx(12.0)
        assert topology.nodes["s1"].h3_index == h3.latlng_to_cell(40.0, -74.0, 8)
        assert topology.nodes["s2"].h3_index != topology.nodes["s1"].h3_index
        assert topology.nodes["s1"].get_health_score() == 1.0


class TestSensorAPI:
    def test_route_registration(self):
        api = SensorAPI({})
        paths = {getattr(route, "path", "") for route in api.app.routes}
        for expected in (
            "/",
            "/sensors",
            "/sensors/{sensor_id}",
            "/measurements",
            "/networks",
            "/health",
            "/spatial/{h3_index}/sensors",
        ):
            assert expected in paths
        assert api.get_app() is api.app

    def test_routes_served(self):
        api = SensorAPI({})
        client = TestClient(api.get_app())

        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["service"] == "GEO-INFER-IOT Sensor API"
        assert client.get("/health").status_code == 200

        sensors = client.get("/sensors")
        assert sensors.status_code == 200
        assert sensors.json()["sensors"] == []
