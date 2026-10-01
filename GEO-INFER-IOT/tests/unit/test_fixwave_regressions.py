"""Regression tests for the fix-wave: import wiring, async processing,
batch filtering, config key paths, candidate derivation, meter distances,
and streaming API catalog honesty."""

import asyncio
from datetime import datetime, UTC

import h3
import numpy as np
import pytest

from geo_infer_iot import (
    AdaptiveSampling,
    BayesianInferenceAPI,
    IoTSystem,
    Measurement,
    MeasurementBatch,
    SpatialInterpolation,
    StreamingAPI,
)
from geo_infer_iot.core.quality_control import QualityController


def _measurement(sensor_id: str, score: float) -> Measurement:
    return Measurement(
        measurement_id=f"m-{sensor_id}",
        sensor_id=sensor_id,
        variable="temperature",
        value=20.0,
        unit="celsius",
        latitude=40.0,
        longitude=-74.0,
        quality={"quality_score": score},
    )


class TestInferenceApiWiring:
    def test_engine_constructed_not_silently_disabled(self):
        """The Bayesian engine must be available in a healthy install."""
        api = BayesianInferenceAPI()
        assert api.inference_engine is not None

    def test_import_location_is_core_inference(self):
        from geo_infer_iot.core.inference import BayesianSpatialInference

        engine = BayesianSpatialInference(
            variable="temperature", spatial_resolution=8, temporal_window="1h"
        )
        assert engine.variable == "temperature"


class TestIoTSystemProcessing:
    def test_start_processing_requires_running_loop(self):
        system = IoTSystem({})
        result = system.start_processing()
        assert result["success"] is False
        assert "event loop" in result["error"]

    def test_start_and_stop_processing_schedule_tasks(self):
        system = IoTSystem({})

        async def run():
            started = system.start_processing()
            assert started["success"] is True
            assert system.is_processing is True
            assert len(system._processing_tasks) == 1
            # Let the scheduled coroutine actually start
            await asyncio.sleep(0)
            assert system.ingestion.is_processing is True

            stopped = system.stop_processing()
            assert stopped["success"] is True
            assert system.is_processing is False
            await asyncio.sleep(0.1)
            assert system.ingestion.is_processing is False

        asyncio.run(run())


class TestMeasurementBatchFiltering:
    def test_all_invalid_batch_returns_empty_batch(self):
        batch = MeasurementBatch(
            batch_id="b1",
            measurements=[_measurement("s1", 0.2), _measurement("s2", 0.3)],
            batch_size=2,
        )
        filtered = batch.filter_by_quality(min_quality_score=0.7)
        assert filtered.batch_size == 0
        assert filtered.measurements == []

    def test_empty_construction_outside_filter_still_rejected(self):
        with pytest.raises(ValueError):
            MeasurementBatch(batch_id="b2", measurements=[], batch_size=0)

    def test_filter_by_variable_can_return_empty(self):
        batch = MeasurementBatch(
            batch_id="b3", measurements=[_measurement("s1", 1.0)], batch_size=1
        )
        filtered = batch.filter_by_variable("humidity")
        assert filtered.batch_size == 0


class TestQualityControlConfigPath:
    def test_nested_temporal_consistency_config_is_honored(self):
        controller = QualityController(
            {"temporal_consistency": {"max_change_rate": 0.01}}
        )
        now = datetime.now(UTC).isoformat()

        # Retain-after-scoring means the rate is computed among stored
        # history; feed a stable pair followed by a jump so the stored
        # history itself contains the excessive rate.
        for value in (10.0, 10.05, 20.0):
            controller.validate_measurement(
                {"sensor_id": "s1", "value": value, "timestamp": now}
            )
        result = controller.validate_measurement(
            {"sensor_id": "s1", "value": 20.0, "timestamp": now}
        )
        assert not result.passed
        assert any("Temporal change rate" in issue for issue in result.issues)


class TestAdaptiveSamplingCandidates:
    def test_candidates_derived_from_uncovered_priority_cells(self):
        sampler = AdaptiveSampling()
        covered = h3.latlng_to_cell(37.00, -122.00, 8)
        gap = h3.latlng_to_cell(37.10, -122.10, 8)
        result = sampler.suggest_locations(
            current_network=[{"sensor_id": "a", "h3_index": covered}],
            priority_areas=[covered, gap],
        )
        assert result["success"] is True
        assert result["analysis"]["coverage_gaps"] == 1
        recs = result["recommendations"]
        assert len(recs) == 1
        expected_lat, expected_lon = h3.cell_to_latlng(gap)
        assert recs[0]["latitude"] == pytest.approx(expected_lat)
        assert recs[0]["longitude"] == pytest.approx(expected_lon)
        # Deterministic cost from config, not random
        assert recs[0]["estimated_cost"] == 500.0


class TestInterpolationUnits:
    def _sensors(self):
        return [
            {"latitude": 40.000, "longitude": -74.000, "value": 10.0},
            {"latitude": 40.001, "longitude": -74.000, "value": 11.0},
            {"latitude": 40.000, "longitude": -74.001, "value": 12.0},
        ]

    def test_max_distance_filter_operates_in_meters(self):
        interp = SpatialInterpolation({"max_distance": 10000})
        # ~50 km away target: beyond the 10 km meter threshold
        far = interp.interpolate_to_grid(self._sensors(), [(40.45, -74.0)])
        assert np.isnan(far["interpolated_values"][0])

        # ~75 m away target: within threshold
        near = interp.interpolate_to_grid(self._sensors(), [(40.0005, -74.0005)])
        assert not np.isnan(near["interpolated_values"][0])

    def test_uncertainty_grows_with_real_distance(self):
        interp = SpatialInterpolation()
        near = interp.interpolate_to_grid(self._sensors(), [(40.0005, -74.0005)])
        far = interp.interpolate_to_grid(
            self._sensors(), [(40.2, -74.0)], method="nearest_neighbor"
        )
        near_unc = near["uncertainty"][0]
        far_unc = far["uncertainty"][0]
        assert far_unc > near_unc
        # Distance-based component is meaningful, not ~0 by construction
        assert near_unc > 0.1


class TestStreamingApi:
    def test_catalog_only_advertises_registered_routes(self):
        api = StreamingAPI({})
        paths = set(api.app.routes and {getattr(r, "path", "") for r in api.app.routes})
        assert "/ws/sensor-stream" in paths
        assert "/ws/spatial-stream" not in paths
        assert "/ws/anomaly-stream" not in paths

    def test_shared_ingestion_instance_is_used(self):
        from geo_infer_iot import IoTDataIngestion

        shared = IoTDataIngestion(None, {})
        api = StreamingAPI({}, ingestion=shared)
        assert api.ingestion is shared

    def test_broadcast_removes_disconnected_clients(self):
        api = StreamingAPI({})

        class FakeWebSocket:
            def __init__(self, fail: bool) -> None:
                self.fail = fail
                self.sent: list[str] = []

            async def send_text(self, text: str) -> None:
                if self.fail:
                    raise RuntimeError("closed")
                self.sent.append(text)

        good, bad = FakeWebSocket(False), FakeWebSocket(True)
        api.sensor_subscriptions["s1"] = {good, bad}
        asyncio.run(api.broadcast_measurement({"sensor_id": "s1"}))
        assert "s1" in api.sensor_subscriptions
        assert bad not in api.sensor_subscriptions["s1"]
        assert good in api.sensor_subscriptions["s1"]
        assert len(good.sent) == 1


class TestDuplicateNames:
    def test_registry_record_is_renamed(self):
        from geo_infer_iot.core import registry as registry_module

        assert not hasattr(registry_module, "SensorNetwork")
        assert hasattr(registry_module, "SensorNetworkRecord")

    def test_package_sensor_network_is_the_pydantic_model(self):
        from geo_infer_iot.models.sensor import SensorNetwork as PydanticNetwork

        network = PydanticNetwork(
            network_id="n1",
            name="N",
            protocol="MQTT",
            spatial_bounds={
                "lat_min": 0.0,
                "lat_max": 1.0,
                "lon_min": 0.0,
                "lon_max": 1.0,
            },
            sensor_types=["temperature"],
        )
        assert network.get_coverage_area() > 0

    def test_ingestion_monitor_renamed(self):
        from geo_infer_iot.core import ingestion as ingestion_module

        assert not hasattr(ingestion_module, "GlobalMonitoringSystem")
        assert hasattr(ingestion_module, "GlobalRadiationMonitor")
