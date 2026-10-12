"""Tiny, real DATA → SPACE → TIME and DATA → ASGI system workflows."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import http.server
import json
from pathlib import Path
import threading

import httpx
import numpy as np
import pandas as pd
import pytest

from geo_infer_api.app import create_app
from geo_infer_api.core.config import get_settings
from geo_infer_api.endpoints import geojson_router
from geo_infer_data.core.ingestion import (
    GenericDataSourceConnector,
    MultiSourceDataIngestion,
)
from geo_infer_data.core.storage import LocalFileBackend
from geo_infer_data.models.schemas import DataLineage, DatasetMetadata
from geo_infer_data.utils.timestamps import normalize_observation_timestamps
from geo_infer_space import (
    H3StateSpace,
    align_h3_observations,
    cell_to_latlng,
    latlng_to_cell,
)
from geo_infer_time import TimeSeries

pytestmark = pytest.mark.system


@contextmanager
def observation_endpoint(records: list[dict]):
    """Use actual loopback HTTP, with bounded lifecycle and no mocks."""
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            payload = (
                {"status": "ok"} if self.path == "/health" else {"records": records}
            )
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
    )
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        worker.join(timeout=2)
        server.server_close()
        assert not worker.is_alive(), "Local observation HTTP server did not stop"


@pytest.fixture
async def stored_observations(tmp_path: Path):
    """Ingest six identified rows and reopen actual Parquet bytes."""
    space = H3StateSpace(["8828308285fffff", "8828308281fffff", "8828308283fffff"])
    axis = pd.date_range("2026-01-01T00:00:00.000000123Z", periods=5, freq="30s")
    assignments = [
        ("neighbor-first", 0, 1, 4.0),
        ("observed-zero", 1, 0, 0.0),
        ("center-first", 1, 1, 2.0),
        ("neighbor-second", 0, 3, 8.0),
        ("center-second", 1, 3, 6.0),
        ("center-last", 1, 4, 10.0),
    ]
    records = []
    for identity, state, instant, value in assignments:
        latitude, longitude = cell_to_latlng(space.cells[state])
        records.append(
            {
                "record_id": identity,
                "cell": space.cells[state],
                "timestamp": axis[instant]
                .tz_convert("America/Los_Angeles")
                .isoformat(),
                "value": value,
                "latitude": latitude,
                "longitude": longitude,
            }
        )
    with observation_endpoint(records) as (endpoint, requests):
        ingestion = MultiSourceDataIngestion(["government"], validation_enabled=False)
        ingestion.connectors["government"] = GenericDataSourceConnector(
            {"base_url": endpoint}
        )
        result = await asyncio.wait_for(
            ingestion.ingest_multi_source(government={}), timeout=10
        )
        assert result["ingested_data"]["government"] == {"records": records}
        assert result["ingestion_metadata"]["sources_processed"] == 1
        # Automatic format detection annotates the empty request's tabular
        # query; the generic connector still parses the real JSON response.
        assert requests == ["/health", "/?format=csv"]

    frame = pd.DataFrame(result["ingested_data"]["government"]["records"])
    frame["timestamp"] = normalize_observation_timestamps(frame["timestamp"])
    frame = frame.set_index("record_id")
    frame.attrs = {
        "source": "local-http",
        "measurement_unit": "fixture-units",
        "provenance": {"instruments": ["fixture-sensor"]},
    }
    snapshot = frame.copy(deep=True)
    metadata = DatasetMetadata(
        title="Six observed system rows",
        lineage=DataLineage(
            source="loopback-http",
            process="parquet-roundtrip",
            created_by="system-test",
        ),
    )
    directory = tmp_path / "storage"
    writer = LocalFileBackend({"base_path": str(directory)})
    identifier = await asyncio.wait_for(writer.store(frame, metadata), timeout=10)
    parquet_files = list(directory.rglob("*.parquet"))
    assert len(parquet_files) == 1 and parquet_files[0].stat().st_size > 0
    assert not list(directory.rglob("*.pkl")), "This fixture requires actual Parquet"
    reader = LocalFileBackend({"base_path": str(directory)})
    reopened = await asyncio.wait_for(reader.retrieve(identifier, {}), timeout=10)
    pd.testing.assert_frame_equal(reopened, snapshot)
    pd.testing.assert_frame_equal(frame, snapshot)
    assert reopened.attrs == snapshot.attrs
    assert reopened.index.tolist() == [row[0] for row in assignments]
    assert reopened["timestamp"].array.asi8.tolist() == [
        1767225600000000123 + 30_000_000_000 * row[2] for row in assignments
    ]
    return space, axis, reopened


@pytest.mark.asyncio
async def test_e2e_data_to_space(stored_observations):
    """Preserve identities, UTC nanoseconds and coordinates through alignment."""
    space, axis, frame = stored_observations
    snapshot = frame.copy(deep=True)
    for row in frame.itertuples():
        assert latlng_to_cell(row.latitude, row.longitude, 8) == row.cell
        assert space.cells[space.locate(row.latitude, row.longitude)] == row.cell
    aligned = align_h3_observations(frame, state_space=space, timestamps=axis)
    assert isinstance(aligned, TimeSeries)
    assert aligned.data.columns.tolist() == list(space.cells)
    np.testing.assert_array_equal(aligned.timestamps.asi8, axis.asi8)
    np.testing.assert_allclose(
        aligned.data.to_numpy(),
        [
            [np.nan, 0, np.nan],
            [4, 2, np.nan],
            [np.nan, np.nan, np.nan],
            [8, 6, np.nan],
            [np.nan, 10, np.nan],
        ],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert aligned.metadata == {
        **frame.attrs,
        "spatial_index": "h3",
        "crs": "EPSG:4326",
    }
    aligned.data.iloc[0, 1] = 999
    aligned.metadata["provenance"]["instruments"].append("output-only-change")
    pd.testing.assert_frame_equal(frame, snapshot)
    assert frame.attrs == snapshot.attrs


@pytest.mark.asyncio
async def test_e2e_space_to_time(stored_observations):
    """Verify actual TIME aggregation, missing values and equivalent UTC bins."""
    space, axis, frame = stored_observations
    series = align_h3_observations(frame, state_space=space, timestamps=axis)
    alternative = align_h3_observations(
        frame.iloc[::-1],
        state_space=space,
        timestamps=axis.tz_convert("America/Los_Angeles"),
    )
    pd.testing.assert_frame_equal(series.data, alternative.data)
    snapshot = series.data.copy(deep=True)
    means = series.resample("1min", method="mean")
    sums = series.resample("1min", method="sum")
    # First center mean is (0 + 2)/2; absent observations are not zeros.
    np.testing.assert_allclose(
        means.data.to_numpy(),
        [
            [4, 1, np.nan],
            [8, 6, np.nan],
            [np.nan, 10, np.nan],
        ],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        sums.data.to_numpy(),
        [
            [4, 2, np.nan],
            [8, 6, np.nan],
            [np.nan, 10, np.nan],
        ],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert str(means.timestamps.tz) == "UTC"
    assert means.timestamps.asi8.tolist() == [
        1767225600000000000,
        1767225660000000000,
        1767225720000000000,
    ]
    assert means.data.columns.tolist() == list(space.cells)
    assert means.metadata["spatial_index"] == "h3"
    assert means.metadata["crs"] == "EPSG:4326"
    for aggregate in (means, sums):
        assert aggregate.metadata["source"] == "local-http"
        assert aggregate.metadata["measurement_unit"] == "fixture-units"
        assert aggregate.metadata["provenance"] == frame.attrs["provenance"]
    means.metadata["provenance"]["instruments"].append("aggregate-only-change")
    assert series.metadata["provenance"] == frame.attrs["provenance"]
    assert sums.metadata["provenance"] == frame.attrs["provenance"]
    pd.testing.assert_frame_equal(series.data, snapshot)


@pytest.mark.asyncio
async def test_e2e_data_to_api(stored_observations, monkeypatch):
    """Send stored observations through the actual public ASGI polygon API."""
    space, _, frame = stored_observations
    monkeypatch.setenv("SECRET_KEY", "public-system-fixture-signing-key")
    get_settings.cache_clear()
    # Isolate the actual process-local demo store; endpoint behavior is real.
    monkeypatch.setattr(geojson_router, "POLYGON_FEATURES", {})
    app = create_app()
    endpoint = "/api/v1/collections/polygons/items"
    transport = httpx.ASGITransport(app=app)
    try:
        # ASGITransport runs in process and does not enforce HTTPX's network
        # timeouts; bound the awaited endpoint workflow explicitly as well.
        async with (
            asyncio.timeout(10),
            httpx.AsyncClient(
                transport=transport, base_url="http://system.local", timeout=5
            ) as client,
        ):
            submitted = {}
            for cell in space.cells[:2]:
                latitude, longitude = cell_to_latlng(cell)
                ring = [
                    [longitude - 0.0001, latitude - 0.0001],
                    [longitude + 0.0001, latitude - 0.0001],
                    [longitude + 0.0001, latitude + 0.0001],
                    [longitude - 0.0001, latitude + 0.0001],
                    [longitude - 0.0001, latitude - 0.0001],
                ]
                selected = frame[frame["cell"] == cell]
                feature = {
                    "type": "Feature",
                    "id": cell,
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "cell": cell,
                        "unit": frame.attrs["measurement_unit"],
                        "observations": [
                            {
                                "id": row.Index,
                                "timestamp": row.timestamp.isoformat(),
                                "value": row.value,
                            }
                            for row in selected.itertuples()
                        ],
                    },
                }
                response = await client.post(endpoint, json=feature)
                assert response.status_code == 201, response.text
                assert response.json() == feature
                submitted[cell] = feature
            center = space.cells[1]
            response = await client.get(f"{endpoint}/{center}")
            assert response.status_code == 200
            assert response.json() == submitted[center]
            observations = response.json()["properties"]["observations"]
            assert observations[0] == {
                "id": "observed-zero",
                "timestamp": "2026-01-01T00:00:00.000000123+00:00",
                "value": 0.0,
            }
            assert [row["value"] for row in observations] == [0, 2, 6, 10]
            assert len(observations) == 4  # Missing interval is not fabricated.
            collection = await client.get(endpoint)
            assert collection.status_code == 200
            assert {f["id"] for f in collection.json()["features"]} == set(submitted)
            latitude, longitude = cell_to_latlng(center)
            bbox = f"{longitude - 0.0002},{latitude - 0.0002},{longitude + 0.0002},{latitude + 0.0002}"
            filtered = await client.get(endpoint, params={"bbox": bbox})
            assert filtered.status_code == 200
            assert filtered.json()["features"] == [submitted[center]]
            duplicate = await client.post(endpoint, json=submitted[center])
            assert duplicate.status_code == 409
            malformed = await client.get(endpoint, params={"bbox": "nan,0,1,1"})
            assert malformed.status_code == 400
            deleted = await client.delete(f"{endpoint}/{center}")
            assert deleted.status_code == 204
            missing = await client.get(f"{endpoint}/{center}")
            assert missing.status_code == 404
            remaining = await client.get(endpoint)
            assert [f["id"] for f in remaining.json()["features"]] == [space.cells[0]]
    finally:
        get_settings.cache_clear()
