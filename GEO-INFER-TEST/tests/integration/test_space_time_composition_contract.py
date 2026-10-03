"""Small real DATA -> SPACE -> TIME -> ACT composition with analytic oracles."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, UTC
import hashlib
import http.server
import json
from pathlib import Path
import threading

import h3
import numpy as np
import pandas as pd
import pytest

from geo_infer_act.core.gnn_contract import GNNArtifact, run_gnn_inference
from geo_infer_data.core.ingestion import (
    GenericDataSourceConnector,
    MultiSourceDataIngestion,
)
from geo_infer_data.core.storage import LocalFileBackend
from geo_infer_data.models.schemas import DataLineage, DatasetMetadata
from geo_infer_space import H3StateSpace, align_h3_observations
from geo_infer_time import normalize_timestamp


@contextmanager
def observation_endpoint(records: list[dict]):
    """Serve real HTTP locally; no connector or inference behavior is mocked."""
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            payload = (
                {"status": "ok"} if self.path == "/health" else {"records": records}
            )
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


def spatial_fixture():
    center = h3.latlng_to_cell(41.75, -124.2, 8)
    neighbor = sorted(set(h3.grid_disk(center, 1)) - {center})[0]
    space = H3StateSpace([neighbor, center])
    axis = ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"]
    records = [
        {"cell": center, "timestamp": "2025-12-31T16:01:00-08:00", "value": 6.0},
        {"cell": neighbor, "timestamp": axis[0], "value": 0.0},
        {"cell": center, "timestamp": axis[0], "value": 3.0},
        {"cell": neighbor, "timestamp": axis[1], "value": 4.0},
    ]
    return space, axis, records


def test_data_transport_storage_alignment_and_actual_act_trace(tmp_path: Path) -> None:
    """Round-trip timestamps/values and verify real posterior/action propagation."""
    space, axis, records = spatial_fixture()
    with observation_endpoint(records) as (endpoint, requests):
        ingestion = MultiSourceDataIngestion(["government"], validation_enabled=False)
        ingestion.connectors["government"] = GenericDataSourceConnector(
            {"base_url": endpoint}
        )
        report = asyncio.run(ingestion.ingest_multi_source(government={}))
        fetched = report["ingested_data"]["government"]
        assert fetched == {"records": records}
        assert report["ingestion_metadata"]["sources_processed"] == 1
        assert "/health" in requests and len(requests) == 2

    frame = pd.DataFrame(fetched["records"])
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"].map(normalize_timestamp), utc=True
    )
    frame.attrs = {
        "source": "localhost-fixture",
        "measurement_unit": "analytical-units",
        "lineage": {"record_ids": ["r3", "r0", "r2", "r1"]},
    }
    metadata = DatasetMetadata(
        title="Local composition fixture",
        lineage=DataLineage(
            source="localhost-fixture",
            process="HTTP ingestion and local parquet",
            created_by="contract-test",
        ),
    )
    writer = LocalFileBackend({"base_path": str(tmp_path / "storage")})
    identifier = asyncio.run(writer.store(frame, metadata))
    # A fresh backend must read the bytes; no in-process storage shortcut.
    reader = LocalFileBackend({"base_path": str(tmp_path / "storage")})
    retrieved = asyncio.run(reader.retrieve(identifier, {}))
    pd.testing.assert_frame_equal(retrieved, frame)
    assert retrieved.attrs == frame.attrs
    with pytest.raises(FileNotFoundError):
        asyncio.run(reader.retrieve("missing-dataset", {}))

    series = align_h3_observations(retrieved, state_space=space, timestamps=axis)
    assert list(series.data.columns) == list(space.cells)
    assert str(series.timestamps.tz) == "UTC"
    np.testing.assert_array_equal(series.data.to_numpy(), [[0.0, 3.0], [4.0, 6.0]])
    assert series.start_time == pd.Timestamp(axis[0])
    assert series.duration.total_seconds() == 60
    assert series.metadata == {
        **frame.attrs,
        "spatial_index": "h3",
        "crs": "EPSG:4326",
    }
    aggregated = series.resample("2min", method="mean")
    np.testing.assert_array_equal(aggregated.data.to_numpy(), [[2.0, 4.5]])
    assert aggregated.metadata["lineage"] == frame.attrs["lineage"]
    aggregated.metadata["lineage"]["record_ids"][0] = "changed"
    assert series.metadata["lineage"]["record_ids"][0] == "r3"
    series.metadata["lineage"]["record_ids"][0] = "changed-series"
    assert retrieved.attrs["lineage"]["record_ids"][0] == "r3"
    assert frame.attrs["lineage"]["record_ids"][0] == "r3"

    transition = space.dense_transition_tensor()
    artifact = GNNArtifact.from_dict(
        {
            "schema_version": "gnn-geo-infer/1",
            "model_type": "categorical",
            "model_name": "stored H3 observations",
            "dimensions": {"states": 2, "observations": 2, "actions": 2},
            "matrices": {
                "A": [[0.9, 0.2], [0.1, 0.8]],
                "B": transition.tolist(),
                "C": [0.0, 2.0],
                "D": [0.99, 0.01],
                "E": [0.5, 0.5],
            },
            "space": {"kind": "h3", "state_ids": list(space.cells)},
            "time": {"step_seconds": 60},
            "provenance": {
                "producer": "geo-infer-test",
                "source_sha256": hashlib.sha256(
                    json.dumps(records).encode()
                ).hexdigest(),
            },
        }
    )
    observations = [
        {"timestamp": timestamp.isoformat(), "observation": int(np.argmax(row))}
        for timestamp, row in series.data.iterrows()
    ]
    trace = run_gnn_inference(artifact, observations, random_seed=42)
    assert trace == run_gnn_inference(artifact, observations, random_seed=42)
    prior = np.array([0.99, 0.01])
    for observation, step in zip(observations, trace["steps"]):
        likelihood = np.array([[0.9, 0.2], [0.1, 0.8]])[observation["observation"]]
        posterior = prior * likelihood / (prior @ likelihood)
        np.testing.assert_allclose(step["posterior"], posterior, atol=1e-12, rtol=0)
        policy = np.asarray(step["policy_posterior"])
        assert np.all(np.isfinite(policy)) and np.all(policy >= 0)
        np.testing.assert_allclose(policy.sum(), 1, atol=1e-12, rtol=0)
        assert step["action"] == int(np.argmax(policy))
        prior = transition[:, :, step["action"]] @ posterior
        np.testing.assert_allclose(step["next_prior"], prior, atol=1e-12, rtol=0)
    assert [step["action"] for step in trace["steps"]] == [1, 0]
    assert not np.allclose(
        trace["steps"][0]["next_prior"], trace["steps"][0]["posterior"]
    )


def test_alignment_is_row_order_invariant_and_state_order_equivariant() -> None:
    space, axis, records = spatial_fixture()
    frame = pd.DataFrame(records)
    baseline = align_h3_observations(frame, state_space=space, timestamps=axis)
    reordered = align_h3_observations(
        frame.iloc[::-1], state_space=space, timestamps=axis
    )
    pd.testing.assert_frame_equal(baseline.data, reordered.data)
    reversed_space = H3StateSpace(list(reversed(space.cells)))
    permuted = align_h3_observations(frame, state_space=reversed_space, timestamps=axis)
    pd.testing.assert_frame_equal(permuted.data, baseline.data.iloc[:, ::-1])


def test_missing_measurement_stays_missing_and_observed_zero_stays_zero() -> None:
    space, axis, records = spatial_fixture()
    aligned = align_h3_observations(
        pd.DataFrame(records[:-1]), state_space=space, timestamps=axis
    )
    assert aligned.data.iloc[0, 0] == 0.0
    assert np.isnan(aligned.data.iloc[1, 0])


@pytest.mark.parametrize(
    "fault", ["naive", "duplicate", "unknown_cell", "unknown_time", "nan"]
)
def test_composition_rejects_invalid_observations(fault: str) -> None:
    space, axis, records = spatial_fixture()
    if fault == "naive":
        records[0]["timestamp"] = datetime(2026, 1, 1)
    elif fault == "duplicate":
        records.append({**records[0], "timestamp": "2026-01-01T00:01:00Z"})
    elif fault == "unknown_cell":
        records[0]["cell"] = h3.latlng_to_cell(0, 0, 8)
    elif fault == "unknown_time":
        records[0]["timestamp"] = datetime(2026, 1, 2, tzinfo=UTC)
    else:
        records[0]["value"] = np.nan
    with pytest.raises((ValueError, TypeError)):
        align_h3_observations(pd.DataFrame(records), state_space=space, timestamps=axis)
