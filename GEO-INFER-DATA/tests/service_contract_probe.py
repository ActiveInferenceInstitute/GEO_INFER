"""Fail-closed real service checks against disposable loopback CI services.

Run explicitly; this is not an absent-service skip or a default pytest fixture.
Uses only isolated test credentials and never the AWS default credential chain.
"""

import asyncio
from datetime import datetime, UTC
from io import BytesIO
import json
import hashlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path
import tempfile
import uuid

import geopandas as gpd
import pandas as pd
from minio import Minio
from shapely.geometry import Point

from geo_infer_data.connectors.cloud import S3Connector
from geo_infer_data.core.storage import PostgreSQLBackend, MinIOBackend, RedisBackend
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
)
from geo_infer_data.utils.secure_serialization import PayloadSecurityError


def metadata():
    return DatasetMetadata(
        title="Disposable service acceptance",
        spatial=SpatialExtent(bbox=[-123, 37, -122, 38]),
        temporal=TemporalExtent(
            start=datetime(2023, 1, 1, tzinfo=UTC), end=datetime(2023, 1, 2, tzinfo=UTC)
        ),
        lineage=DataLineage(
            source="CI fixture", process="service roundtrip", created_by="CI"
        ),
    )


async def postgres_contract():
    backend = PostgreSQLBackend(
        {
            "host": "127.0.0.1",
            "port": 15432,
            "user": "geo_test",
            "password": "disposable-test-only",
            "database": "geo_test",
        }
    )
    identifiers = []
    try:
        frames = [
            pd.DataFrame({"value": [1, 2]}),
            gpd.GeoDataFrame(
                {"value": [3, 9]},
                geometry=[Point(-122.42, 37.77), Point(-122.41, 37.78)],
                crs="EPSG:4326",
            ),
            gpd.GeoDataFrame(
                {"value": [3, 9]},
                geometry=[Point(-122.42, 37.77), Point(-122.41, 37.78)],
                crs="EPSG:4326",
            )
            .to_crs(3857)
            .rename_geometry("location"),
        ]
        frames += [frames[-1].rename_geometry(name) for name in ("Location", "order")]
        from sqlalchemy import create_engine, text

        engine = create_engine(backend.connection_string)
        try:
            with engine.connect() as connection:
                versions = {
                    "postgresql": connection.execute(
                        text("SELECT version()")
                    ).scalar_one(),
                    "postgis": connection.execute(
                        text("SELECT PostGIS_Full_Version()")
                    ).scalar_one(),
                }
        finally:
            engine.dispose()
        for frame in frames:
            data_id = await backend.store(frame, metadata())
            identifiers.append(data_id)
            retrieved = await backend.retrieve(data_id, {})
            if isinstance(frame, gpd.GeoDataFrame):
                from geopandas.testing import assert_geodataframe_equal

                assert_geodataframe_equal(retrieved, frame)
                selected = await backend.retrieve(
                    data_id, {"spatial": [-122.43, 37.76, -122.415, 37.775]}
                )
                assert selected["value"].tolist() == [3]
            else:
                pd.testing.assert_frame_equal(retrieved, frame)
        for value in [{"value": 7}, None, [1, "two"]]:
            data_id = await backend.store(value, metadata())
            identifiers.append(data_id)
            assert await backend.retrieve(data_id, {}) == value
        assert len(set(identifiers)) == len(identifiers)
        for data_id in identifiers:
            assert await backend.delete(data_id)
            assert not await backend.delete(data_id)
            try:
                await backend.retrieve(data_id, {})
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Deleted PostgreSQL value was still readable")
    finally:
        for data_id in identifiers:
            await backend.delete(data_id)
    return {
        "status": "passed",
        "objects": len(identifiers),
        "versions": versions,
        "operations": "tabular/geospatial/projected-custom-geometry/generic/read/spatial/delete",
    }


async def redis_contract():
    backend = RedisBackend(
        {
            "host": "127.0.0.1",
            "port": 16379,
            "signing_key": b"disposable-integrity-test-key-32-bytes",
        }
    )
    data_id = None
    try:
        server_version = backend.client.info("server")["redis_version"]
        data_id = await backend.store({"value": 7}, metadata())
        assert await backend.retrieve(data_id, {}) == {"value": 7}
        raw = backend.client.get(data_id)
        backend.client.set(data_id, raw[:-1] + bytes([raw[-1] ^ 1]))
        try:
            await backend.retrieve(data_id, {})
        except PayloadSecurityError:
            pass
        else:
            raise AssertionError("Tampered Redis payload accepted")
        assert await backend.delete(data_id)
        assert not await backend.delete(data_id)
    finally:
        if data_id is not None:
            await backend.delete(data_id)
        backend.client.close()
    return {
        "status": "passed",
        "operations": "signed-store/read/tamper-rejection/delete",
        "server_version": server_version,
    }


async def object_contract():
    bucket = f"geo-service-{uuid.uuid4().hex}"
    config = {
        "endpoint": "127.0.0.1:19000",
        "access_key": "geo_test",
        "secret_key": "disposable-test-only",
        "bucket": bucket,
        "secure": False,
        "signing_key": b"disposable-integrity-test-key-32-bytes",
    }
    backend = MinIOBackend(config)
    client = Minio(
        config["endpoint"],
        access_key=config["access_key"],
        secret_key=config["secret_key"],
        secure=False,
    )
    try:
        data_id = await backend.store({"value": 7}, metadata())
        assert await backend.retrieve(data_id, {}) == {"value": 7}
        response = client.get_object(bucket, f"{data_id}.bin")
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
        damaged = raw[:-1] + bytes([raw[-1] ^ 1])
        client.put_object(bucket, f"{data_id}.bin", BytesIO(damaged), len(damaged))
        try:
            await backend.retrieve(data_id, {})
        except PayloadSecurityError:
            pass
        else:
            raise AssertionError("Tampered MinIO payload accepted")
        assert await backend.delete(data_id)
        assert not await backend.delete(data_id)
        connector = S3Connector(
            {
                "bucket": bucket,
                "endpoint_url": "http://127.0.0.1:19000",
                "access_key": "geo_test",
                "secret_key": "disposable-test-only",
            }
        )
        try:
            assert await connector.connect()
            with tempfile.TemporaryDirectory(prefix="geo-s3-roundtrip-") as directory:
                source, target = (
                    Path(directory) / "source.bin",
                    Path(directory) / "target.bin",
                )
                source.write_bytes(bytes(range(256)))
                assert (
                    await connector.upload_file(str(source), "roundtrip.bin")
                    == "roundtrip.bin"
                )
                assert "roundtrip.bin" in await connector.list_files()
                await connector.download_file("roundtrip.bin", str(target))
                assert target.read_bytes() == source.read_bytes()
                assert await connector.delete_file("roundtrip.bin")
                assert "roundtrip.bin" not in await connector.list_files()
        finally:
            if connector._client is not None:
                connector._client.close()
    finally:
        if client.bucket_exists(bucket):
            for item in client.list_objects(bucket, recursive=True):
                client.remove_object(bucket, item.object_name)
            client.remove_bucket(bucket)
    return {
        "status": "passed",
        "operations": "MinIO-signed-store/read/tamper-rejection/delete;S3-compatible-byte-upload/download/list/delete",
    }


def source_identity():
    root = Path(__file__).resolve().parents[2]

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", *arguments], cwd=root, text=True
        ).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "clean": not bool(git("status", "--porcelain")),
    }


async def main():
    results = {}
    receipt = {
        "schema": "geo-infer-service-contracts/v1",
        "source_before": source_identity(),
        "python": sys.version.split()[0],
        "clients": {
            name: importlib.metadata.version(name)
            for name in (
                "geopandas",
                "pandas",
                "geoalchemy2",
                "psycopg2-binary",
                "redis",
                "minio",
                "boto3",
            )
        },
        "services": results,
        "status": "running",
    }
    try:
        assert receipt["source_before"]["clean"], (
            "Service acceptance requires a clean source checkout"
        )
        for name, contract in [
            ("postgresql_postgis", postgres_contract),
            ("redis", redis_contract),
            ("minio_s3_compatible", object_contract),
        ]:
            try:
                results[name] = await contract()
            except Exception as error:
                results[name] = {"status": "failed", "error_type": type(error).__name__}
                raise
        receipt["status"] = "passed"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        raise
    finally:
        receipt["source_after"] = source_identity()
        if receipt["source_before"] != receipt["source_after"]:
            receipt["status"] = "failed"
            receipt["source_mutation"] = True
        print(json.dumps(receipt, indent=2))
    assert receipt["status"] == "passed", "Service acceptance source identity changed"


if __name__ == "__main__":
    asyncio.run(main())
