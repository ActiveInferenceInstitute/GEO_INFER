"""
Tests for cloud storage connectors in geo_infer_data.connectors.cloud.
"""

import asyncio

import pytest

from geo_infer_data.connectors.cloud import (
    CloudConnector,
    NotConnectedError,
    S3Connector,
    GCSConnector,
    AzureConnector,
)


def _run(coro):
    return asyncio.run(coro)


class _FakeS3Client:
    """In-memory boto3 client double recording calls and serving objects."""

    def __init__(self):
        self.objects: dict = {}
        self.buckets: set = {"geo-infer-data", "my-bucket"}
        self.calls: list = []

    def head_bucket(self, Bucket):
        self.calls.append(("head_bucket", Bucket))
        if Bucket not in self.buckets:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadBucket",
            )

    def upload_file(self, Filename, Bucket, Key):
        self.calls.append(("upload_file", Bucket, Key))
        with open(Filename, "rb") as f:
            self.objects[Key] = f.read()

    def download_file(self, Bucket, Key, Filename):
        self.calls.append(("download_file", Bucket, Key))
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject"
            )
        with open(Filename, "wb") as f:
            f.write(self.objects[Key])

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"

        class _Paginator:
            def __init__(self, client):
                self.client = client

            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in self.client.objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]}

        return _Paginator(self)

    def delete_object(self, Bucket, Key):
        self.calls.append(("delete_object", Bucket, Key))
        self.objects.pop(Key, None)


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeS3Client()
    monkeypatch.setattr(S3Connector, "_create_client", lambda self: client)
    return client


# ---------------------------------------------------------------------------
# CloudConnector base class
# ---------------------------------------------------------------------------


class TestCloudConnectorBase:
    def test_connect_raises(self):
        connector = CloudConnector()
        with pytest.raises(RuntimeError):
            _run(connector.connect())

    def test_upload_file_raises(self):
        connector = CloudConnector()
        with pytest.raises(RuntimeError):
            _run(connector.upload_file("/local", "/remote"))

    def test_download_file_raises(self):
        connector = CloudConnector()
        with pytest.raises(RuntimeError):
            _run(connector.download_file("/remote", "/local"))

    def test_list_files_raises(self):
        connector = CloudConnector()
        with pytest.raises(RuntimeError):
            _run(connector.list_files())

    def test_delete_file_raises(self):
        connector = CloudConnector()
        with pytest.raises(RuntimeError):
            _run(connector.delete_file("/remote"))

    def test_disconnect_does_not_raise(self):
        connector = CloudConnector()
        _run(connector.disconnect())

    def test_read_byte_range_local_file(self, tmp_path):
        connector = S3Connector({})
        file_path = tmp_path / "test.geoparquet"
        file_path.write_bytes(b"PAR1_GEOPARQUET_MAGIC_BYTES_FOOTER_PAR1")
        data = _run(connector.read_byte_range(str(file_path), start_byte=0, end_byte=3))
        assert data == b"PAR1"
        data_footer = _run(
            connector.read_byte_range(str(file_path), start_byte=35, end_byte=38)
        )
        assert data_footer == b"PAR1"

        with pytest.raises(ValueError):
            _run(connector.read_byte_range(str(file_path), start_byte=-1, end_byte=5))


# ---------------------------------------------------------------------------
# S3Connector
# ---------------------------------------------------------------------------


class TestS3Connector:
    def test_init_defaults(self):
        connector = S3Connector({})
        assert connector.bucket == "geo-infer-data"
        assert connector.region == "us-east-1"

    def test_init_custom(self):
        config = {"bucket": "my-bucket", "region": "eu-west-1"}
        connector = S3Connector(config)
        assert connector.bucket == "my-bucket"
        assert connector.region == "eu-west-1"

    def test_init_without_connection(self):
        connector = S3Connector({})
        with pytest.raises(NotConnectedError):
            _run(connector.list_files())

    def test_connect_verifies_bucket(self, fake_client):
        connector = S3Connector({"bucket": "my-bucket"})
        assert _run(connector.connect()) is True
        assert ("head_bucket", "my-bucket") in fake_client.calls

    def test_connect_missing_bucket_raises(self, fake_client):
        from botocore.exceptions import ClientError

        connector = S3Connector({"bucket": "no-such-bucket"})
        with pytest.raises(ClientError):
            _run(connector.connect())

    def test_upload_download_roundtrip(self, fake_client, tmp_path):
        connector = S3Connector({})
        assert _run(connector.connect()) is True

        local = tmp_path / "data.geojson"
        local.write_bytes(b'{"type": "FeatureCollection"}')

        key = _run(connector.upload_file(str(local), "data/data.geojson"))
        assert key == "data/data.geojson"
        assert fake_client.objects["data/data.geojson"] == local.read_bytes()

        out = tmp_path / "out.geojson"
        result = _run(connector.download_file("data/data.geojson", str(out)))
        assert result == str(out)
        assert out.read_bytes() == local.read_bytes()

    def test_list_files_returns_real_keys(self, fake_client):
        connector = S3Connector({})
        assert _run(connector.connect()) is True
        fake_client.objects = {f"prefix/file_{i}.geojson": b"" for i in range(3)}
        fake_client.objects["other/file.txt"] = b""

        keys = _run(connector.list_files("prefix/"))
        assert sorted(keys) == [
            "prefix/file_0.geojson",
            "prefix/file_1.geojson",
            "prefix/file_2.geojson",
        ]
        assert _run(connector.list_files("missing/")) == []

    def test_delete_file(self, fake_client):
        connector = S3Connector({})
        assert _run(connector.connect()) is True
        fake_client.objects["data/old.geojson"] = b""
        assert _run(connector.delete_file("data/old.geojson")) is True
        assert "data/old.geojson" not in fake_client.objects

    def test_disconnect_requires_reconnect(self, fake_client):
        connector = S3Connector({})
        assert _run(connector.connect()) is True
        _run(connector.disconnect())
        with pytest.raises(NotConnectedError):
            _run(connector.list_files())


# ---------------------------------------------------------------------------
# GCSConnector / AzureConnector: explicit placeholders
# ---------------------------------------------------------------------------


class TestGCSConnector:
    def test_init(self):
        connector = GCSConnector({"bucket": "gcs-bucket", "project": "my-project"})
        assert connector.bucket == "gcs-bucket"
        assert connector.project == "my-project"

    def test_operations_raise_clear_error(self):
        connector = GCSConnector({})
        with pytest.raises(RuntimeError, match="google-cloud-storage"):
            _run(connector.connect())
        with pytest.raises(RuntimeError, match="google-cloud-storage"):
            _run(connector.upload_file("/local/f.tif", "remote/f.tif"))
        with pytest.raises(RuntimeError, match="google-cloud-storage"):
            _run(connector.list_files())
        with pytest.raises(RuntimeError, match="google-cloud-storage"):
            _run(connector.delete_file("remote/f.tif"))


class TestAzureConnector:
    def test_init(self):
        connector = AzureConnector(
            {"container": "az-container", "account_name": "myaccount"}
        )
        assert connector.container == "az-container"
        assert connector.account_name == "myaccount"

    def test_operations_raise_clear_error(self):
        connector = AzureConnector({})
        with pytest.raises(RuntimeError, match="azure-storage-blob"):
            _run(connector.connect())
        with pytest.raises(RuntimeError, match="azure-storage-blob"):
            _run(connector.upload_file("/local/f.parquet", "remote/f.parquet"))
        with pytest.raises(RuntimeError, match="azure-storage-blob"):
            _run(connector.list_files())
        with pytest.raises(RuntimeError, match="azure-storage-blob"):
            _run(connector.delete_file("remote/f.parquet"))
