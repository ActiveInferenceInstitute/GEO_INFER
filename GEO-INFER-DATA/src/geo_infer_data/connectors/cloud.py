"""
Cloud connectors for GEO-INFER-DATA.

This module provides cloud storage connectivity for AWS S3 (via ``boto3``)
and explicit, non-fabricating placeholders for Google Cloud Storage and Azure
Blob Storage, whose client libraries are not declared dependencies of this
package.

The S3 connector performs real uploads, downloads, listings, and deletions
against the configured bucket. Blocking ``boto3`` calls are executed through
``asyncio.to_thread`` so the async connector surface does not block the event
loop.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import boto3


logger = logging.getLogger(__name__)


class NotConnectedError(RuntimeError):
    """Raised when an operation requires an established cloud connection."""


class CloudConnector:
    """
    Base class for cloud storage connectors.

    This abstract base class defines the interface for connecting to cloud
    storage services including AWS S3, Google Cloud Storage, and Azure Blob
    Storage. Subclasses implement the operations against their real client
    libraries or raise :class:`RuntimeError` when the backend dependency is
    not part of this package's declared dependencies.
    """

    async def connect(self) -> bool:
        """
        Establish connection to cloud storage.

        Returns:
            True if connection successful
        """
        raise RuntimeError("Cloud connector subclasses must implement connect()")

    async def upload_file(self, local_path: str, remote_path: str) -> str:
        """
        Upload file to cloud storage.

        Args:
            local_path: Local file path
            remote_path: Remote storage path

        Returns:
            Remote path of uploaded file
        """
        raise RuntimeError("Cloud connector subclasses must implement upload_file()")

    async def download_file(self, remote_path: str, local_path: str) -> str:
        """
        Download file from cloud storage.

        Args:
            remote_path: Remote storage path
            local_path: Local file path

        Returns:
            Local path of downloaded file
        """
        raise RuntimeError("Cloud connector subclasses must implement download_file()")

    async def list_files(self, prefix: str = "") -> list[str]:
        """
        List files in cloud storage.

        Args:
            prefix: Path prefix to filter files

        Returns:
            List of file paths
        """
        raise RuntimeError("Cloud connector subclasses must implement list_files()")

    async def delete_file(self, remote_path: str) -> bool:
        """
        Delete file from cloud storage.

        Args:
            remote_path: Remote storage path

        Returns:
            True if deletion successful
        """
        raise RuntimeError("Cloud connector subclasses must implement delete_file()")

    async def read_byte_range(
        self, remote_path: str, start_byte: int, end_byte: int
    ) -> bytes:
        """
        Read a byte range from a remote file (HTTP/Cloud range request).

        Args:
            remote_path: Path or key of the remote resource
            start_byte: 0-indexed start offset
            end_byte: 0-indexed end offset (inclusive)

        Returns:
            Extracted byte buffer
        """
        if start_byte < 0 or end_byte < start_byte:
            raise ValueError("Invalid byte range coordinates")
        path = Path(remote_path)
        if path.exists():
            with path.open("rb") as f:
                f.seek(start_byte)
                return f.read(end_byte - start_byte + 1)
        raise FileNotFoundError(
            f"Remote resource not available for range reading: {remote_path}"
        )

    async def disconnect(self) -> None:
        """Close cloud storage connection."""
        logger.debug(
            "%s has no persistent cloud connection to close", type(self).__name__
        )


class S3Connector(CloudConnector):
    """
    AWS S3 cloud storage connector backed by ``boto3``.

    The connector talks to real S3 (or any S3-compatible endpoint via the
    ``endpoint_url`` config key). ``connect()`` verifies bucket access with a
    ``head_bucket`` call; subsequent operations delegate to the boto3 client.

    Config keys:
        bucket: Target bucket (default ``geo-infer-data``).
        region: AWS region (default ``us-east-1``).
        access_key / secret_key: Static credentials. When omitted, boto3's
            default credential chain (environment, profile, IAM role) is used.
        endpoint_url: Custom S3-compatible endpoint (e.g. MinIO).
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.bucket = config.get("bucket", "geo-infer-data")
        self.region = config.get("region", "us-east-1")
        self.access_key = config.get("access_key")
        self.secret_key = config.get("secret_key")
        self.endpoint_url = config.get("endpoint_url")
        self._client: Any = None

        logger.info(f"Initialized S3Connector for bucket: {self.bucket}")

    def _create_client(self) -> Any:
        """Create the boto3 S3 client from the connector configuration."""
        client_kwargs: dict[str, Any] = {"region_name": self.region}
        if self.access_key and self.secret_key:
            client_kwargs["aws_access_key_id"] = self.access_key
            client_kwargs["aws_secret_access_key"] = self.secret_key
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        return boto3.client("s3", **client_kwargs)

    def _require_client(self) -> Any:
        """Return the connected boto3 client or raise."""
        if self._client is None:
            raise NotConnectedError(
                "S3Connector is not connected; call connect() first"
            )
        return self._client

    def _list_keys(self, prefix: str) -> list[str]:
        """List object keys under ``prefix`` using pagination."""
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    async def connect(self) -> bool:
        """
        Connect to AWS S3 and verify the bucket is accessible.

        Returns:
            True if connection successful

        Raises:
            botocore.exceptions.ClientError: If credentials are rejected or
                the bucket does not exist / is not accessible.
        """
        client = await asyncio.to_thread(self._create_client)
        await asyncio.to_thread(client.head_bucket, Bucket=self.bucket)
        self._client = client
        logger.info(f"Connected to S3 bucket: {self.bucket}")
        return True

    async def upload_file(self, local_path: str, remote_path: str) -> str:
        """
        Upload a local file to S3.

        Args:
            local_path: Local file path
            remote_path: S3 key path

        Returns:
            S3 key of the uploaded file
        """
        client = self._require_client()
        await asyncio.to_thread(
            client.upload_file, str(local_path), self.bucket, remote_path
        )
        logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{remote_path}")
        return remote_path

    async def download_file(self, remote_path: str, local_path: str) -> str:
        """
        Download an S3 object to a local file.

        Args:
            remote_path: S3 key path
            local_path: Local destination path

        Returns:
            Local path of the downloaded file
        """
        client = self._require_client()
        await asyncio.to_thread(
            client.download_file, self.bucket, remote_path, str(local_path)
        )
        logger.info(f"Downloaded s3://{self.bucket}/{remote_path} to {local_path}")
        return local_path

    async def list_files(self, prefix: str = "") -> list[str]:
        """
        List object keys in the bucket under ``prefix``.

        Args:
            prefix: Key prefix to filter files

        Returns:
            List of S3 keys (may be empty when nothing matches)
        """
        self._require_client()
        keys = await asyncio.to_thread(self._list_keys, prefix)
        logger.info(f"Listed {len(keys)} files in s3://{self.bucket}/{prefix}")
        return keys

    async def delete_file(self, remote_path: str) -> bool:
        """
        Delete an object from the bucket.

        Args:
            remote_path: S3 key path

        Returns:
            True once the deletion request completed successfully
        """
        client = self._require_client()
        await asyncio.to_thread(
            client.delete_object, Bucket=self.bucket, Key=remote_path
        )
        logger.info(f"Deleted s3://{self.bucket}/{remote_path}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from S3 (the boto3 client holds no persistent socket)."""
        self._client = None
        logger.info("S3 connection closed")


class _UnavailableCloudConnector(CloudConnector):
    """
    Base for cloud backends whose client libraries are not declared deps.

    Subclasses raise a clear :class:`RuntimeError` on every operation instead
    of silently returning fabricated results. The backend can be enabled by
    declaring its client library and implementing the operations.
    """

    backend: str = "cloud"
    dependency: str = ""

    def _unavailable(self, operation: str) -> RuntimeError:
        return RuntimeError(
            f"{type(self).__name__} does not implement {operation}: the "
            f"{self.backend} client library ('{self.dependency}') is not a "
            "declared dependency of geo-infer-data. Use S3Connector (boto3) "
            "or the local/Redis storage backends instead."
        )

    async def connect(self) -> bool:
        raise self._unavailable("connect()")

    async def upload_file(self, local_path: str, remote_path: str) -> str:
        raise self._unavailable("upload_file()")

    async def download_file(self, remote_path: str, local_path: str) -> str:
        raise self._unavailable("download_file()")

    async def list_files(self, prefix: str = "") -> list[str]:
        raise self._unavailable("list_files()")

    async def delete_file(self, remote_path: str) -> bool:
        raise self._unavailable("delete_file()")


class GCSConnector(_UnavailableCloudConnector):
    """
    Google Cloud Storage connector for the not-installed-client case.

    ``google-cloud-storage`` is not a declared dependency of this package, so
    every operation raises ``RuntimeError`` naming the missing library rather
    than returning fabricated results.
    """

    backend = "Google Cloud Storage"
    dependency = "google-cloud-storage"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.bucket = config.get("bucket", "geo-infer-data")
        self.project = config.get("project", "default-project")

        logger.info(f"Initialized GCSConnector for bucket: {self.bucket}")


class AzureConnector(_UnavailableCloudConnector):
    """
    Azure Blob Storage connector for the not-installed-client case.

    ``azure-storage-blob`` is not a declared dependency of this package, so
    every operation raises ``RuntimeError`` naming the missing library rather
    than returning fabricated results.
    """

    backend = "Azure Blob Storage"
    dependency = "azure-storage-blob"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.container = config.get("container", "geo-infer-data")
        self.account_name = config.get("account_name")

        logger.info(f"Initialized AzureConnector for container: {self.container}")
