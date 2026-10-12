"""The DATA base import stays usable when integration extras are absent."""

import os
import subprocess
import sys


def test_missing_extras_fail_at_selected_operations(tmp_path):
    raster = tmp_path / "test.tif"
    raster.write_bytes(b"II*\x00" + b"\x00" * 16)
    script = f"""
import asyncio, importlib.abc, sys
missing = {{'boto3', 'rasterio', 'redis', 'minio', 'psycopg2', 'asyncpg'}}
class Missing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in missing:
            raise ModuleNotFoundError('test excludes optional extras', name=root)
sys.meta_path.insert(0, Missing())
import geo_infer_data
from geo_infer_data.connectors.cloud import S3Connector
from geo_infer_data.connectors.database import DatabaseConnector
from geo_infer_data.utils.format_detection import FormatDetector
from geo_infer_data.utils.dependencies import MissingOptionalDependency
from geo_infer_data.core.storage import MinIOBackend, RedisBackend
raster_path = {str(raster)!r}
cases = [
    ('s3', lambda: S3Connector({{}})._create_client()),
    ('raster', lambda: FormatDetector().detect_from_path(raster_path)),
    ('redis', lambda: RedisBackend({{}})),
    ('minio', lambda: asyncio.run(MinIOBackend({{'endpoint': 'localhost:9000', 'access_key': 'test', 'secret_key': 'test', 'bucket': 'test'}})._store_to_minio(None, 'test', None))),
    ('postgres', lambda: DatabaseConnector('postgresql', 'postgresql://user:pass@localhost/db')),
]
for extra, operation in cases:
    try:
        operation()
    except MissingOptionalDependency as error:
        assert f'geo-infer-data[{{extra}}]' in str(error), (extra, error)
    else:
        raise AssertionError(f'selected unavailable backend {{extra}} did not fail')
"""
    environment = dict(os.environ)
    # One fresh interpreter proves all selected boundaries, without repeatedly
    # starting native numerical thread pools for the same base import.
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
