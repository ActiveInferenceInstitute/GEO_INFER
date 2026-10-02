"""Operation-level probes for independently installed base/extra wheels.

These probes use real libraries and local data. Remote services, licensed
datasets, pretrained weights, and accelerator hardware are separate evidence.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WheelProfile:
    """A separate environment, its selected extras, and an executable oracle."""

    package: str
    extras: tuple[str, ...]
    code: str

    @property
    def name(self) -> str:
        return self.package + "[" + (",".join(self.extras) or "base") + "]"


REQUIRED_PROFILES = (
    WheelProfile(
        "geo_infer_data",
        (),
        """
from geo_infer_data.utils.dependencies import require_dependency, MissingOptionalDependency
from rtree import index
tree = index.Index()
tree.insert(7, (0, 0, 1, 1))
assert list(tree.intersection((0.5, 0.5, 0.5, 0.5))) == [7]
for module, extra in [('psycopg2','postgres'), ('boto3','s3'), ('minio','minio'), ('redis','redis'), ('rasterio','raster')]:
    try:
        require_dependency(module, extra)
    except MissingOptionalDependency as exc:
        assert f'geo-infer-data[{extra}]' in str(exc)
    else:
        raise AssertionError(f'Base installation unexpectedly includes {module}')
""",
    ),
    WheelProfile(
        "geo_infer_data",
        ("postgres",),
        """
from geo_infer_data.utils.dependencies import require_dependency
assert require_dependency('psycopg2', 'postgres').extensions.adapt(42).getquoted() == b'42'
assert require_dependency('asyncpg', 'postgres').__version__
""",
    ),
    WheelProfile(
        "geo_infer_data",
        ("s3",),
        """
from geo_infer_data.utils.dependencies import require_dependency
session = require_dependency('boto3', 's3').Session()
assert 's3' in session.get_available_services()
""",
    ),
    WheelProfile(
        "geo_infer_data",
        ("minio",),
        """
from geo_infer_data.utils.dependencies import require_dependency
client = require_dependency('minio', 'minio').Minio('localhost:9000', secure=False)
assert client is not None
""",
    ),
    WheelProfile(
        "geo_infer_data",
        ("redis",),
        """
from geo_infer_data.utils.dependencies import require_dependency
client = require_dependency('redis', 'redis').Redis(host='localhost', port=6379)
assert client.connection_pool.connection_kwargs['port'] == 6379
client.close()
""",
    ),
    WheelProfile(
        "geo_infer_data",
        ("raster",),
        """
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
values = np.array([[0, 1], [2, 3]], dtype='uint8')
with MemoryFile() as memory:
    with memory.open(driver='GTiff', height=2, width=2, count=1, dtype=values.dtype, transform=from_origin(0, 2, 1, 1), crs='EPSG:4326') as dataset:
        dataset.write(values, 1)
    with memory.open() as dataset:
        np.testing.assert_array_equal(dataset.read(1), values)
""",
    ),
    WheelProfile(
        "geo_infer_data",
        ("integrations",),
        """
import geopandas as gpd
from shapely.geometry import Point
from geo_infer_data.utils.duckdb_spatial import provision_spatial_extension, read_cloud_native_vector
receipt = provision_spatial_extension()
assert receipt['status'] == 'verified' and receipt['extension'] == 'spatial'
source = gpd.GeoDataFrame({'identity': [7, 3], 'value': [0., -2.]}, geometry=[Point(-122.5, 45.5), Point(-124., 41.)], crs='EPSG:4326')
source.to_file('observations.fgb', driver='FlatGeobuf')
actual = read_cloud_native_vector('observations.fgb', require_duckdb=True).sort_values('identity').reset_index(drop=True)
expected = source.sort_values('identity').reset_index(drop=True)
assert actual.identity.tolist() == [3, 7] and actual.value.tolist() == [-2., 0.]
assert actual.crs == expected.crs and actual.geometry.equals(expected.geometry)
""",
    ),
    WheelProfile(
        "geo_infer_ops",
        (),
        """
from urllib.request import urlopen
from geo_infer_ops.core.monitoring import start_metrics_server
with start_metrics_server(0) as port:
    with urlopen(f'http://127.0.0.1:{port}/metrics', timeout=3) as response:
        assert response.status == 200 and b'# HELP' in response.read()
with start_metrics_server(port) as reopened_port:
    assert reopened_port == port, 'Cleanup must reopen the exact port, not move to another'
""",
    ),
    WheelProfile(
        "geo_infer_iot",
        (),
        """
import sys
assert 'geo_infer_iot.utils.visualization' not in sys.modules
try:
    package.IoTVisualization
except ImportError as exc:
    assert 'geo-infer-iot[visualization]' in str(exc)
else:
    raise AssertionError('Base installation unexpectedly includes visualization')
""",
    ),
    WheelProfile(
        "geo_infer_iot",
        ("visualization",),
        """
from pathlib import Path
result = package.IoTVisualization().create_sensor_map([], output_file='map.html')
assert isinstance(result, dict) and Path('map.html').is_file()
assert '<html' in Path('map.html').read_text().lower()
""",
    ),
    WheelProfile(
        "geo_infer_art",
        (),
        """
from geo_infer_art.core.aesthetics.style_transfer import StyleTransfer
try:
    StyleTransfer()
except ImportError as exc:
    assert 'neural' in str(exc)
else:
    raise AssertionError('Base installation unexpectedly includes TensorFlow')
""",
    ),
    WheelProfile(
        "geo_infer_art",
        ("neural",),
        """
import tensorflow as tf
from geo_infer_art.core.aesthetics.style_transfer import StyleTransfer
assert StyleTransfer(seed=0).model is None
with tf.device('/CPU:0'):
    assert float(tf.reduce_sum(tf.constant([1., 2.])).numpy()) == 3.
""",
    ),
    WheelProfile(
        "geo_infer_cog",
        (),
        """
from geo_infer_cog.api.rest_api import create_cog_api_app
try:
    create_cog_api_app()
except ImportError as exc:
    assert 'geo-infer-cog[api]' in str(exc)
else:
    raise AssertionError('Base installation unexpectedly includes Flask')
""",
    ),
    WheelProfile(
        "geo_infer_cog",
        ("api",),
        """
from geo_infer_cog.api.rest_api import create_cog_api_app
response = create_cog_api_app().test_client().get('/health')
assert response.status_code == 200 and all(response.json['components'].values())
""",
    ),
    WheelProfile(
        "geo_infer_place",
        ("cascadia",),
        """
from importlib.resources import files
from geo_infer_place.locations.cascadia.cli import main
from geo_infer_place.locations.cascadia.server import create_app
from geo_infer_place.locations.cascadia.core.enhanced_config import EnhancedConfigManager
assert callable(main) and callable(create_app)
resource = files('geo_infer_place.locations.cascadia.config').joinpath('cascadia_config.yaml')
assert resource.is_file() and resource.read_text()
manager = EnhancedConfigManager()
assert manager.config.bioregion['name'] == 'Cascadia Bioregion'
assert manager.config.visualization.default_center == [45.5, -122.5]
assert not __import__('pathlib').Path('config').exists()
""",
    ),
)
