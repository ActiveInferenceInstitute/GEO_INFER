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
        "geo_infer_api",
        (),
        """
import json, subprocess, sys
from pathlib import Path
from geo_infer_api.agent_runtime import AgentSession
session = AgentSession()
inventory = session.execute({'op':'modules'})
assert inventory['ok'] and len(inventory['result']) == 45
assert [row['module'] for row in inventory['result'] if row['installed']] == ['API']
assert session.execute({'op':'catalog','module':'API'})['result']['total'] > 50
assert len(session.execute({'op':'skills'})['result']) == 46
assert session.execute({'op':'skill','module':'SPACE'})['result']['name'] == 'SPACE'
point = session.execute({'op':'call','target':'geo_infer_api:Point','kwargs':{'coordinates':[1,2]}})
assert point == {'ok':True,'result':{'type':'Point','coordinates':[1.,2.]}}
cli = Path(sys.executable).with_name('geo-infer.exe' if sys.platform == 'win32' else 'geo-infer')
result = subprocess.run([str(cli),'schema'], capture_output=True, text=True, check=True, timeout=30)
assert 'oneOf' in json.loads(result.stdout)['result']
assert not session.execute({'op':'call','target':'geo_infer_space:latlng_to_cell','args':[0,0,1]})['ok']
""",
    ),
    WheelProfile(
        "geo_infer_api",
        ("mcp",),
        """
import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
async def check():
    parameters = StdioServerParameters(command=sys.executable,args=['-I','-m','geo_infer_api.agent_mcp'])
    async with stdio_client(parameters) as (reader,writer), ClientSession(reader,writer) as client:
        await client.initialize()
        tools = await client.list_tools()
        assert len(tools.tools) == 11
        result = await client.call_tool('geo_call',{'target':'geo_infer_api:Point','kwargs':{'coordinates':[1,2]}})
        assert not result.isError and result.structuredContent['result']['coordinates'] == [1.,2.]
        error = await client.call_tool('geo_call',{'target':'os:system'})
        assert error.isError
        resource = await client.read_resource('geo-infer://schema')
        assert resource.contents[0].text
        skill = await client.read_resource('geo-infer://skills/SPACE')
        assert skill.contents[0].text
asyncio.run(check())
""",
    ),
    WheelProfile(
        "geo_infer_api",
        ("scientific",),
        """
import numpy as np
from geo_infer_api.agent_runtime import AgentSession
session = AgentSession()
array = session.decode({'$type':'ndarray','data':[[1,2],[3,4]],'dtype':'float64'})
np.testing.assert_array_equal(array.sum(axis=0),[4,6])
frame = session.decode({'$type':'dataframe','data':[[1,2]],'columns':['a','b']})
assert frame['b'].sum() == 2
series = session.decode({'$type':'dataarray','data':[100,110,120],'dims':['time'],'coords':{'time':[2020,2021,2022]}})
assert float(series.mean()) == 110
geometry = session.decode({'$type':'geometry','data':{'type':'Point','coordinates':[1,2]}})
assert geometry.x == 1 and geometry.y == 2
handle = session.encode(frame)
assert session.decode(handle) is frame
""",
    ),
    WheelProfile(
        "geo_infer_act",
        ("backends",),
        """
import numpy as np
import jax.numpy as jnp
from geo_infer_act.core.generative_model import GenerativeModel
model = GenerativeModel('categorical', {'state_dim': 2, 'random_seed': 7})
result = model.integrate_bayeux(lambda x: -0.5*jnp.sum(((x-1.5)/0.7)**2), {'x':jnp.array([1.5])}, backend='bayeux', n_samples=100, warmup=100)
assert result['backend'] == 'bayeux' and result['diagnostics']['sampler'] == 'numpyro_nuts'
samples = result['posterior_samples']['x']
assert samples.size == 100 and np.isfinite(samples).all()
np.testing.assert_allclose(samples.mean(), 1.5, atol=.2, rtol=0)
np.testing.assert_allclose(samples.var(), .49, atol=.2, rtol=0)
local = model.integrate_rxinfer('', {'observations':[1,2,3], 'prior_mean':4, 'prior_precision':2, 'measurement_precision':.5}, backend='local_gaussian')
assert local['backend'] == 'local_gaussian'
np.testing.assert_allclose(local['posterior_marginals']['mean'], 11/3.5, rtol=1e-12)
np.testing.assert_allclose(local['posterior_marginals']['variance'], 1/3.5, rtol=1e-12)
""",
    ),
    WheelProfile(
        "geo_infer_time",
        (),
        """
from geo_infer_time.core.visualization import TemporalVisualization
try:
    TemporalVisualization().plot_timeseries([0., 2.])
except RuntimeError as exc:
    assert 'geo-infer-time[visualization]' in str(exc)
else:
    raise AssertionError('Base installation unexpectedly includes matplotlib')
""",
    ),
    WheelProfile(
        "geo_infer_time",
        ("visualization",),
        """
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from geo_infer_time.core.visualization import TemporalVisualization
axis = ['2026-10-01T00:00:00Z', '2026-09-30T18:00:00-07:00']
figure = TemporalVisualization().plot_timeseries([0., 2.], timestamps=axis, save_path=Path('series.png'))
np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [0., 2.])
assert Path('series.png').stat().st_size > 0
plt.close(figure)
""",
    ),
    WheelProfile(
        "geo_infer_space",
        (),
        """
import h3
import numpy as np
import pandas as pd
from geo_infer_space import H3StateSpace, align_h3_observations
cells = sorted(h3.grid_disk(h3.latlng_to_cell(41.75, -124.2, 8), 1), reverse=True)[:2]
axis = ['2026-10-01T00:00:00Z', '2026-10-01T01:00:00Z']
source = pd.DataFrame({'cell': [cells[1], cells[0]], 'timestamp': [axis[1], '2026-09-30T17:00:00-07:00'], 'value': [9., 0.]})
original = source.copy(deep=True)
series = align_h3_observations(source, state_space=H3StateSpace(cells), timestamps=axis)
expected = pd.DataFrame([[0., np.nan], [np.nan, 9.]], columns=cells, index=pd.DatetimeIndex(axis))
pd.testing.assert_frame_equal(series.to_dataframe(), expected)
pd.testing.assert_frame_equal(source, original)
source.loc[0, 'value'] = 99.
pd.testing.assert_frame_equal(series.to_dataframe(), expected)
""",
    ),
    WheelProfile(
        "geo_infer_spm",
        ("bayesian",),
        """
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from geo_infer_spm.models.data_models import SPMData
from geo_infer_spm.core.bayesian import BayesianSPM
x = pt.dvector('x')
compiled = pytensor.function([x], x*x+3*x, mode=Mode(linker='c', optimizer='fast_run'))
np.testing.assert_array_equal(compiled(np.array([-2., 0., 4.])), [-2., 0., 28.])
assert type(compiled.maker.linker).__name__ == 'CLinker'
response = 1.5 + np.tile(np.array([-.15, -.05, .05, .15]), 6)
data = SPMData(data=response, coordinates=np.column_stack([np.linspace(0, 1, response.size), np.zeros(response.size)]))
model = BayesianSPM()
result = model.fit_bayesian_glm(data, np.ones((response.size, 1)), n_samples=100, n_tune=100, random_seed=83, chains=2, cores=1)
assert result.model_diagnostics['method'] == 'Bayesian_GLM_PyMC'
np.testing.assert_allclose(result.beta_coefficients, [1.5], atol=.05, rtol=0)
assert model.posterior_samples['beta'].shape == (200, 1)
assert np.isfinite(model.posterior_samples['sigma']).all()
""",
    ),
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
import pandas as pd
import geo_infer_data.utils.duckdb_spatial as backend
def forbidden_fallback(*args, **kwargs):
    raise AssertionError('installed native acceptance entered fallback')
backend._fallback_read_vector = forbidden_fallback
axis = pd.DatetimeIndex(['2026-11-01T08:30:00.123456789Z', '2026-11-01T09:30:00.123456790Z', '2026-11-01T09:30:00.123456791Z'])
index = pd.Index(['sensor-z', 'sensor-a', 'sensor-q'], name='observation_identity')
source = gpd.GeoDataFrame({'value': [0., -2., 7.], 'timestamp': axis, 'local': axis.tz_convert('America/Los_Angeles'), 'category': pd.Categorical(['z', 'a', 'z'], categories=['unused', 'z', 'a'], ordered=True), 'label': pd.array(['first', None, 'last'], dtype='string'), 'nullable': pd.array([0., None, 7.], dtype='Float64')}, geometry=[Point(500000, 4649776), None, Point(500100, 4649800)], crs='EPSG:32610', index=index).rename_geometry('shape"; DROP TABLE pretend; --')
source.to_parquet('observations.parquet')
actual = read_cloud_native_vector('observations.parquet', require_duckdb=True)
pd.testing.assert_frame_equal(actual, source)
assert actual.timestamp.iloc[2].value - actual.timestamp.iloc[1].value == 1
actual.iloc[0, 0] = 99.
pd.testing.assert_frame_equal(read_cloud_native_vector('observations.parquet', require_duckdb=True), source)
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
import numpy as np
from pathlib import Path
from geo_infer_art import GeoArt
from geo_infer_art.core.aesthetics.style_transfer import StyleTransfer
try:
    StyleTransfer()
except ImportError as exc:
    assert 'neural' in str(exc)
else:
    raise AssertionError('Base installation unexpectedly includes TensorFlow')
data = np.arange(48, dtype=float).reshape(3, 4, 4)
resized = GeoArt(data, metadata={'raster_layout':'band_first'}).optimize_for_performance((2, 2))
assert resized.data.shape == (3, 2, 2)
np.testing.assert_allclose(resized.data[0], [[2.5,4.5],[10.5,12.5]])
with resized:
    resized.apply_style(raster_band=0).save('resampled.png')
assert Path('resampled.png').stat().st_size > 100
""",
    ),
    WheelProfile(
        "geo_infer_math",
        (),
        """
import math
import numpy as np
from geo_infer_math.core.information_theory.channel_capacity import mimo_channel_capacity, waterfilling_power_allocation
from geo_infer_math.core.information_theory.entropy import spatial_entropy
assert mimo_channel_capacity(np.eye(2), 1, 0) == 0
assert math.isclose(mimo_channel_capacity(np.eye(2), 1, 10), 2*math.log2(6), rel_tol=1e-12)
allocation, rate = waterfilling_power_allocation([0,4,1],1,2)
np.testing.assert_allclose(allocation,[0,1.375,.625])
assert math.isclose(rate,math.log2(6.5)+math.log2(1.625),rel_tol=1e-12)
assert math.isclose(spatial_entropy([[0,0],[0,1],[1,0],[1,1]],bins=2,base=math.e),math.log(4),rel_tol=1e-12)
""",
    ),
    WheelProfile(
        "geo_infer_app",
        (),
        """
import json
from datetime import date
from geo_infer_app import AgentState, AgentType, AgentVisualization
state=AgentState('wheel',AgentType.BDI,'active',beliefs={'observed':date(2026,10,18)})
result=AgentVisualization.state_to_dashboard_data(state)
assert json.loads(json.dumps(result,allow_nan=False))['widgets']['beliefs']['value']['observed']=='2026-10-18'
state.beliefs['confidence']=float('nan')
try:
    AgentVisualization.state_to_dashboard_data(state)
except ValueError as error:
    assert 'finite JSON' in str(error)
else:
    raise AssertionError('Nonfinite dashboard payload was accepted')
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
        ("california",),
        """
from pathlib import Path
import hashlib
import stat
import sys
import sysconfig
from geo_infer_place.locations.california import contracts

site = Path(sysconfig.get_paths()['purelib']).resolve()
assert Path(contracts.__file__).resolve().is_relative_to(site)
owner_paths = (
    'GEO-INFER-PLACE/src/geo_infer_place/locations/california/__init__.py',
    'GEO-INFER-PLACE/src/geo_infer_place/locations/california/acquisition.py',
    'GEO-INFER-PLACE/src/geo_infer_place/locations/california/contracts.py',
    'GEO-INFER-PLACE/src/geo_infer_place/locations/california/geometry.py',
    'GEO-INFER-PLACE/src/geo_infer_place/locations/california/inference.py',
    'GEO-INFER-PLACE/src/geo_infer_place/locations/california/models.py',
    'GEO-INFER-PLACE/src/geo_infer_place/locations/california/monitoring.py',
    'GEO-INFER-PLACE/src/geo_infer_place/locations/california/report.py',
    'GEO-INFER-ACT/src/geo_infer_act/core/gnn_factored_batch.py',
    'GEO-INFER-ACT/src/geo_infer_act/core/gnn_factored_contract.py',
    'GEO-INFER-ACT/src/geo_infer_act/core/resource_allocation.py',
    'GEO-INFER-SPACE/src/geo_infer_space/core/graph_statistics.py',
    'GEO-INFER-PLACE/src/geo_infer_place/core/regional_layers.py',
    'GEO-INFER-PLACE/src/geo_infer_place/core/_regional_download_worker.py',
)
expected = {
    key: hashlib.sha256(site.joinpath(*key.split('/')[2:]).read_bytes()).hexdigest()
    for key in owner_paths
}
assert contracts.execution_binding().owners == expected
for name in ('geo_infer_act', 'geo_infer_space', 'geo_infer_place'):
    assert Path(sys.modules[name].__file__).resolve().is_relative_to(site)

# The wheel driver gives each operation profile its own disposable environment.
# A missing installed owner must fail, even after its package was imported.
target = site / 'geo_infer_act/core/gnn_factored_batch.py'
assert target.resolve().is_relative_to(site) and not target.is_symlink()
original = target.read_bytes()
mode = stat.S_IMODE(target.stat().st_mode)
try:
    target.unlink()
    try:
        contracts.execution_binding()
    except FileNotFoundError as error:
        assert Path(error.filename).resolve() == target.resolve()
    else:
        raise AssertionError('Missing installed ACT owner resource was accepted')
finally:
    target.write_bytes(original)
    target.chmod(mode)
assert target.read_bytes() == original and stat.S_IMODE(target.stat().st_mode) == mode
assert contracts.execution_binding().owners == expected
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
from pathlib import Path
from geo_infer_place.core.cascadia_boundary import verify_cascadia_boundary
from geo_infer_place.core.bioregion_visualization import create_bioregion_map
config = Path(str(resource.parent))
receipt = verify_cascadia_boundary(config)
assert receipt['native_ids'] == [1] and receipt['feature_count'] == 1
create_bioregion_map(config, {}, Path('cascadia-boundary.html'))
assert 'CC BY-NC-SA 4.0' in Path('cascadia-boundary.html').read_text(encoding='utf-8')
manager = EnhancedConfigManager()
assert manager.config.bioregion['name'] == 'Cascadia Bioregion'
assert manager.config.visualization.default_center == [45.5, -122.5]
assert not __import__('pathlib').Path('config').exists()
""",
    ),
)
