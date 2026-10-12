"""The visualization export is resolved only when a caller selects it."""

import os
import subprocess
import sys


def test_core_import_without_folium_and_selected_export_fails_clearly():
    script = """
import importlib.abc, sys
class MissingFolium(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'folium':
            raise ModuleNotFoundError('test excludes folium', name='folium')
sys.meta_path.insert(0, MissingFolium())
import geo_infer_iot
assert 'geo_infer_iot.utils.visualization' not in sys.modules
assert 'IoTVisualization' in geo_infer_iot.__all__
try:
    geo_infer_iot.IoTVisualization
except ImportError as error:
    assert 'geo-infer-iot[visualization]' in str(error), error
else:
    raise AssertionError('unavailable visualization extra did not fail')
"""
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
