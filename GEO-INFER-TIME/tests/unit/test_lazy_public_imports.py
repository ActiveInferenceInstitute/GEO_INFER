"""Scalar UTC boundary imports must not load analytical dependencies."""

import subprocess
import sys

import geo_infer_time
import geo_infer_time.core


def test_scalar_normalization_imports_without_analytical_dependencies():
    script = """
import importlib.abc
import sys
class RejectAnalyticalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"numpy", "pandas", "scipy", "statsmodels", "sklearn", "matplotlib"}:
            raise AssertionError("Unexpected analytical dependency: " + fullname)
sys.meta_path.insert(0, RejectAnalyticalImports())
from geo_infer_time import normalize_timestamp
from geo_infer_time.core import normalize_timestamp as core_normalize
from geo_infer_time.core.timestamps import normalize_timestamp as canonical
assert normalize_timestamp is core_normalize is canonical
assert canonical("2023-12-31T16:00:00-08:00").isoformat() == "2024-01-01T00:00:00+00:00"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_public_exports_resolve_to_real_components():
    for package in [geo_infer_time, geo_infer_time.core]:
        for name in package.__all__:
            assert getattr(package, name) is not None
            assert name in dir(package)
