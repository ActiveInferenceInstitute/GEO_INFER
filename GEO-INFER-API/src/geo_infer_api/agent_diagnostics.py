"""Import-free runtime facts for agents selecting available execution profiles."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import importlib.util
import os
import platform
import sys
import sysconfig
from typing import Any

from geo_infer_api.agent_catalog import module_inventory


def runtime_diagnostics() -> dict[str, Any]:
    """Report actual interpreter/dependency facts without importing backends.

    Availability is not qualification: sampling permissions, native compilers,
    credentials and external services are never silently probed or asserted.
    """
    dependencies = {}
    for package in (
        "mcp",
        "numpy",
        "pandas",
        "xarray",
        "geopandas",
        "shapely",
        "pytensor",
        "pymc",
        "jax",
        "jaxlib",
        "rpds-py",
        "psutil",
    ):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    jit = getattr(sys, "_jit", None)
    return {
        "python": {
            "version": platform.python_version(),
            "full_version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "gil_disabled_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "jit_available": bool(jit and jit.is_available()),
            "jit_enabled": bool(jit and jit.is_enabled()),
        },
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "profiling": {
            "tracing_available": True,
            "sampling_available": importlib.util.find_spec("profiling") is not None,
            "sampling_permissions": "not_probed",
        },
        "native": {
            "pytensor_cxx_explicit": any(
                v.strip().startswith("cxx=")
                for v in os.environ.get("PYTENSOR_FLAGS", "").split(",")
            ),
            "compiler_qualification": "not_probed",
        },
        "dependencies": dependencies,
        "modules": module_inventory(),
    }
