"""Real filesystem profile controls and native GRIB/GIS process coexistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from geo_infer_test.native_io_environment import (
    NativeIOProfileError,
    VARIABLES,
    build_environment,
    installed_environment,
    write_github_environment,
)

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "GEO-INFER-TEST/native_io_profile.py"


def _layout(root: Path, directory: str = "lib64") -> tuple[Path, Path]:
    """Path-resolution inputs only; these bytes never supply native evidence."""
    codes, eckit = root / "codes", root / "eckit"
    for package, library in ((codes, "libeccodes.so"), (eckit, "libeckit_geo.so")):
        folder = package / directory
        folder.mkdir(parents=True)
        (folder / library).write_bytes(b"path resolution input")
    return codes, eckit


@pytest.mark.parametrize("directory", ["lib", "lib64"])
def test_environment_selects_exact_paths_and_preserves_caller(tmp_path, directory):
    codes, eckit = _layout(tmp_path, directory)
    original = {
        "LD_LIBRARY_PATH": "/system/one:/system/two",
        "PRIVATE_VALUE": "not-output",
    }
    result = build_environment(codes, eckit, original)
    assert set(result) == VARIABLES
    assert result["ECCODESLIB_HOME"] == str(codes.resolve())
    assert (
        result["LD_LIBRARY_PATH"]
        == f"{eckit / directory}:{codes / directory}:/system/one:/system/two"
    )
    assert result["FINDLIBS_DISABLE_PACKAGE"] == "yes"
    assert result["FINDLIBS_DISABLE_PYTHON"] == "yes"
    assert result["FINDLIBS_DISABLE_HOME"] == "no"
    assert original == {
        "LD_LIBRARY_PATH": "/system/one:/system/two",
        "PRIVATE_VALUE": "not-output",
    }
    assert "PRIVATE_VALUE" not in result


def test_environment_without_existing_search_path(tmp_path):
    codes, eckit = _layout(tmp_path)
    result = build_environment(codes, eckit, {})
    assert result["LD_LIBRARY_PATH"] == f"{eckit / 'lib64'}:{codes / 'lib64'}"


@pytest.mark.parametrize("missing", ["codes", "eckit"])
def test_missing_layout_refuses_configuration(tmp_path, missing):
    codes, eckit = _layout(tmp_path)
    with pytest.raises(NativeIOProfileError, match="exactly one"):
        build_environment(
            tmp_path / "absent" if missing == "codes" else codes,
            tmp_path / "absent" if missing == "eckit" else eckit,
            {},
        )


def test_ambiguous_layout_refuses_configuration(tmp_path):
    codes, eckit = _layout(tmp_path)
    (codes / "lib").mkdir()
    (codes / "lib/libeccodes.so").write_bytes(b"second path-resolution input")
    with pytest.raises(NativeIOProfileError, match="exactly one"):
        build_environment(codes, eckit, {})


@pytest.mark.parametrize("unsafe", [":", "\n", "\r"])
def test_unrepresentable_library_root_refuses_configuration(tmp_path, unsafe):
    with pytest.raises(NativeIOProfileError, match="separators"):
        build_environment(tmp_path / ("unsafe" + unsafe), tmp_path, {})


@pytest.mark.parametrize("unsafe", ["one\ntwo", "one\rtwo", "one\0two"])
def test_multiline_existing_search_refuses_configuration(tmp_path, unsafe):
    codes, eckit = _layout(tmp_path)
    with pytest.raises(NativeIOProfileError, match="separator"):
        build_environment(codes, eckit, {"LD_LIBRARY_PATH": unsafe})


def test_github_environment_appends_complete_bindings_and_rejects_injection(tmp_path):
    codes, eckit = _layout(tmp_path)
    bindings = build_environment(codes, eckit, {})
    output = tmp_path / "github-env"
    output.write_text("EARLIER=value\n")
    write_github_environment(output, bindings)
    expected = "EARLIER=value\n" + "".join(
        f"{key}={bindings[key]}\n" for key in sorted(bindings)
    )
    assert output.read_text() == expected
    for changed in (
        {key: value for key, value in bindings.items() if key != "ECCODESLIB_HOME"},
        {**bindings, "EXTRA": "value"},
        {**bindings, "ECCODESLIB_HOME": "bad\nINJECTED=value"},
    ):
        with pytest.raises(NativeIOProfileError):
            write_github_environment(output, changed)
        assert output.read_text() == expected
    with pytest.raises(IsADirectoryError):
        write_github_environment(tmp_path, bindings)


def test_installed_configuration_is_explicit_about_platform():
    if sys.platform == "linux":
        result = installed_environment(os.environ)
        assert set(result) == VARIABLES
        assert Path(result["ECCODESLIB_HOME"]).is_dir()
    else:
        with pytest.raises(NativeIOProfileError, match="Linux-only"):
            installed_environment(os.environ)


def test_launcher_arguments_and_real_process_status(tmp_path):
    def invoke(*args):
        return subprocess.run(
            [sys.executable, "-B", str(LAUNCHER), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert invoke().returncode == 2
    assert invoke("--describe", "--", sys.executable).returncode == 2
    described = invoke("--describe")
    if sys.platform != "linux":
        assert described.returncode == 2 and "Linux-only" in described.stderr
        return
    assert described.returncode == 0, described.stderr
    assert json.loads(described.stdout)["execution_verified"] is False
    assert invoke("--", sys.executable, "-c", "raise SystemExit(7)").returncode == 7
    assert invoke("--", str(tmp_path / "missing-program")).returncode == 127
    program = tmp_path / "not-executable"
    program.write_text("#!/bin/sh\nexit 0\n")
    program.chmod(0o600)
    assert invoke("--", str(program)).returncode == 126
    assert invoke("--github-env", str(tmp_path)).returncode == 2
    env_file = tmp_path / "ci-env"
    assert invoke("--github-env", str(env_file)).returncode == 0
    assert len(env_file.read_text().splitlines()) == len(VARIABLES)


NATIVE_IO_BODY = r"""
import json
import math
from pathlib import Path
import sys
if sys.argv[2] == 'eccodes-first':
    import eccodes
    import rasterio
else:
    import rasterio
    import eccodes
import h5py
import netCDF4
import numpy as np
import pyogrio
import pyproj
import xarray as xr
from rasterio.transform import from_origin

root = Path(sys.argv[1])
root.mkdir(exist_ok=True)
engines = xr.backends.list_engines()
assert {'cfgrib', 'rasterio', 'netcdf4'} <= set(engines)
handle = eccodes.codes_grib_new_from_samples('regular_ll_sfc_grib2')
assert handle is not None
try:
    for name, value in {'Ni': 2, 'Nj': 2, 'latitudeOfFirstGridPointInDegrees': 1, 'latitudeOfLastGridPointInDegrees': 0, 'longitudeOfFirstGridPointInDegrees': 0, 'longitudeOfLastGridPointInDegrees': 1, 'iDirectionIncrementInDegrees': 1, 'jDirectionIncrementInDegrees': 1}.items():
        eccodes.codes_set(handle, name, value)
    eccodes.codes_set_values(handle, [1., 2., 3., 4.])
    with (root / 'sample.grib2').open('wb') as stream:
        eccodes.codes_write(handle, stream)
finally:
    eccodes.codes_release(handle)
assert (root / 'sample.grib2').read_bytes().startswith(b'GRIB')
with xr.open_dataset(root / 'sample.grib2', engine='cfgrib', decode_timedelta=True, backend_kwargs={'indexpath': ''}) as dataset:
    variable = next(iter(dataset.data_vars))
    assert np.array_equal(dataset[variable].values, [[1, 2], [3, 4]])
values = np.array([[1, 2], [3, 4]], dtype='uint8')
with rasterio.open(root / 'raster.tif', 'w', driver='GTiff', height=2, width=2, count=1, dtype='uint8', crs='EPSG:4326', transform=from_origin(0, 2, 1, 1)) as dataset:
    dataset.write(values, 1)
with rasterio.open(root / 'raster.tif') as dataset:
    assert dataset.crs.to_epsg() == 4326
    assert np.array_equal(dataset.read(1), values)
with h5py.File(root / 'array.h5', 'w') as dataset:
    dataset.create_dataset('values', data=values)
with h5py.File(root / 'array.h5', 'r') as dataset:
    assert np.array_equal(dataset['values'][:], values)
data = xr.Dataset({'values': (('y', 'x'), values)})
data.to_netcdf(root / 'array.nc', engine='netcdf4')
with xr.open_dataset(root / 'array.nc', engine='netcdf4') as restored:
    assert np.array_equal(restored['values'].values, values)
x, y = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(1, 1)
angle = math.pi / 180
assert math.isclose(x, 6378137 * angle, rel_tol=0, abs_tol=1e-7)
assert math.isclose(y, 6378137 * math.log(math.tan(math.pi / 4 + angle / 2)), rel_tol=0, abs_tol=1e-7)
print(json.dumps({'engines': sorted(engines), 'grib_values': [1, 2, 3, 4], 'raster_sum': int(values.sum()), 'hdf5_and_netcdf': True, 'projection_checked': True}))
"""


@pytest.mark.parametrize("order", ["eccodes-first", "rasterio-first"])
def test_native_grib_gis_and_array_backends_coexist(tmp_path, order):
    command = [
        sys.executable,
        "-B",
        "-X",
        "faulthandler",
        "-W",
        "error",
        "-c",
        textwrap.dedent(NATIVE_IO_BODY),
        str(tmp_path),
        order,
    ]
    if sys.platform == "linux":
        command = [sys.executable, "-B", str(LAUNCHER), "--", *command]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (tmp_path / "native.stdout").write_text(result.stdout)
    (tmp_path / "native.stderr").write_text(result.stderr)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    record = json.loads(result.stdout)
    assert record["raster_sum"] == 10 and record["hdf5_and_netcdf"]
    assert record["projection_checked"]
