"""The explicit macOS compiler adapter preserves all native compiler arguments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import numpy as np


_SCRIPT = Path(__file__).resolve().parents[2] / "pytensor_clang++_macos27.py"
_SPEC = importlib.util.spec_from_file_location("spm_compiler_profile", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_profile_removes_only_obsolete_ld64_flag():
    arguments = [
        "-dynamiclib",
        "-ld64",
        "-o",
        "file with spaces.so",
        "-Lfoo",
        "-ld64x",
        "source.cpp",
    ]
    assert _MODULE.compiler_arguments(arguments, "Darwin", "27.0.1") == [
        "/usr/bin/clang++",
        "-dynamiclib",
        "-o",
        "file with spaces.so",
        "-Lfoo",
        "-ld64x",
        "source.cpp",
    ]
    assert arguments[1] == "-ld64"


@pytest.mark.parametrize(
    "system,version",
    [("Linux", "27.0"), ("Windows", "27.0"), ("Darwin", "15.0"), ("Darwin", "26.0")],
)
def test_profile_rejects_other_platforms(system, version):
    with pytest.raises(ValueError, match="macOS 27"):
        _MODULE.compiler_arguments(["--version"], system, version)


def test_native_c_linker_evaluates_independent_polynomial_oracle():
    """Both ordinary Linux and named macOS profiles retain native C execution."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor.compile.mode import Mode

    x = pt.dvector("x")
    compiled = pytensor.function(
        [x], x * x + 3 * x, mode=Mode(linker="c", optimizer="fast_run")
    )
    np.testing.assert_array_equal(
        compiled(np.array([-2.0, 0.0, 4.0])), np.array([-2.0, 0.0, 28.0])
    )
    assert type(compiled.maker.linker).__name__ == "CLinker"
