"""
Tests for the gpu_acceleration module.

These tests exercise the CPU fallback paths, which run on every machine
regardless of whether cupy/torch GPU backends are installed.
"""

import numpy as np
import pytest

from geo_infer_math import GPUAccelerator, get_gpu_info, is_gpu_available
from geo_infer_math.core.gpu_acceleration import GPUAccelerator as _GPUAccelerator


class TestGPUAccelerator:
    """Test GPUAccelerator CPU fallback behavior."""

    def test_instantiates_without_gpu(self):
        acc = GPUAccelerator()
        assert isinstance(acc.gpu_available, bool)
        assert isinstance(acc.backends, dict)

    def test_cpu_matrix_multiply_fallback(self):
        acc = GPUAccelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.eye(2)
        with pytest.warns(UserWarning, match="GPU acceleration not available"):
            results = acc.accelerate_matrix_operations([a, b], operation="multiply")
        assert len(results) == 1
        np.testing.assert_allclose(results[0], a @ b)

    def test_cpu_add_and_transpose_fallback(self):
        acc = GPUAccelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.ones((2, 2))
        with pytest.warns(UserWarning, match="GPU acceleration not available"):
            added = acc.accelerate_matrix_operations([a, b], operation="add")
            transposed = acc.accelerate_matrix_operations([a], operation="transpose")
        np.testing.assert_allclose(added[0], a + b)
        np.testing.assert_allclose(transposed[0], a.T)

    def test_unsupported_operation_returns_empty(self):
        acc = GPUAccelerator()
        a = np.eye(2)
        with pytest.warns(UserWarning, match="GPU acceleration not available"):
            results = acc.accelerate_matrix_operations([a], operation="determinant")
        assert results == []

    def test_performance_info_structure(self):
        acc = GPUAccelerator()
        info = acc.get_performance_info()
        assert isinstance(info, dict)


class TestGPUHelpers:
    """Test module-level GPU helper functions."""

    def test_is_gpu_available_bool(self):
        assert isinstance(is_gpu_available(), bool)

    def test_get_gpu_info_dict(self):
        info = get_gpu_info()
        assert isinstance(info, dict)

    def test_package_exports_match(self):
        assert _GPUAccelerator is GPUAccelerator


def test_math_import_defers_gpu_runtime_initialization():
    """Core spatial math imports do not probe accelerator backends."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; import geo_infer_math; assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules; assert 'cupy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
